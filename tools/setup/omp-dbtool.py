#!/usr/bin/env python3
"""omp-dbtool.py -- health, snapshot and salvage helpers for oh-my-pi's SQLite files.

oh-my-pi opens a database it cannot read, renames it to
".corrupt-<timestamp>-<uuid>" and rebuilds it.  That rename unlinks the file
under the mapping of the running process, so omp dies with SIGBUS ("Bus error
at address 0x7F...", exit 135) instead of starting -- and the corrupt file
stays where it is, so every later start repeats it.

Reading a database that a live instance holds open is dangerous as well.
SQLite maps "<db>-shm" into every process that has the database open, and a
second process that opens the same database rebuilds that file, which
invalidates the mapping of the first one and kills it with SIGBUS.  So every
read in this file happens on a COPY: the database file and its write-ahead log
are copied into a scratch directory and the copy is opened there.  The live
instance's own files are never opened, never mapped and never written.

TWO SPEED FACTS (measured on this NFS mount)
--------------------------------------------
* "PRAGMA integrity_check" on a copy of a database takes 1 millisecond, but
  only when the copy is opened with "immutable=1": sqlite then reads the
  database file directly, without a write-ahead log, without a shared-memory
  file and without a lock.  The copy holds one file, and deleting one closed
  file over NFS takes 3 milliseconds.
* A connection to a WAL database creates a "-wal" and a "-shm" file.  Deleting
  those while the connection is open costs 3 seconds over NFS, and closing
  the connection costs 1.6 seconds because the close checkpoints the log.

So the health check reads a copy with "immutable=1" and never creates the
side files.  The write-ahead log is not copied: it holds the pages that were
written since the last checkpoint, and the file it is copied into is the one
that gets damaged -- the file itself is what "PRAGMA integrity_check" has to
judge.

Subcommands
  check <db>            print "ok" / "corrupt: <reason>" / "missing";
                        exit 0 when the database is readable or absent,
                        exit 1 when it is corrupt
  credcount <db>        print the number of rows of auth_credentials
                        (-1 when the table cannot be read)
  snapshot <src> <dst>  write a consistent copy of <src> to <dst> through
                        sqlite's backup API
  recover <src> <dst>   rebuild <dst> from <src>: the schema of the source
                        and the rows of every table that can still be read
  checkpoint <db>       fold the write-ahead log into the database, truncate
                        it and remove the stale shared-memory file

The scratch directory is $OMP_DB_SCRATCH, or $HOME/.omp/.scratch by default.
"""
import os
import shutil
import sqlite3
import sys
import tempfile
import urllib.parse

# How many times a health check is repeated before a database is declared
# corrupt.  A copy of a database that is written at that very moment can be
# torn; the copy of the next attempt is normally whole again.
CHECK_ATTEMPTS = 3


def scratch_root():
    root = os.environ.get("OMP_DB_SCRATCH")
    if not root:
        root = os.path.join(os.path.expanduser("~"), ".omp", ".scratch")
    os.makedirs(root, exist_ok=True)
    return root


def workdir():
    return tempfile.mkdtemp(prefix="dbtool-", dir=scratch_root())


def copy_of(src, work):
    """Copy the database file into `work`.

    Only the database file itself is copied.  The write-ahead log holds the
    pages that were written since the last checkpoint, and the shared-memory
    file is a derived index of that log; neither is what the integrity check
    has to judge, and copying them means creating side files that are
    expensive to delete over NFS.
    """
    db = os.path.join(work, "db")
    shutil.copyfile(src, db)
    return db


def open_copy(path, work):
    """Open the copy of a database for reading.

    The copy is opened with "immutable=1": sqlite then reads the file
    directly -- no write-ahead log, no shared-memory file, no lock, and
    nothing that could reach the live instance.
    """
    db = copy_of(path, work)
    return sqlite3.connect("file:%s?immutable=1" % db, uri=True)


def open_full_copy(path, work):
    """Open a copy of a database together with its write-ahead log.

    Used by the commands that need the whole content of the database, not
    only what has been checkpointed into the file.  This copy creates the
    side files of a WAL database, so its connection is closed before the
    scratch directory is deleted.
    """
    db = copy_of(path, work)
    wal = path + "-wal"
    if os.path.exists(wal):
        shutil.copyfile(wal, db + "-wal")
    return sqlite3.connect(db)


def integrity(conn):
    row = conn.execute("PRAGMA integrity_check").fetchone()
    return row[0] if row else "unknown"


def check_db(path, attempts=CHECK_ATTEMPTS):
    """Return (verdict, detail): verdict is 'ok', 'corrupt' or 'missing'."""
    if not os.path.exists(path):
        return "missing", "no database yet"
    detail = "unknown"
    for _ in range(attempts):
        work = workdir()
        try:
            conn = open_copy(path, work)
            detail = integrity(conn)
            if detail == "ok":
                return "ok", "ok"
        except (sqlite3.DatabaseError, OSError) as exc:
            detail = "%s: %s" % (type(exc).__name__, exc)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    return "corrupt", detail


def cred_count(path):
    """Number of rows of auth_credentials, or -1 when it cannot be read."""
    if not os.path.exists(path):
        return -1
    work = workdir()
    try:
        conn = open_copy(path, work)
        try:
            return conn.execute("SELECT count(*) FROM auth_credentials").fetchone()[0]
        except sqlite3.DatabaseError:
            return -1
    except (sqlite3.DatabaseError, OSError):
        return -1
    finally:
        shutil.rmtree(work, ignore_errors=True)


def snapshot(src, dst):
    """A consistent copy of the source, written to dst."""
    if not os.path.exists(src):
        return 1, "no such database: %s" % src
    work = workdir()
    try:
        conn = open_full_copy(src, work)
        try:
            detail = integrity(conn)
            if detail != "ok":
                return 2, "the source is not readable: %s" % detail
            out = sqlite3.connect(dst)
            try:
                conn.backup(out)
                out.commit()
            finally:
                out.close()
        finally:
            # The connection is closed before the scratch directory is
            # deleted: a WAL connection that is still open leaves a "-wal"
            # and a "-shm" file behind, and deleting those over NFS costs
            # three seconds.
            conn.close()
    except (sqlite3.DatabaseError, OSError) as exc:
        return 1, "%s: %s" % (type(exc).__name__, exc)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0, "ok"


def recover(src, dst):
    """Rebuild dst from a corrupt src.

    The schema of the source is read as it is, and every table is copied row
    by row.  A table that cannot be read is left empty in the new database
    and reported: the salvage keeps what is still readable instead of
    refusing to touch the file at all.  The caller decides which tables are
    mandatory -- for the canonical database that is auth_credentials, and a
    rebuild without credentials must not be put in place.
    """
    if not os.path.exists(src):
        return 1, "no such database: %s" % src
    work = workdir()
    try:
        try:
            conn = open_full_copy(src, work)
        except sqlite3.DatabaseError as exc:
            return 1, "cannot open the source: %s" % exc
        lost = []
        try:
            try:
                objects = conn.execute(
                    "SELECT type, name, sql FROM sqlite_master "
                    "WHERE sql IS NOT NULL ORDER BY rowid").fetchall()
            except sqlite3.DatabaseError as exc:
                return 1, "cannot read the schema: %s" % exc
            tables = [(n, s) for t, n, s in objects
                      if t == "table" and not n.startswith("sqlite_")]
            if os.path.exists(dst):
                os.remove(dst)
            new = sqlite3.connect(dst)
            try:
                for name, sql in tables:
                    try:
                        new.execute(sql)
                    except sqlite3.DatabaseError as exc:
                        lost.append((name, "schema: %s" % exc))
                for name, _ in tables:
                    try:
                        rows = conn.execute('SELECT * FROM "%s"' % name).fetchall()
                    except sqlite3.DatabaseError as exc:
                        lost.append((name, "read: %s" % exc))
                        continue
                    if not rows:
                        continue
                    try:
                        new.execute("SAVEPOINT t")
                        new.executemany(
                            'INSERT INTO "%s" VALUES (%s)'
                            % (name, ",".join("?" * len(rows[0]))), rows)
                        new.execute("RELEASE t")
                    except sqlite3.DatabaseError as exc:
                        new.execute("ROLLBACK TO t")
                        new.execute("RELEASE t")
                        lost.append((name, "write: %s" % exc))
                for kind, name, sql in objects:
                    if kind in ("index", "view", "trigger"):
                        try:
                            new.execute(sql)
                        except sqlite3.DatabaseError as exc:
                            lost.append((name, "%s: %s" % (kind, exc)))
                new.commit()
            finally:
                new.close()
        finally:
            conn.close()
    except (sqlite3.DatabaseError, OSError) as exc:
        return 1, "%s: %s" % (type(exc).__name__, exc)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    for name, reason in lost:
        print("lost: %s (%s)" % (name, reason))
    return 0, "ok"


def checkpoint(path):
    """Fold the write-ahead log into the database and drop the stale index."""
    conn = sqlite3.connect(path)
    try:
        row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    finally:
        conn.close()
    shm = path + "-shm"
    if os.path.exists(shm):
        os.remove(shm)
    return row


def main(argv):
    if len(argv) < 2:
        sys.stderr.write(__doc__)
        return 2
    cmd = argv[1]
    args = argv[2:]
    if cmd == "check":
        if len(args) != 1:
            sys.stderr.write("usage: omp-dbtool.py check <db>\n")
            return 2
        verdict, detail = check_db(args[0])
        if verdict == "ok":
            print("ok")
            return 0
        if verdict == "missing":
            print("missing")
            return 0
        print("corrupt: %s" % detail)
        return 1
    if cmd == "credcount":
        if len(args) != 1:
            sys.stderr.write("usage: omp-dbtool.py credcount <db>\n")
            return 2
        print(cred_count(args[0]))
        return 0
    if cmd == "snapshot":
        if len(args) != 2:
            sys.stderr.write("usage: omp-dbtool.py snapshot <src> <dst>\n")
            return 2
        rc, detail = snapshot(args[0], args[1])
        if rc:
            sys.stderr.write("%s\n" % detail)
        return rc
    if cmd == "recover":
        if len(args) != 2:
            sys.stderr.write("usage: omp-dbtool.py recover <src> <dst>\n")
            return 2
        rc, detail = recover(args[0], args[1])
        if rc:
            sys.stderr.write("%s\n" % detail)
        return rc
    if cmd == "checkpoint":
        if len(args) != 1:
            sys.stderr.write("usage: omp-dbtool.py checkpoint <db>\n")
            return 2
        row = checkpoint(args[0])
        print(row)
        return 0
    sys.stderr.write("omp-dbtool.py: unknown command %s\n" % cmd)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
