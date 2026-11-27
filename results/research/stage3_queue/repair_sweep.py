"""Run the Stage3 gate's own export repair for the queue's repair arms.

Mimics stage3_export_gate's repair branch exactly (static arms: PY3 and no --tree;
run-layout arms: the run's source tree and its interpreter), then re-reads the
manifest and records the verdict. No dispatch; no binding.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from tools.research.stage2.export_policy import manifest_path, read_manifest, archived_sids, decide  # noqa: E402
from tools.issue252.run_workspace import run_from_sid  # noqa: E402


def load_shared():
    spec = importlib.util.spec_from_file_location("srv", REPO / "tools/genrec-mcp/server.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    shared = load_shared()
    queue = json.loads((REPO / "results/research/stage3_queue/queue_dispatch.json").read_text())
    arms = [e for e in queue["order"] if e.get("manifest") == "needs-export-repair"]
    archived = archived_sids(REPO)
    out = REPO / "results/research/stage3_queue/repair_sweep.jsonl"
    with out.open("a", encoding="utf-8") as sink:
        for e in arms:
            sid = (REPO / e["sid"]).resolve()
            sha = __import__("hashlib").sha256(sid.read_bytes()).hexdigest()
            man = read_manifest(sid)
            if man is not None:
                v = decide(sha, man, archived)
                rec = {"arm": e["arm"], "action": "existing-manifest", "state": v["state"], "reason": v["reason"]}
            else:
                run = run_from_sid(REPO, sid)
                tree = Path(run["source"]) if run else shared.resolve_stage2_tree(sid.parent.name)[0]
                python = shared.tree_python(tree) if tree is not None else shared.PY3
                cmd = [python, "-m", "tools.research.stage2.export_check", "--sid", str(sid), "--rewrite"]
                if tree is not None:
                    cmd.extend(["--tree", str(tree)])
                done = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True,
                                      timeout=1800, env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin:/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin"})
                man = read_manifest(sid)
                v = decide(__import__("hashlib").sha256(sid.read_bytes()).hexdigest(), man, archived) if man else {"state": "REFUSED", "reason": "no manifest"}
                rec = {"arm": e["arm"], "action": "export_check", "rc": done.returncode,
                       "state": v["state"], "reason": (done.stdout or done.stderr or "").strip()[-300:],
                       "rewrote": bool((man or {}).get("rewritten_from_checkpoint"))}
            sink.write(json.dumps(rec) + "\n"); sink.flush()
            print(json.dumps(rec)[:400], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
