"""What a sub-agent may read, decided by the framework and not by the model.

The caller names the research files a task is allowed to see; this module
refuses anything wider, and the executor mounts only what the policy accepted.
Three rules make the boundary a fact rather than a request:

* an allowlist entry must live inside the roots the task family declares
  (the round's own directory and its trees, or the legacy run's own paths);
* an allowlist entry may never contain the repository root, another arm's
  results, the Stage 3 outputs, the shared opencode data, the global opencode
  configuration, or the directory that holds the other agents' answers;
* the one permitted baseline is readable, and so is the task's own candidate
  tree, because both are research inputs the framework itself points at.

``BOUNDARY_VERSION`` changes whenever these rules change. Every receipt carries
it, so a verdict produced under an older boundary is visible as such and is not
reused silently. Version 3 added the append-only progress channel: the agent
has exactly one write, and it appends to the file that was bound for it.
"""
from pathlib import Path
from .. import settings

#: The version of the boundary itself. Bump it when the rules below change.
BOUNDARY_VERSION = "agent-boundary/3"

#: The one source tree a candidate may be read against.
BASELINE_ARM = "TIGER_RQ-VAE"


class PolicyError(RuntimeError):
    """The requested read allowlist is wider than the task may have."""


def _resolved(path: Path) -> Path:
    try:
        return Path(path).resolve()
    except OSError:
        return Path(path)


def _inside(path: Path, root: Path) -> bool:
    try:
        Path(path).relative_to(Path(root))
        return True
    except ValueError:
        return False


def repository_root() -> Path:
    """The repository root, found by the unambiguous marker."""
    return Path(settings.ROOT).resolve()


def forbidden_trees() -> list:
    """Directories no allowlist entry may live in."""
    repo = repository_root()
    home = Path.home()
    return [
        repo / "results" / "stage2_RQ-VAE",
        repo / "results" / "stage3_T5Train",
        repo / "stage2_RQ-VAE",
        home / ".config" / "opencode",
        home / ".cache" / "opencode",
        home / ".local" / "share" / "opencode",
    ]


def forbidden_exact() -> list:
    """Paths no allowlist entry may name."""
    return [repository_root()]


def _exemptions(candidate_tree) -> list:
    """The paths the framework itself points at, which are therefore readable."""
    allowed = [repository_root() / "stage2_RQ-VAE" / BASELINE_ARM]
    if candidate_tree is not None:
        allowed.append(Path(candidate_tree))
    return [_resolved(path) for path in allowed]


def check_read_paths(paths, *, roots, task_root=None, candidate_tree=None) -> list:
    """Return the accepted read allowlist, or refuse the task.

    ``roots`` are the directories the task family may read from at all;
    ``task_root`` is the run or round directory, whose ``agents`` directory
    holds the other agents' answers and may never be readable; and
    ``candidate_tree`` is the one candidate this task is about.
    """
    if roots is None:
        raise PolicyError("a task must declare the roots its allowlist may live in")
    root_paths = [_resolved(Path(root)) for root in roots]
    exempt = _exemptions(candidate_tree)
    accepted = []
    for entry in paths or []:
        path = Path(entry)
        if not path.is_absolute():
            raise PolicyError("the read allowlist must be absolute: " + str(entry))
        if not path.exists():
            raise PolicyError("the read allowlist names a missing path: " + str(entry))
        resolved = _resolved(path)
        if resolved in [_resolved(item) for item in forbidden_exact()]:
            raise PolicyError("the read allowlist may not contain the repository root: "
                              + str(entry))
        for tree in forbidden_trees():
            if _inside(resolved, _resolved(tree)) and resolved not in exempt:
                raise PolicyError("the read allowlist may not contain " + str(entry) +
                                  ": it is inside " + str(tree))
        if not any(_inside(resolved, root) for root in root_paths):
            raise PolicyError("the read allowlist may not contain " + str(entry) +
                              ": it is outside the roots this task declares")
        if task_root is not None:
            agents = _resolved(Path(task_root) / "agents")
            if _inside(resolved, agents):
                raise PolicyError("the read allowlist may not contain " + str(entry) +
                                  ": it holds the other agents' answers")
            if _inside(agents, resolved):
                # The path contains the directory that holds the other agents'
                # answers, so it contains them too.
                raise PolicyError("the read allowlist may not contain " + str(entry) +
                                  ": it contains the other agents' answers")
        accepted.append(path)
    return accepted
