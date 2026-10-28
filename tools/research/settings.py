"""One source of configuration; no CLI flag or environment override."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE_ROOT = ROOT / "results" / "research" / "issue117"
TRAIN = ROOT / "results" / "stage0_build_parquet" / "train.parquet"
VALID = ROOT / "results" / "stage0_build_parquet" / "valid.parquet"
TEST = ROOT / "results" / "stage0_build_parquet" / "test.parquet"
STAGE2_BASELINE = ROOT / "stage2_RQ-VAE" / "TIGER_RQ-VAE"
STAGE2_BEHAVIOR = ROOT / "stage2_RQ-VAE" / "TIGER_BEHAVIOR_SAMPLE_RQ-VAE"
STAGE3 = ROOT / "stage3_T5Train"
VERIFICATION = ROOT / "tools" / "research" / "verification" / "admission.py"
MCP = ROOT / "tools" / "genrec-mcp" / "server.py"

# Change the source constant, not an argv flag. Inspect never spawns an agent.
ACTION = "inspect"  # inspect | behavior | worker
# Live OpenCode Agents run inside two layers written by
# tools/research/agents/runner.py: (1) a per-task opencode permission object
# next to the session (shell denied for read-only roles, external paths denied,
# edits denied), and (2) for the round agents, a bubblewrap mount namespace that
# mounts only the workspace and the explicit read allowlist, so the repository,
# the other arms' results and the earlier sessions do not exist inside it. PWD is
# set to the workspace because opencode resolves its project from PWD; leaving it
# at the server's directory silently disabled every per-workspace rule.
LIVE_AGENTS_ALLOWED = True
RESEARCH_PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10"
AGENT_BINARY = "opencode"
AGENT_MODEL = "opencode-go/deepseek-v4.1-flash"
AGENT_TIMEOUT_SECONDS = 1800
LEASE_SECONDS = 1200
#: No numeric cap on repairable errors or evidence-based MATERIAL revisions.
#: FATAL/PARKED scientific verdicts, cancellation and resource failures remain stops.
MAX_ATTEMPTS = None

# Explicitly forbidden here. Real training remains behind the existing MCP.
AUTO_STAGE2 = False
AUTO_STAGE3 = False
