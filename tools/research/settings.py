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
# Live OpenCode Agents run only inside the per-task permission boundary that
# tools/research/agents/runner.py writes next to every agent session: shell and
# Code Mode denied, external paths denied, and edits limited to the agent's own
# workspace (a relative path that climbs out of it is denied too). That boundary
# was validated against the real CLI: a relative write inside the workspace is
# accepted, while a parent-traversal write and an absolute write into the
# repository are both refused.
LIVE_AGENTS_ALLOWED = True
RESEARCH_PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10"
AGENT_BINARY = "opencode"
AGENT_MODEL = "openai/gpt-6-luna"
AGENT_TIMEOUT_SECONDS = 900
LEASE_SECONDS = 1200
#: No numeric cap on repairable errors or evidence-based MATERIAL revisions.
#: FATAL/PARKED scientific verdicts, cancellation and resource failures remain stops.
MAX_ATTEMPTS = None

# Explicitly forbidden here. Real training remains behind the existing MCP.
AUTO_STAGE2 = False
AUTO_STAGE3 = False
