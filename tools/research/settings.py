"""One source of configuration; no CLI flag or environment override."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE_ROOT = ROOT / "results" / "research" / "issue116"
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
LIVE_AGENTS_ALLOWED = False
AGENT_BINARY = "opencode"
AGENT_MODEL = "openai/gpt-6-luna"
AGENT_TIMEOUT_SECONDS = 900
LEASE_SECONDS = 1200
MAX_ATTEMPTS = 2

# Explicitly forbidden here. Real training remains behind the existing MCP.
AUTO_STAGE2 = False
AUTO_STAGE3 = False
