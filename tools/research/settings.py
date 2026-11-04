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
# Live agents run inside the filesystem boundary that
# tools/research/agents/runtime.py enforces: the agent's tools read only the
# workspace and the policy-accepted read allowlist, write only the workspace,
# and have no shell. The agent's own write is its progress file: the
# append_progress tool appends one JSONL record to a file that was bound before
# the agent started. The deadline below is counted from the dispatch and closes
# that file; a task that runs out of time has delivered what it delivered, and
# no more. Every receipt carries the boundary version, and evidence produced
# under another boundary is not reused.
#
LIVE_AGENTS_ALLOWED = True
RESEARCH_PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10"
#: The chat model the agents are created with. Empty means the runtime
#: refuses to run: an agent without a model is not an agent.
AGENT_CHAT_MODEL = "deepseek/deepseek-v4.1-flash"
#: "on" makes the SDK enforce the JSON schema on the agent's answer (the
#: structured output the SDK supports natively). A provider that refuses
#: response_format=json_schema falls back to the prompted JSON, once per agent.
AGENT_STRUCTURED_OUTPUT = "on"
AGENT_TIMEOUT_SECONDS = 300
#: The deadline of one sub-agent of a round, counted from the moment the task is
#: dispatched: the initialisation, the model calls, the retries and the
#: wrap-up all fit in it. The agent's progress file is frozen when it passes,
#: and a timeout is never a passing verdict.
AGENT_DEADLINE_SECONDS = 300
#: How long before the deadline the agent is asked to wrap up. The progress
#: tool says so in its answer, so the last record lands while there is time.
AGENT_WRAPUP_MARGIN_SECONDS = 60
LEASE_SECONDS = 1200
#: No numeric cap on repairable errors or evidence-based MATERIAL revisions.
#: FATAL/PARKED scientific verdicts, cancellation and resource failures remain stops.
MAX_ATTEMPTS = None

# Explicitly forbidden here. Real training remains behind the existing MCP.
AUTO_STAGE2 = False
AUTO_STAGE3 = False
