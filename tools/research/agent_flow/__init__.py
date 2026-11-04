"""The research flow of a round, driven by the OpenAI Agents SDK.

The SDK creates and runs the agents; the flow decides which phases run and in
which order, and the round's own state machine merges what the agents answered.
There is no graph engine in between: a phase is the framework's existing phase
function, and the agents inside it run concurrently through the SDK.
"""
from tools.research.agent_flow.flow import run_flow

__all__ = ["run_flow"]
