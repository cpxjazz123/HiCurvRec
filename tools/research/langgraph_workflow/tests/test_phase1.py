"""Phase 1 of the LangGraph migration must not change the research result.

These checks compare the graph with the concurrent runner it replaces: the same
agents, the same answers, the same merged ideas and the same selection. The
agents and the probe are faked, so the checks are fast and deterministic; what
is under test is the orchestration, not the science.
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.research.langgraph_workflow import graph as langgraph_graph
from tools.research.langgraph_workflow.nodes import ideate as ideate_nodes
from tools.research.langgraph_workflow.nodes import probe as probe_nodes
from tools.research.langgraph_workflow.nodes import score as score_nodes
from tools.research.rounds import ideas as ideas_module
from tools.research.rounds import pipeline as pipeline_module
from tools.research.rounds import worker

ROUND_ID = "round_langgraph_test"


def _idea(index: int) -> dict:
    return {"idea_id": "idea%d" % index, "title": "t", "motivation": "m",
            "novelty": "n", "key_change": "k", "specification": "s",
            "cost_estimate": "c", "falsifiable_hypothesis": "f", "risks": ["r"],
            "behavior_signal": "b", "behavior_computation": "c",
            "coupling_path": "p", "hyperbolic_operator": "h"}


def _fake_agents(counter: list, failing: set = frozenset()):
    """A stand-in for the agent executor: one idea per agent, and a counter."""
    def run(root, task_id, prompt, output_name, parser, **kwargs):
        counter.append(task_id)
        index = int(task_id.rsplit("_", 1)[-1])
        if index in failing:
            raise RuntimeError("agent %d failed" % index)
        return [_idea(index)]
    return run


def _fake_collect(per_agent, require_minimum=True):
    """The merge without the scope gate, which the scope tests cover."""
    merged = []
    for agent, payload in per_agent:
        ideas = payload.get("ideas") if isinstance(payload, dict) else payload
        for raw in ideas:
            record = dict(raw)
            record["origin_agent"] = str(agent)
            merged.append(record)
    return merged


def _fake_probe(root, round_id, *, log=print):
    """The probe phase without the measurements: the state moves to scoring."""
    directory = pipeline_module.find_round_directory(root, round_id)
    state = pipeline_module.load_state(directory)
    state["step"] = "scoring"
    state["evidence_bundles"] = {idea["idea_id"]: {"idea_id": idea["idea_id"]}
                                 for idea in state.get("ideas") or []}
    pipeline_module.save_state(directory, state)
    return {"phase": "probe", "ran": True, "step": "scoring"}


def _fake_reviewer(root, task_id, prompt):
    """One reviewer's answer in the shape the scorer validates."""
    return {"ranking": [{"idea_id": "idea%d" % position, "position": position,
                         "reason": "RANK: idea%d | position=%d | reason=test"
                                  % (position, position)}
                        for position in range(1, pipeline_module.IDEA_AGENTS + 1)],
            "objection": "none"}


class GraphEquivalenceTests(unittest.TestCase):
    """The graph and the runner must reach the same round state."""

    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.repo = Path(self.work.name) / "repo"
        (self.repo / "results").mkdir(parents=True)
        (self.repo / "results" / "research" / "issue121" / "rounds").mkdir(parents=True)
        (self.repo / "stage2_RQ-VAE").mkdir(parents=True)
        pipeline_module.new_round(self.repo, ROUND_ID, objective="test objective")

    def tearDown(self):
        self.work.cleanup()

    def _run_graph(self):
        counter: list = []
        with mock.patch.object(worker, "_run_agent", side_effect=_fake_agents(counter)), \
                mock.patch.object(worker, "_rank_with_retry",
                                  side_effect=_fake_reviewer), \
                mock.patch.object(ideas_module, "collect_ideas",
                                  side_effect=_fake_collect), \
                mock.patch.object(worker, "phase_probe", side_effect=_fake_probe):
            state = langgraph_graph.run_phase1(self.repo, ROUND_ID,
                                               objective="test objective")
        return state, counter

    def test_one_branch_per_ideation_agent(self):
        state, counter = self._run_graph()
        ideation = [task for task in counter if task.startswith("ideation_")]
        self.assertEqual(len(ideation), pipeline_module.IDEA_AGENTS,
                         "every ideation agent must run exactly once")
        self.assertEqual(len(set(ideation)), pipeline_module.IDEA_AGENTS)
        self.assertEqual(len(state.get("ideas") or []), pipeline_module.IDEA_AGENTS)

    def test_the_graph_and_the_runner_agree(self):
        """The same agents, the same merged ideas and the same decision."""
        graph_state, _counter = self._run_graph()
        graph_ids = [idea["idea_id"] for idea in graph_state.get("ideas") or []]
        graph_status = (graph_state.get("selection") or {}).get("status")

        other = Path(self.work.name) / "runner"
        shutil.copytree(self.repo, other)
        shutil.rmtree(pipeline_module.round_root(other) / ROUND_ID)
        pipeline_module.new_round(other, ROUND_ID, objective="test objective")
        with mock.patch.object(worker, "_run_agent", side_effect=_fake_agents([])), \
                mock.patch.object(worker, "_rank_with_retry",
                                  side_effect=_fake_reviewer), \
                mock.patch.object(ideas_module, "collect_ideas",
                                  side_effect=_fake_collect), \
                mock.patch.object(worker, "phase_probe", side_effect=_fake_probe):
            worker.phase_ideation(other, ROUND_ID)
            worker.phase_probe(other, ROUND_ID)
            runner_result = worker.phase_scoring(other, ROUND_ID)
        directory = pipeline_module.find_round_directory(other, ROUND_ID)
        runner_state = pipeline_module.load_state(directory)
        runner_ids = [idea["idea_id"] for idea in runner_state.get("ideas") or []]
        self.assertEqual(sorted(graph_ids), sorted(runner_ids))
        self.assertEqual(graph_status, runner_result.get("status"),
                         "the graph and the runner must reach the same decision")

    def test_a_failed_agent_does_not_cost_the_others_their_ideas(self):
        with mock.patch.object(worker, "_run_agent",
                               side_effect=_fake_agents([], failing={3})), \
                mock.patch.object(worker, "_rank_with_retry",
                                  side_effect=_fake_reviewer), \
                mock.patch.object(ideas_module, "collect_ideas",
                                  side_effect=_fake_collect), \
                mock.patch.object(worker, "phase_probe", side_effect=_fake_probe):
            state = langgraph_graph.run_phase1(self.repo, ROUND_ID)
        self.assertTrue(state.get("errors"), "the failure must be recorded")
        self.assertEqual(len(state.get("ideas") or []),
                         pipeline_module.IDEA_AGENTS - 1)

    def test_the_round_state_is_written_by_the_graph(self):
        state, _counter = self._run_graph()
        directory = pipeline_module.find_round_directory(self.repo, ROUND_ID)
        on_disk = pipeline_module.load_state(directory)
        self.assertTrue(on_disk.get("ideas"))
        self.assertEqual(len(on_disk.get("ideas") or []),
                         len(state.get("ideas") or []))
        self.assertEqual((on_disk.get("selection") or {}).get("status"),
                         (state.get("selection") or {}).get("status"))


class GraphWiringTests(unittest.TestCase):
    """The graph's shape: parallel branches, then the phases in order."""

    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.repo = Path(self.work.name) / "repo"
        (self.repo / "results" / "research" / "issue121" / "rounds").mkdir(parents=True)
        (self.repo / "stage2_RQ-VAE").mkdir(parents=True)
        pipeline_module.new_round(self.repo, ROUND_ID, objective="test objective")

    def tearDown(self):
        self.work.cleanup()

    def test_the_graph_has_the_three_phases(self):
        compiled = langgraph_graph.build_graph()
        nodes = set(compiled.get_graph().nodes)
        for name in ("idea_agent", "merge_ideas", "probe", "scorer", "finalize"):
            self.assertIn(name, nodes)

    def test_the_dispatch_covers_every_agent(self):
        sends = ideate_nodes.dispatch_ideation({"root": "/tmp", "objective": "",
                                                "extra": False})
        self.assertEqual(len(sends), pipeline_module.IDEA_AGENTS)
        self.assertEqual(len(score_nodes.dispatch_scoring(
            {"root": "/tmp", "ideas": [], "round_state": {}})), worker.SCORERS)

    def test_the_probe_node_returns_the_round_state(self):
        with mock.patch.object(probe_nodes.probe, "run_probe_phase",
                               side_effect=_fake_probe):
            state = probe_nodes.probe_node({"root": str(self.repo),
                                            "round_id": ROUND_ID})
        self.assertIn("round_state", state)
        self.assertEqual(state["round_state"]["step"], "scoring")


if __name__ == "__main__":
    unittest.main()
