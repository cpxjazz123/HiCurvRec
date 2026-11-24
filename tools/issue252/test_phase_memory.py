"""Allocator regression: entering a child must not erase the parent's peak."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
from unittest import TestCase, main
from unittest.mock import patch
from tools.issue252.profiling import Tracker


class FakeAllocator:
    def __init__(self):
        self.current = self.peak = 0

    def allocate(self, size):
        self.current = size
        self.peak = max(self.peak, size)

    def reset(self):
        self.peak = self.current


class Event:
    def record(self):
        pass


class PhaseMemoryTests(TestCase):
    def test_nested_span_preserves_prechild_and_transient_child_peaks(self):
        allocator = FakeAllocator()
        tracker = Tracker('.', 0, None)
        tracker.stage = 'training_step'
        with patch('tools.issue252.profiling.torch.cuda.Event', return_value=Event()), \
             patch('tools.issue252.profiling.torch.cuda.max_memory_allocated', side_effect=lambda: allocator.peak), \
             patch('tools.issue252.profiling.torch.cuda.reset_peak_memory_stats', side_effect=allocator.reset), \
             patch('tools.issue252.profiling.record_function', side_effect=lambda _: nullcontext()):
            with tracker.section('parent'):
                allocator.allocate(900)
                allocator.allocate(100)
                with tracker.section('child'):
                    allocator.allocate(500)
                    allocator.allocate(100)
                with tracker.section('sibling'):
                    allocator.allocate(200)
                    allocator.allocate(100)
        self.assertEqual(tracker.phase_peaks[('training_step', 'parent')], 900)
        self.assertEqual(tracker.phase_peaks[('training_step', 'child')], 500)
        self.assertEqual(tracker.phase_peaks[('training_step', 'sibling')], 200)
        self.assertEqual(tracker.step_peak, 900)


class SteadyProfileTests(TestCase):
    @staticmethod
    def entry():
        root = Path(__file__).resolve().parents[2]
        path = root / 'stage2_RQ-VAE/_archive/TIGER_gnn_attention_sectional_rq/steady_profile_entry.py'
        spec = importlib.util.spec_from_file_location('issue252_steady_profile', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_every_rank_profiles_only_between_complete_epoch_boundaries(self):
        module = self.entry()
        for rank in range(4):
            recorded = []
            training = SimpleNamespace(_record=lambda *args, **kwargs: recorded.append((args, kwargs)))
            tracker = module.SteadyTracker('.', 0, training)
            tracker.install_record_hook()
            with patch.object(tracker, 'start_window') as start, \
                 patch.object(tracker, 'stop_window') as stop:
                training._record(rank, 'initialization_complete', initialization_seconds=1)
                training._record(rank, 'train', epoch=1)
                start.assert_not_called()
                training._record(rank, 'epoch_boundary', epoch=1, start=1, end=2)
                start.assert_called_once_with()
                stop.assert_not_called()
                training._record(rank, 'epoch_boundary', epoch=2, start=2, end=3)
                stop.assert_called_once_with()
                training._record(rank, 'epoch_boundary', epoch=3, start=3, end=4)
                training._record(rank, 'diagnostic_end', completed_epochs=3, epoch_budget=3000,
                                 checkpoint_written=False, sid_exported=False)
            self.assertEqual(tracker.completed_epochs, 3)
            self.assertEqual([row['epoch'] for row in tracker.boundaries], [1, 2, 3])
            self.assertEqual(len(recorded), 6)
            self.assertEqual(tracker.diagnostic_end['epoch_budget'], 3000)

    def test_disabled_wrappers_leave_initialization_and_other_epochs_unmeasured(self):
        module = self.entry()
        owner = SimpleNamespace(call=lambda value: value + 1)
        tracker = module.SteadyTracker('.', 0, None)
        tracker.wrap(owner, 'call', 'production_call')
        with patch.object(tracker, 'section', side_effect=lambda _: nullcontext()) as section:
            self.assertEqual(owner.call(1), 2)
            section.assert_not_called()
            tracker.enabled = True
            self.assertEqual(owner.call(2), 3)
            section.assert_called_once_with('production_call')
            tracker.enabled = False
            self.assertEqual(owner.call(3), 4)
            self.assertEqual(section.call_count, 1)


if __name__ == '__main__':
    main()
