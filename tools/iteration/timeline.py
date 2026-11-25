"""Wall-clock, CPU and GPU accounting for one iteration, overlap-aware.

A single agent runs phases that overlap (a CPU analysis while a GPU diagnostic
runs) and phases that must not (two formal trainings).  Summing durations
double-counts the overlapped wall time, so this module keeps *intervals* and
reports both the phased sum and the merged wall time per resource mask.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

SCHEMA = 'genrec.pipeline.timeline/1'


class Timeline:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, payload):
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(payload, sort_keys=True) + '\n')

    def start(self, phase, *, resource='cpu', gpu_mask='', note=''):
        event = {'schema': SCHEMA, 'event': 'start', 'phase': str(phase),
                 'resource': str(resource), 'gpu_mask': str(gpu_mask),
                 'wall': time.time(), 'monotonic': time.monotonic(),
                 'cpu_s': time.process_time(), 'pid': os.getpid(), 'note': str(note)}
        self._append(event)
        return event

    def stop(self, phase, *, status='PASS', note=''):
        event = {'schema': SCHEMA, 'event': 'stop', 'phase': str(phase),
                 'wall': time.time(), 'monotonic': time.monotonic(),
                 'cpu_s': time.process_time(), 'status': str(status), 'note': str(note)}
        self._append(event)
        return event

    def events(self):
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding='utf-8').splitlines() if line.strip()]

    def _open(self):
        counts = {}
        for event in self.events():
            delta = 1 if event['event'] == 'start' else -1
            counts[event['phase']] = counts.get(event['phase'], 0) + delta
        return [phase for phase, count in counts.items() if count > 0]

    def spans(self):
        open_by_phase = {}
        spans = []
        for event in self.events():
            phase = event['phase']
            if event['event'] == 'start':
                open_by_phase[phase] = event
                continue
            start = open_by_phase.pop(phase, None)
            if start is None:
                continue
            spans.append({'phase': phase, 'resource': start['resource'],
                          'gpu_mask': start['gpu_mask'],
                          'start': start['monotonic'], 'stop': event['monotonic'],
                          'wall_s': event['monotonic'] - start['monotonic'],
                          'cpu_s': max(0.0, event['cpu_s'] - start['cpu_s']),
                          'status': event.get('status', 'PASS')})
        return spans

    def summary(self):
        spans = self.spans()
        phased = {}
        for span in spans:
            row = phased.setdefault(span['phase'], {'spans': 0, 'wall_s': 0.0, 'cpu_s': 0.0,
                                                    'resource': span['resource'], 'status': set()})
            row['spans'] += 1
            row['wall_s'] += span['wall_s']
            row['cpu_s'] += span['cpu_s']
            row['status'].add(span['status'])
        for row in phased.values():
            row['wall_s'] = round(row['wall_s'], 3)
            row['cpu_s'] = round(row['cpu_s'], 3)
            row['status'] = sorted(row['status'])
        return {'schema': 'genrec.pipeline.timeline_summary/1',
                'phases': phased,
                'merged_wall_s': round(merged(spans), 3),
                'gpu_wall_s': round(sum(s['wall_s'] for s in spans if s['resource'] == 'gpu'), 3),
                'cpu_wall_s': round(sum(s['wall_s'] for s in spans if s['resource'] == 'cpu'), 3),
                'span_count': len(spans),
                'open_phases': sorted(self._open())}


def merged(spans):
    """Length of the union of the spans' intervals (no double counting)."""
    intervals = sorted((s['start'], s['stop']) for s in spans if s['stop'] >= s['start'])
    total = 0.0
    current_start = current_stop = None
    for start, stop in intervals:
        if current_start is None:
            current_start, current_stop = start, stop
        elif start <= current_stop:
            current_stop = max(current_stop, stop)
        else:
            total += current_stop - current_start
            current_start, current_stop = start, stop
    if current_start is not None:
        total += current_stop - current_start
    return total


def exclusive(spans, gpu_mask):
    """Wall time during which exactly this GPU mask was held.

    Used for the resource claim: two phases on the same mask serialise, phases
    on disjoint masks overlap.
    """
    intervals = sorted((s['start'], s['stop']) for s in spans if s['gpu_mask'] == gpu_mask)
    return merged([{'start': a, 'stop': b} for a, b in intervals])
