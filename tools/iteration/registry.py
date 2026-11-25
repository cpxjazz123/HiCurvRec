"""One round's state machine, persisted; no step can be skipped and no state is inferred.

The pipeline drives a round through fixed states.  Each transition records the
evidence that justified it and the artefacts it produced, so a resumed agent can
tell what actually happened without reading a chat log.  A transition that is not
in the table is refused.
"""
from __future__ import annotations

import json
from pathlib import Path

SCHEMA = 'genrec.pipeline.round/1'

STATES = (
    'REGISTERED',        # contract frozen, run id and arm fixed
    'VALIDATED',         # the dependency-aware validation plan ran green
    'ADMITTED',          # real entry + behaviour graph machine gates
    'STAGE2_RUNNING',
    'STAGE2_DONE',       # 3000 epochs, export exact, quality recorded
    'STAGE3_RUNNING',
    'STAGE3_DONE',       # frozen 150 epochs, complete test, independent reduction
    'ANALYSED',          # automated analysis + report
    'ARCHIVED',          # four artefacts in R2, ledger entry verified
    'CLOSED',            # verdict written and the issue closed
)

TRANSITIONS = {STATES[0]: {STATES[1]}, STATES[1]: {STATES[2]}, STATES[2]: {STATES[3]},
               STATES[3]: {STATES[4]}, STATES[4]: {STATES[5]}, STATES[5]: {STATES[6]},
               STATES[6]: {STATES[7]}, STATES[7]: {STATES[8]}, STATES[8]: {STATES[9]},
               STATES[9]: set()}

#: States that may be re-entered from themselves after a resumed/retried phase.
SELF_RETRY = {STATES[3], STATES[5]}


class Round:
    def __init__(self, root, round_id):
        self.root = Path(root)
        self.round_id = str(round_id)
        self.path = self.root / (self.round_id + '.json')
        if not self.path.is_file():
            raise ValueError('round is not registered: ' + self.round_id)
        self.payload = json.loads(self.path.read_text(encoding='utf-8'))

    @classmethod
    def register(cls, root, round_id, *, arm, run_id, contract_sha256, hypothesis, notes=''):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        path = root / (str(round_id) + '.json')
        if path.exists():
            raise ValueError('round already registered: ' + str(round_id))
        payload = {'schema': SCHEMA, 'round_id': str(round_id), 'arm': str(arm),
                   'run_id': str(run_id), 'contract_sha256': str(contract_sha256),
                   'hypothesis': str(hypothesis), 'notes': str(notes),
                   'state': STATES[0], 'history': []}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
        return cls(root, round_id)

    @property
    def state(self):
        return self.payload['state']

    def can(self, target):
        if target == self.state and self.state in SELF_RETRY:
            return True
        return target in TRANSITIONS[self.state]

    def advance(self, target, *, evidence=None, note=''):
        if target not in STATES:
            raise ValueError('unknown state: ' + str(target))
        if not self.can(target):
            raise ValueError('%s cannot move to %s (allowed: %s)'
                             % (self.state, target, sorted(TRANSITIONS[self.state])))
        self.payload['history'].append({'from': self.state, 'to': target,
                                        'evidence': evidence or {}, 'note': str(note)})
        self.payload['state'] = target
        self.path.write_text(json.dumps(self.payload, indent=2, sort_keys=True) + '\n')
        return self.payload

    def evidence(self, name):
        for row in reversed(self.payload['history']):
            if name in (row.get('evidence') or {}):
                return row['evidence'][name]
        return None
