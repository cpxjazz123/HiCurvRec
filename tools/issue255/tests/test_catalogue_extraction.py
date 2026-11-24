"""Small deterministic consumer artifacts; no GPU work or production dispatch."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.issue255 import evaluate_arm as evaluation
from tools.issue255 import mechanism_metrics as metrics


class CatalogueExtractionTest(unittest.TestCase):
    def setUp(self):
        scratch = evaluation.ROOT / 'results/research/issue255'
        scratch.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.books = [np.array([[1.e8, 1.], [2., 3.]], dtype=np.float32),
                      np.array([[-1.e8, 4.], [5., 6.]], dtype=np.float32),
                      np.array([[1., 7.], [8., 9.]], dtype=np.float32)]
        self.rows = np.array([[0, 0, 0, 6], [0, 0, 0, 7], [1, 1, 1, 6]], dtype=np.int64)
        self.arrays = {'sid': self.rows.copy(),
                       'prefixes': metrics.cumulative_prefixes(self.books, self.rows),
                       'weights': np.full((3, 3, 3), np.float32(1 / 3)),
                       'attention': np.full((3, 3, 3, 3), np.float32(1 / 3)),
                       'token_mask': np.ones((3, 3), dtype=np.bool_),
                       'reconstruction': np.zeros((3, 4), dtype=np.float32)}
        self.catalogue = self.root / 'issue255_catalogue.npz'
        self.context = {'generations': {'issue255': {
            'run_id': 'corrected_run', 'source': 'immutable/src',
            'artifacts': {'sid': {'sha256': 'sid-digest'},
                          'checkpoint': {'sha256': 'checkpoint-digest'}}}},
            'inputs': {'manifest': {'sha256': 'manifest-digest'}}}
        self.receipt = self.root / 'EXTRACTION.json'
        self.document = {'schema': 'genrec.issue255.catalogue_extraction/1', 'status': 'COMPLETE',
                         'context': copy.deepcopy(self.context),
                         'metric': evaluation.common_metric(),
                         'catalogues': {'issue255': {
                             'execution': {'device_type': 'cuda', 'catalogue_passes': 1,
                                           'matmul_precision': 'high',
                                           'no_grad': True, 'inference_mode': False,
                                           'actual_sid_reexport_equal': True,
                                           'all_levels_bitwise_equal': True},
                             'artifact': None}}}
        self.publish()

    def publish(self):
        np.savez(self.catalogue, **self.arrays)
        self.document['catalogues']['issue255']['artifact'] = metrics.file_record(self.catalogue)
        self.receipt.write_text(json.dumps(self.document))

    def consume(self):
        return evaluation.load_extraction(
            self.receipt, context=self.context, rows={'issue255': self.rows},
            books={'issue255': self.books}, embedding_shape=(3, 4),
            token_columns={'issue255': 3})

    def test_complete_small_catalogue_and_hard_prefix_rounding(self):
        arrays = self.consume()['issue255']
        np.testing.assert_array_equal(arrays['prefixes'], self.arrays['prefixes'])
        self.assertEqual(arrays['prefixes'][2, 0, 0], 1.)
        self.assertEqual(metrics.sid_structure(arrays['sid'])['distinct_l3_triples'], 2)

    def test_wrong_run_sid_and_checkpoint_receipts_are_rejected(self):
        for field in ('run_id', 'sid', 'checkpoint'):
            with self.subTest(field=field):
                document = copy.deepcopy(self.document)
                identity = document['context']['generations']['issue255']
                if field == 'run_id':
                    identity[field] = 'invalid_implementation'
                else:
                    identity['artifacts'][field]['sha256'] = 'another-digest'
                self.receipt.write_text(json.dumps(document))
                with self.assertRaisesRegex(ValueError, 'provenance'):
                    self.consume()

    def test_altered_catalogue_bytes_are_rejected(self):
        self.arrays['reconstruction'][0, 0] = 1.
        np.savez(self.catalogue, **self.arrays)
        with self.assertRaisesRegex(ValueError, 'catalogue.*changed'):
            self.consume()

    def test_changed_prefix_rejected_even_with_refreshed_artifact_hash(self):
        self.arrays['prefixes'][0, 0, 1] += 1.
        self.publish()
        with self.assertRaisesRegex(ValueError, 'prefix differs'):
            self.consume()

    def test_suffix_excluded_from_prefix_but_exact_export_still_required(self):
        changed = self.rows.copy()
        changed[:, 3] += 100
        np.testing.assert_array_equal(metrics.cumulative_prefixes(self.books, changed),
                                      self.arrays['prefixes'])
        self.arrays['sid'] = changed
        self.publish()
        with self.assertRaisesRegex(ValueError, 'actual exported SID'):
            self.consume()

    def test_common_metric_drift_rejected(self):
        self.document['metric']['tangent_scale'] *= 2
        self.publish()
        with self.assertRaisesRegex(ValueError, 'common_fixed3'):
            self.consume()

    def test_runtime_from_a_different_sid_export_is_rejected(self):
        self.document['catalogues']['issue255']['execution']['matmul_precision'] = 'highest'
        self.publish()
        with self.assertRaisesRegex(ValueError, 'export numerical policy'):
            self.consume()



if __name__ == '__main__':
    unittest.main()
