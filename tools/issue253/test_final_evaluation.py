"""Independent C4/C16/C17 contracts; CPU only and real file/source bindings."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from tools.issue253 import evaluation_reference as reference

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'stage2_RQ-VAE').is_dir())
HISTORICAL_SHA256 = '0ca5cd32c307460cad7ecd72734538dff2ca3427f64286063ece8c290f8e28f0'


class AmbientMetricTests(unittest.TestCase):
    def test_collinear_known_answer_and_three_five_distinction(self):
        origin = np.zeros((1, 2))
        endpoint = np.array([[2.5, 0.]])
        actual = reference.fixed_metrics_numpy(origin, endpoint, scale=1.)
        # At radius2.5 the +3 sphere wraps: acos(cos(sqrt(3)*2.5)).
        wrapped = (2*math.pi-math.sqrt(3)*2.5)**2/3
        np.testing.assert_allclose(actual['fixed3'], [[6.25]], atol=1e-12)
        np.testing.assert_allclose(actual['fixed5'], [[(4*6.25+wrapped)/5]], atol=1e-12)
        self.assertGreater(abs(actual['fixed3'][0, 0]-actual['fixed5'][0, 0]), .1)

    def test_shared_matrix_paired_and_streamed_contracts(self):
        x = np.array([[0., 0.], [.3, -.2], [.7, .1]])
        y = np.array([[.1, .4], [-.2, .5], [.6, -.3]])
        both = reference.fixed_metrics_numpy(x, y, scale=1.3)
        paired = reference.fixed_metrics_numpy(x, y, scale=1.3, paired=True)
        for name, matrix, pair in (
                ('fixed3', reference.fixed_three_squared_numpy, reference.fixed_three_paired_numpy),
                ('fixed5', reference.fixed_five_squared_numpy, reference.fixed_five_paired_numpy)):
            np.testing.assert_allclose(matrix(x, y, scale=1.3), both[name], atol=1e-12)
            np.testing.assert_allclose(pair(x, y, scale=1.3), np.diag(both[name]), atol=1e-12)
            np.testing.assert_allclose(paired[name], np.diag(both[name]), atol=1e-12)
            tiled = np.empty_like(both[name])
            for left, right, values in reference.matrix_blocks(x, y, scale=1.3, chunk=2):
                tiled[left, right] = values[name]
            np.testing.assert_allclose(tiled, both[name], atol=1e-12)
        self.assertEqual(reference.fixed_three_squared_numpy(x[:0], y).shape, (0, 3))
        self.assertEqual(reference.fixed_three_paired_numpy(x[:0], y[:0]).shape, (0,))

    def test_invalid_scale_and_pair_alignment_fail_closed(self):
        x = np.zeros((2, 2))
        for scale in (0., -1., float('nan'), float('inf')):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                reference.fixed_metrics_numpy(x, x, scale=scale)
        for y in (x[:1], np.zeros((2, 3)), np.full_like(x, np.nan)):
            with self.subTest(shape=y.shape), self.assertRaises(ValueError):
                reference.fixed_metrics_numpy(x, y, paired=True)
        with self.assertRaises(ValueError):
            reference.fixed_metrics_numpy(np.array([[301., 0.]]), x)


class CommonScaleAndRepresentationTests(unittest.TestCase):
    def test_scale_comes_only_from_exact_bound_historical_checkpoint(self):
        from tools.issue253.final_evaluation import comparison_scale
        state = {'state_dict': {'tangent_scale': torch.tensor(.125)}}
        self.assertEqual(comparison_scale(state, checkpoint_sha256=HISTORICAL_SHA256), .125)
        with self.assertRaisesRegex(ValueError, 'checkpoint|hash|historical'):
            comparison_scale(state, checkpoint_sha256='a'*64)
        for value in (0., -1., float('nan'), float('inf'), [1., 2.]):
            invalid = {'state_dict': {'tangent_scale': torch.tensor(value)}}
            with self.subTest(value=value), self.assertRaises(ValueError):
                comparison_scale(invalid, checkpoint_sha256=HISTORICAL_SHA256)
        with self.assertRaises(ValueError):
            comparison_scale({'state_dict': {}}, checkpoint_sha256=HISTORICAL_SHA256)

    def test_hard_q_is_three_actual_books_not_z_or_collision_counter(self):
        from tools.graph_infra.fidelity import decode_hard_sid
        from tools.issue253.final_evaluation import validate_representation
        books = [np.array([[level+.25, 0.], [0., level+.5]]) for level in range(3)]
        sid = np.array([[0, 1, 0, 0], [1, 0, 1, 99]], dtype=np.int64)
        hard = decode_hard_sid(sid, books)
        np.testing.assert_array_equal(hard, [[2.5, 1.5], [1.25, 3.]])
        validate_representation(hard, representation='hard_q', sid=sid, codebooks=books)
        continuous = hard+.125
        validate_representation(continuous, representation='continuous_z', sid=sid, codebooks=books)
        for value, label in ((continuous, 'hard_q'), (hard[:1], 'continuous_z'),
                             (np.full_like(hard, np.nan), 'continuous_z'), (hard, 'z_as_q')):
            with self.subTest(label=label), self.assertRaises(ValueError):
                validate_representation(value, representation=label, sid=sid, codebooks=books)


class FullCoverageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue253')
        self.addCleanup(self.temporary.cleanup)
        self.assets = Path(self.temporary.name)
        (self.assets/'shards').mkdir()
        self.q = np.array([[0., 0.], [.3, .1], [-.2, .4], [0., 0.], [.5, -.1]])
        self.labels = np.array([0, 0, 0, 1, 1])
        # Duplicate occurrences and unequal centre counts are deliberate.
        self.triples = np.array([[0, 1, 2], [0, 1, 2], [1, 0, 2], [3, 3, 4]])
        self.targets = np.array([-1., -1., 0., 2.])
        self.graph_squared = np.array([4., 4., 1., 0.])
        np.savez(self.assets/'node_moments.npz', component=self.labels)
        np.savez(self.assets/'index.npz', shard_names=np.array(['a.npz', 'b.npz']))
        for name, selected in (('a.npz', slice(0, 3)), ('b.npz', slice(3, 4))):
            triples = self.triples[selected]
            np.savez(self.assets/'shards'/name,
                     centre=np.append(triples[:, 0], -1), b=np.append(triples[:, 1], -1),
                     c=np.append(triples[:, 2], -1), valid=np.array([True]*len(triples)+[False]),
                     curvature=np.append(self.targets[selected], np.nan),
                     bc_dist_sq=np.append(self.graph_squared[selected], np.nan))

    def test_existing_full_evaluate_accepts_both_independent_metrics(self):
        from tools.issue251.full_evaluation import full_evaluate
        for name, matrix, paired in (
                ('fixed3', reference.fixed_three_squared_numpy, reference.fixed_three_paired_numpy),
                ('fixed5', reference.fixed_five_squared_numpy, reference.fixed_five_paired_numpy)):
            got = full_evaluate(self.q, self.assets, scale=.8, metric=matrix,
                                paired_metric=paired, chunk=2, triple_chunk=2)
            predicted = reference.a1_predictions(self.q, self.triples, self.labels,
                                                 scale=.8, experts=int(name[-1]))
            expected = reference.reduce_curvature(self.targets, predicted, self.triples[:, 0])
            self.assertEqual(got['curvature']['count'], 4)
            self.assertEqual(got['curvature']['excluded'], 2)
            for field in ('micro_mae', 'macro_mae'):
                self.assertAlmostEqual(got['curvature'][field], expected[field], places=12)
            self.assertAlmostEqual(got['curvature']['sign']['balanced_accuracy'],
                                   expected['sign']['balanced_accuracy'], places=12)
            for sign in ('negative', 'zero', 'positive'):
                self.assertEqual(got['curvature']['sign']['recall'][sign], expected['sign']['recall'][sign])
            self.assertEqual(got['distance_distortion']['count'], 4)
            self.assertFalse(got['distance_distortion']['scale_fitted_on_evaluation'])

    def test_joint_cells_preserve_scales_full_rows_signs_and_distortion(self):
        from tools.issue253.final_evaluation import evaluate_cells
        cells = evaluate_cells(self.q, self.assets, native_scale=.7, common_scale=.8,
                               chunk=2, triple_chunk=2)
        self.assertEqual(set(cells), {'native_fixed3', 'common_fixed3', 'common_fixed5'})
        for name, scale, experts in (('native_fixed3', .7, 3), ('common_fixed3', .8, 3),
                                     ('common_fixed5', .8, 5)):
            got = cells[name]
            predicted = reference.a1_predictions(self.q, self.triples, self.labels,
                                                 scale=scale, experts=experts)
            expected = reference.reduce_curvature(self.targets, predicted, self.triples[:, 0])
            self.assertEqual(got['curvature']['count'], 4)
            self.assertEqual(got['curvature']['excluded'], 2)
            self.assertEqual(got['tangent_scale'], scale)
            self.assertEqual(got['expert_curvatures'], [-1, 0, 1] if experts == 3 else [-3, -1, 0, 1, 3])
            self.assertFalse(got['distance_distortion']['scale_fitted_on_evaluation'])
            for field in ('macro_mae', 'micro_mae'):
                self.assertAlmostEqual(got['curvature'][field], expected[field], places=12)
            self.assertAlmostEqual(got['curvature']['sign']['balanced_accuracy'],
                                   expected['sign']['balanced_accuracy'], places=12)
            for sign in ('negative', 'zero', 'positive'):
                self.assertEqual(got['curvature']['sign']['recall'][sign],
                                 expected['sign']['recall'][sign])
            pair = reference.fixed_metrics_numpy(self.q[self.triples[:, 1]],
                    self.q[self.triples[:, 2]], scale=scale, paired=True)['fixed'+str(experts)]
            error = np.abs(np.sqrt(pair)-np.sqrt(self.graph_squared))
            self.assertAlmostEqual(got['distance_distortion']['mean_abs_error'], error.mean(), places=12)
            sufficient = [row['distance_distortion'] for row in got['curvature']['per_item'].values()]
            self.assertEqual(sum(row['count'] for row in sufficient), 4)
            self.assertAlmostEqual(math.fsum(row['absolute_sum'] for row in sufficient)/4,
                                   error.mean(), places=12)
        historical = evaluate_cells(self.q, self.assets, common_scale=.8, chunk=2, triple_chunk=2)
        self.assertEqual(set(historical), {'common_fixed3', 'common_fixed5'})

    def test_targets_cannot_fit_or_change_geometric_scale(self):
        from tools.issue253.final_evaluation import evaluate_cells
        before = evaluate_cells(self.q, self.assets, common_scale=.8, chunk=2, triple_chunk=2)
        path = self.assets/'shards/a.npz'
        with np.load(path) as blob:
            data = {key: blob[key] for key in blob.files}
        data['curvature'][data['valid']] *= 100
        np.savez(path, **data)
        after = evaluate_cells(self.q, self.assets, common_scale=.8, chunk=2, triple_chunk=2)
        for cell in before:
            self.assertEqual(before[cell]['tangent_scale'], after[cell]['tangent_scale'])
            self.assertEqual(before[cell]['distance_distortion'], after[cell]['distance_distortion'])
            self.assertNotEqual(before[cell]['curvature']['micro_mae'],
                                after[cell]['curvature']['micro_mae'])

    def test_reconciliation_rejects_incomplete_full_count(self):
        from tools.issue253.reconcile_results import reconcile_curvature
        predicted = reference.a1_predictions(self.q, self.triples, self.labels, scale=.8, experts=3)
        report = reference.reduce_curvature(self.targets, predicted, self.triples[:, 0])
        reconcile_curvature(report, expected_count=4)
        with self.assertRaisesRegex(ValueError, 'coverage|count|full'):
            reconcile_curvature(report, expected_count=28391948)
        for field in ('macro_mae', 'micro_mae'):
            invalid = copy.deepcopy(report)
            invalid[field] += .1
            with self.subTest(field=field), self.assertRaises(ValueError):
                reconcile_curvature(invalid, expected_count=4)


class SourceBoundCheckpointTests(unittest.TestCase):
    def setUp(self):
        from tools.issue252 import archive_audit
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue253')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.tree = self.root/'source'
        (self.tree/'model').mkdir(parents=True)
        # Source fixtures are hash-bound files, never imported dependency stubs.
        (self.tree/'train_rqvae.py').write_text('from model import VALUE\n')
        (self.tree/'model/__init__.py').write_text('VALUE=1\n')
        self.sid = self.root/'item_sids.json'
        self.sid.write_text('{"0":[0,0,0,0]}')
        self.checkpoint = self.root/'rqvae_best.pth'
        payload = {'epoch': 3000, 'selection': 'final_epoch', 'run_id': 'test-formal',
                   'source_tree': str(self.tree), 'state_dict': {'tangent_scale': torch.tensor(.125)},
                   'provenance': {'candidate_sources': archive_audit.source_hashes(self.tree),
                                  'infrastructure_sources': {}}}
        torch.save(payload, self.checkpoint)
        self.export = self.root/'export_manifest.json'
        self.export.write_text(json.dumps({'schema': 'genrec.stage2.export_manifest/1',
            'tree': str(self.tree), 'checkpoint': str(self.checkpoint),
            'checkpoint_sha256': archive_audit.sha256(self.checkpoint),
            'sid': str(self.sid), 'sid_sha256': archive_audit.sha256(self.sid),
            'tokens_match': True, 'same_l3': 1., 'differing_items': 0}))
        self.binding = self.root/'archive_binding.json'
        archive_audit.create_binding(self.tree, self.checkpoint, self.sid, self.export,
                                     output=self.binding, repo=self.root)

    def test_reused_binding_api_rejects_crossed_and_modified_artifacts(self):
        from tools.issue252.final_evaluation import load_bound_checkpoint
        args = dict(checkpoint=self.checkpoint, sid=self.sid, tree=self.tree,
                    binding=self.binding, export_manifest=self.export)
        payload, receipt = load_bound_checkpoint(**args)
        self.assertEqual(payload['run_id'], receipt['run_id'])
        for field in ('sid', 'checkpoint', 'export_manifest'):
            other = self.root/('other-'+Path(args[field]).name)
            other.write_bytes(Path(args[field]).read_bytes())
            with self.subTest(field=field), self.assertRaises(ValueError):
                load_bound_checkpoint(**{**args, field: other})
        self.sid.write_text('{"0":[1,0,0,0]}')
        with self.assertRaisesRegex(ValueError, 'SID'):
            load_bound_checkpoint(**args)
        self.sid.write_text('{"0":[0,0,0,0]}')
        self.checkpoint.write_bytes(self.checkpoint.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            load_bound_checkpoint(**args)

    def test_mutated_executed_source_fails_before_any_model_import(self):
        from tools.issue252.final_evaluation import load_bound_checkpoint
        (self.tree/'model/__init__.py').write_text('VALUE=2\n')
        with self.assertRaisesRegex(ValueError, 'source'):
            load_bound_checkpoint(checkpoint=self.checkpoint, sid=self.sid, tree=self.tree,
                                  binding=self.binding, export_manifest=self.export)


class PerUserRecallTests(unittest.TestCase):
    def test_independent_reduction_uses_global_users_and_zero_based_ranks(self):
        from tools.issue252.reconcile_results import remeasure_test
        populations = ([0, 4, None], [5], [9, 10], [19, None])
        shards = [{'rank': rank, 'n_eval': len(values),
                   'per_user': [{'local_row_index': i, 'target_rank': value}
                                for i, value in enumerate(values)]}
                  for rank, values in enumerate(populations)]
        expected = reference.reduce_user_ranks([value for values in populations for value in values])
        got = remeasure_test(shards)
        self.assertEqual(got['n_eval'], 8)
        self.assertEqual(got['test_recall@5'], .25)
        self.assertEqual(got['test_recall@10'], .5)
        for key in expected:
            self.assertAlmostEqual(got[key], expected[key], places=14)
        with self.assertRaises(ValueError):
            remeasure_test(shards[:-1])
        invalid = copy.deepcopy(shards)
        invalid[0]['per_user'][0]['target_rank'] = True
        with self.assertRaises(ValueError):
            remeasure_test(invalid)


class ReconstructionReductionTests(unittest.TestCase):
    def test_mse_uses_all_coordinates_and_actual_representation(self):
        from tools.issue251.evaluation import evaluate_reconstruction
        target = np.array([[1., 2.], [3., 4.]])
        reconstructed = np.array([[2., 2.], [1., 6.]])
        actual = evaluate_reconstruction(target, reconstructed, representation='hard-q')
        self.assertEqual(actual['mean_squared_error'], 9/4)
        self.assertEqual(actual['source'], 'caller-supplied hard-q reconstruction')
        np.testing.assert_allclose(actual['per_item_l2'], [1., math.sqrt(8)])
        with self.assertRaises(ValueError):
            evaluate_reconstruction(target, reconstructed[:1], representation='hard-q')


class HistoricalArchivedSourceTests(unittest.TestCase):
    def setUp(self):
        from tools.issue252.archive_audit import sha256
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue253')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.live = self.root/'live'
        self.archive = self.root/'archive'
        self.prefix = 'HistoricalArm'
        self.maps = {'candidate_sources': {}, 'infrastructure_sources': {}, 'repository_configs': {}}
        self.files = {}
        entries = (
            ('candidate_sources', 'train_rqvae.py', b'from model import VALUE\n'),
            ('candidate_sources', 'model/__init__.py', b'VALUE=1\n'),
            ('infrastructure_sources', 'tools/issue252/training_admission.py', b'ARM="historical"\n'),
            ('infrastructure_sources', 'tools/issue252/archive_audit.py', b'AUDIT_SCHEMA=1\n'),
            ('repository_configs', 'assets/manifest.json', b'{"generation":"historical"}\n'),
        )
        for group, name, content in entries:
            archived = self.archive/self.prefix/name
            archived.parent.mkdir(parents=True, exist_ok=True)
            archived.write_bytes(content)
            digest = sha256(archived)
            self.maps[group][name] = digest
            self.files[self.prefix+'/'+name] = digest
        self.receipt = {'repository': str(self.live), 'source_tree': str(self.live/'source'),
                        **copy.deepcopy(self.maps),
                        'audit': {'path': str(self.live/'tools/issue252/archive_audit.py'),
                                  'sha256': self.maps['infrastructure_sources']['tools/issue252/archive_audit.py']}}

    def validate(self, **overrides):
        from tools.issue253.final_evaluation import validate_archived_sources
        args = {'files': self.files, 'prefix': self.prefix}
        args.update(overrides)
        return validate_archived_sources(self.receipt, self.archive, **args)

    def test_live_infrastructure_drift_does_not_rewrite_historical_authority(self):
        live = self.live/'tools/issue252/training_admission.py'
        live.parent.mkdir(parents=True)
        live.write_text('ARM="new-three-expert"\n')
        original = copy.deepcopy(self.receipt)
        self.assertEqual(self.validate(), self.maps)
        self.assertEqual(self.receipt, original)
        self.assertEqual(live.read_text(), 'ARM="new-three-expert"\n')

    def test_changed_archived_sources_configs_or_audit_fail_closed(self):
        for group, values in self.maps.items():
            for name in values:
                archived = self.archive/self.prefix/name
                original = archived.read_bytes()
                archived.write_bytes(original+b'changed')
                try:
                    with self.subTest(group=group, name=name), self.assertRaises(ValueError):
                        self.validate()
                finally:
                    archived.write_bytes(original)

    def test_missing_archived_infrastructure_cannot_fall_back_to_live_source(self):
        name = 'tools/issue252/training_admission.py'
        archived = self.archive/self.prefix/name
        live = self.live/name
        live.parent.mkdir(parents=True)
        live.write_bytes(archived.read_bytes())
        archived.unlink()
        with self.assertRaises(ValueError):
            self.validate()

    def test_symlinked_archive_ancestor_is_not_an_immutable_source(self):
        original = self.archive/self.prefix/'tools'
        moved = self.archive/self.prefix/'real_tools'
        original.rename(moved)
        original.symlink_to(moved, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.validate()

    def test_ledger_source_hash_disagreement_and_traversal_are_rejected(self):
        crossed = dict(self.files)
        crossed[self.prefix+'/train_rqvae.py'] = 'c'*64
        with self.assertRaises(ValueError):
            self.validate(files=crossed)
        self.receipt['infrastructure_sources']['../outside.py'] = 'a'*64
        with self.assertRaises(ValueError):
            self.validate()


class FinalProtocolTests(unittest.TestCase):
    def test_nearest_rank_p95_and_inter_epoch_wait_remain_inclusive(self):
        from tools.issue253.reconcile_results import inclusive_window
        rows, start = [], 0.
        for epoch in range(1, 21):
            rows.append({'epoch': epoch, 'start': start, 'end': start+epoch})
            start += epoch+.5
        actual = inclusive_window(rows)
        self.assertEqual(actual['wall_seconds'], 219.5)
        self.assertEqual(actual['mean_seconds'], 219.5/20)
        self.assertEqual(actual['p95_seconds'], 19.5)
        self.assertEqual(actual['inter_boundary_wait_seconds'], 9.5)
        invalid = copy.deepcopy(rows)
        invalid[3]['epoch'] += 1
        with self.assertRaises(ValueError):
            inclusive_window(invalid)

    def test_full_user_coverage_keeps_original_rank_local_populations(self):
        from tools.issue253.reconcile_results import reconcile_per_user
        shards = [{'rank': rank, 'n_eval': len(range(rank, 9, 4)),
                   'per_user': [{'local_row_index': i, 'target_rank': None}
                                for i in range(len(range(rank, 9, 4)))]}
                  for rank in range(4)]
        self.assertEqual(reconcile_per_user(shards, expected_users=9)['n_eval'], 9)
        with self.assertRaises(ValueError):
            reconcile_per_user(shards, expected_users=10)
        # Preserve global denominator but transfer one row across rank ownership.
        shards[0]['per_user'].pop()
        shards[0]['n_eval'] -= 1
        shards[1]['per_user'].append({'local_row_index': 2, 'target_rank': None})
        shards[1]['n_eval'] += 1
        with self.assertRaises(ValueError):
            reconcile_per_user(shards, expected_users=9)

    def test_heldout_broadcast_adapter_preserves_paired_shape(self):
        from tools.issue253.final_evaluation import _broadcast_metric
        left = torch.tensor([[[.2, .4]], [[-.3, .1]]], dtype=torch.float64)
        right = torch.tensor([[[.1, -.2]], [[.3, .5]]], dtype=torch.float64)
        for experts in (3, 5):
            got = _broadcast_metric(experts)(left, right, scale=.8)
            expected = reference.fixed_metrics_numpy(left[:, 0].numpy(), right[:, 0].numpy(),
                scale=.8, paired=True, metrics=(experts,))['fixed'+str(experts)]
            self.assertEqual(got.shape, (2, 1))
            np.testing.assert_allclose(got[:, 0], expected, atol=1e-12)


class StoredIdentityTests(unittest.TestCase):
    """The archived binding stores absolute identities, the archived export
    manifest stores repository-relative ones; both name one generation."""

    def test_relative_and_absolute_identities_resolve_to_one_path(self):
        from tools.issue253.final_evaluation import _identity_path
        relative = 'results/stage2_RQ-VAE/_runs/issue252_formal_r2'
        absolute = str(ROOT/relative)
        self.assertNotEqual(absolute, relative)
        self.assertEqual(_identity_path(absolute), _identity_path(relative))
        self.assertEqual(_identity_path(relative), (ROOT/relative).resolve())

    def test_resolution_uses_the_bound_repository_root(self):
        from tools.issue253.final_evaluation import _identity_path
        relative = 'results/stage2_RQ-VAE/_runs/issue252_formal_rq'
        other = ROOT.parent/'other_repository'
        self.assertEqual(_identity_path(relative, base=other), (other/relative).resolve())
        self.assertNotEqual(_identity_path(relative, base=other), _identity_path(relative))

    def test_empty_or_non_string_identities_fail_closed(self):
        from tools.issue253.final_evaluation import _identity_path
        for value in ('', '   ', None, 7, ['a']):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _identity_path(value)


class ArchiveIdentityTests(unittest.TestCase):
    def test_ledger_needs_exact_sid_and_checkpoint_not_prefix_or_arm_name(self):
        from tools.issue252.reconcile_results import bound_archive
        sid, checkpoint = 'a'*64, 'b'*64
        key = 'TIGER_three_expert_attention_rq/'+sid[:12]
        entry = {'key': key, 'files': {'item_sids.json': sid, 'rqvae_best.pth': checkpoint}}
        self.assertEqual(bound_archive({'101': entry}, key, sid, checkpoint)[0], '101')
        for field in ('item_sids.json', 'rqvae_best.pth'):
            crossed = copy.deepcopy(entry)
            crossed['files'][field] = 'c'*64
            with self.subTest(field=field), self.assertRaises(ValueError):
                bound_archive({'101': crossed}, key, sid, checkpoint)


if __name__ == '__main__':
    unittest.main()
