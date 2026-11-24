"""Archive generation identities and the four-piece archive are data contracts."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from tools.issue252 import archive_audit

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "stage2_RQ-VAE").is_dir())


def uploader():
    spec = importlib.util.spec_from_file_location(
        "issue252_uploader", ROOT / "results/research/issue139/upload_stage3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ArchiveBindingTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "results/research/issue252")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.result = self.root / "stage2_RQ-VAE/Arm"
        self.result.mkdir(parents=True)
        self.sid = self.result / "item_sids.json"
        self.sid.write_text('{"0":[0,0,0,0]}')
        self.checkpoint = self.result / "rqvae_best.pth"
        self.source = self.root / "source"
        (self.source / "model").mkdir(parents=True)
        (self.source / "config").mkdir()
        (self.source / "train_rqvae.py").write_text('from model import VALUE\nARM_NAME="Arm"\n')
        (self.source / "model/__init__.py").write_text("VALUE=1\n")
        (self.source / "config/train_config.json").write_text('{"epochs":3000}\n')
        self.payload = {
            "state_dict": {"value": torch.tensor([1.0])},
            "run_id": "actual-run",
            "source_tree": str(self.source),
            "provenance": {
                "candidate_sources": archive_audit.source_hashes(self.source),
                "infrastructure_sources": {},
            },
        }
        torch.save(self.payload, self.checkpoint)
        self.export = self.result / "export_manifest.json"
        self.export.write_text(json.dumps({
            "schema": "genrec.stage2.export_manifest/1",
            "tree": str(self.source), "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": archive_audit.sha256(self.checkpoint),
            "sid": str(self.sid), "sid_sha256": archive_audit.sha256(self.sid),
            "tokens_match": True, "same_l3": 1.0, "differing_items": 0,
        }))
        self.binding = self.result / "archive_binding.json"
        archive_audit.create_binding(self.source, self.checkpoint, self.sid,
                                     self.export, output=self.binding, repo=self.root)
        self.stage3 = self.root / "stage3/logs"
        self.stage3.mkdir(parents=True)
        self.final = self.stage3 / "test_final.json"
        self.block = {
            "test_R@10": .01, "unusual_original_key": {"keep": [1, 2]},
            "sid_path": str(self.sid), "sid_sha256": archive_audit.sha256(self.sid),
            "stage2_checkpoint": str(self.checkpoint),
            "stage2_checkpoint_sha256": archive_audit.sha256(self.checkpoint),
            "archive_binding": str(self.binding),
            "archive_binding_sha256": archive_audit.sha256(self.binding),
            "stage2_run_id": "actual-run",
        }
        self.final.write_text(json.dumps(self.block))
        (self.stage3 / "HG_Rec.log").write_text("complete actual log\nlast line\n")
        (self.stage3 / "training_metrics.jsonl").write_text(
            '{"event":"train_start"}\n{"event":"train_end","epoch":150}\n')
        # This older optional fifth object must never be copied.
        weights = self.stage3 / "HG_Rec_best.pth"
        weights.write_bytes(b"stage3-weights")
        self.block["best_checkpoint"] = str(weights)
        self.final.write_text(json.dumps(self.block))
        self.module = uploader()

    def stage(self, name="staging", tree=None):
        with patch.object(self.module, "RESULTS", self.root), \
                patch.object(self.module, "_latest_test_final", return_value=self.final):
            return self.module.stage("Arm", self.root / name, tree)

    def update_final(self, **fields):
        self.block.update(fields)
        self.final.write_text(json.dumps(self.block))

    def test_stage3_from_another_sid_is_refused_before_copy(self):
        self.update_final(sid_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "SID"):
            self.stage(tree=self.source)
        self.assertFalse((self.root / "staging").exists())

    def test_crossed_checkpoint_is_refused_before_copy(self):
        self.update_final(stage2_checkpoint_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            self.stage()
        self.assertFalse((self.root / "staging").exists())

    def test_mutated_source_is_refused_even_with_explicit_tree(self):
        (self.source / "model/__init__.py").write_text("VALUE=2\n")
        with self.assertRaisesRegex(ValueError, "source"):
            self.stage(tree=self.source)

    def test_another_tree_cannot_override_the_bound_tree(self):
        other = self.root / "different"
        other.mkdir()
        (other / "train_rqvae.py").write_text("VALUE=9\n")
        with self.assertRaisesRegex(ValueError, "source"):
            self.stage(tree=other)

    def test_missing_final_logs_and_binding_fail_closed(self):
        for path in (self.stage3 / "HG_Rec.log", self.stage3 / "training_metrics.jsonl",
                     self.binding, self.final):
            with self.subTest(path=path.name):
                content = path.read_bytes()
                path.unlink()
                try:
                    with self.assertRaises(ValueError):
                        self.stage()
                finally:
                    path.write_bytes(content)
        self.assertFalse((self.root / "staging").exists())

    def test_no_nearby_checkpoint_substitution(self):
        self.checkpoint.rename(self.result / "other-generation.pth")
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            self.stage()

    def test_correct_bundle_preserves_recall_logs_and_reproducible_tar(self):
        files, key = self.stage()
        staging = self.root / "staging"
        self.assertEqual({p.name for p in staging.iterdir()},
                         {"Arm", "item_sids.json", "rqvae_best.pth", "recall.json"})
        self.assertEqual(key, "Arm/" + archive_audit.sha256(self.sid)[:12])
        recall = json.loads((staging / "recall.json").read_text())
        for name, value in self.block.items():
            self.assertEqual(recall[name], value)
        evidence = recall["stage3_archive_evidence"]
        for name in ("HG_Rec.log", "training_metrics.jsonl"):
            self.assertEqual(evidence[name]["content"], (self.stage3 / name).read_text())
            self.assertEqual(evidence[name]["sha256"], archive_audit.sha256(self.stage3 / name))
        self.assertEqual((staging / "rqvae_best.pth").read_bytes(), self.checkpoint.read_bytes())
        archive = staging / "Arm/stage2_source.tar.gz"
        with tarfile.open(archive) as bundle:
            for name in ("train_rqvae.py", "model/__init__.py", "config/train_config.json"):
                self.assertEqual(bundle.extractfile(name).read(), (self.source / name).read_bytes())
            self.assertEqual(json.loads(bundle.extractfile("archive_binding.json").read()),
                             json.loads(self.binding.read_text()))
        self.stage("second")
        self.assertEqual(archive.read_bytes(),
                         (self.root / "second/Arm/stage2_source.tar.gz").read_bytes())
        self.assertIn("Arm/stage2_source.tar.gz", files)
    def test_nested_immutable_generation_archives_its_adapter_and_exact_inputs(self):
        import torch
        from tools.issue252.run_workspace import freeze_run
        tree=self.root/'stage2_RQ-VAE/Candidate'
        (tree/'model').mkdir(parents=True)
        (tree/'train_rqvae.py').write_text('VALUE=1\n')
        (tree/'model/__init__.py').write_text('VALUE=2\n')
        run=freeze_run(self.root,tree,'actual-run','Arm')
        self.source=Path(run['source']);self.result=Path(run['result'])
        self.sid=self.result/'item_sids.json';self.sid.write_text('{"0":[0,0,0,0]}')
        self.checkpoint=self.result/'rqvae_best.pth'
        self.payload.update(source_tree=str(self.source),provenance={
            'candidate_sources':archive_audit.source_hashes(self.source),
            'infrastructure_sources':{}})
        torch.save(self.payload,self.checkpoint)
        self.export=self.result/'export_manifest.json'
        self.export.write_text(json.dumps({
            'schema':'genrec.stage2.export_manifest/1','tree':str(self.source),
            'checkpoint':str(self.checkpoint),'checkpoint_sha256':archive_audit.sha256(self.checkpoint),
            'sid':str(self.sid),'sid_sha256':archive_audit.sha256(self.sid),
            'tokens_match':True,'same_l3':1.0,'differing_items':0}))
        self.binding=self.result/'archive_binding.json'
        archive_audit.create_binding(self.source,self.checkpoint,self.sid,self.export,
                                     output=self.binding,repo=self.root)
        self.update_final(sid_path=str(self.sid),sid_sha256=archive_audit.sha256(self.sid),
                          stage2_checkpoint=str(self.checkpoint),
                          stage2_checkpoint_sha256=archive_audit.sha256(self.checkpoint),
                          archive_binding=str(self.binding),
                          archive_binding_sha256=archive_audit.sha256(self.binding))
        with patch.object(self.module,'REPO',self.root):
            self.stage()
            with tarfile.open(self.root/'staging/Arm/stage2_source.tar.gz') as bundle:
                self.assertEqual(bundle.extractfile('execution_entry.py').read(),
                                 Path(run['entry']).read_bytes())
            with patch.object(self.module,'_latest_test_final',return_value=self.final):
                with self.assertRaisesRegex(ValueError,'arm'):
                    self.module.stage('WrongArm',self.root/'crossed-arm')
        self.assertEqual({p.name for p in (self.root/'staging').iterdir()},
                         {'Arm','item_sids.json','rqvae_best.pth','recall.json'})

    def test_original_evidence_key_is_not_clobbered(self):
        self.update_final(stage3_archive_evidence={"already": "owned"})
        with self.assertRaisesRegex(ValueError, "key"):
            self.stage()

    def test_binding_refuses_source_not_recorded_in_checkpoint(self):
        (self.source / "model/__init__.py").write_text("VALUE=7\n")
        with self.assertRaisesRegex(ValueError, "source"):
            archive_audit.create_binding(self.source, self.checkpoint, self.sid,
                                         self.export, repo=self.root)

    def test_mathematical_contract_is_issue252_and_archived(self):
        current = self.root / "results/research/issue252/CONTRACT.md"
        previous = self.root / "results/research/issue251/CONTRACT.md"
        current.parent.mkdir(parents=True)
        previous.parent.mkdir(parents=True)
        current.write_text("frozen issue252 mathematics\n")
        previous.write_text("different issue251 contract\n")
        provenance = {"mathematical_contract_sha256": archive_audit.sha256(current)}
        self.assertEqual(archive_audit._repository_configs(provenance, self.root),
                         {"results/research/issue252/CONTRACT.md": archive_audit.sha256(current)})
        current.write_text("changed current contract\n")
        # The contract document is resolved by the digest the checkpoint holds,
        # so a changed document is no longer the bound one at all.
        with self.assertRaisesRegex(ValueError, "contract"):
            archive_audit._repository_configs(provenance, self.root)

    def test_checkpoint_names_the_contract_document_it_was_bound_to(self):
        current = self.root / "results/research/issue253/CONTRACT.md"
        current.parent.mkdir(parents=True)
        current.write_text("frozen issue253 mathematics\n")
        digest = archive_audit.sha256(current)
        named = {"mathematical_contract_path": "results/research/issue253/CONTRACT.md",
                 "mathematical_contract_sha256": digest}
        self.assertEqual(archive_audit._repository_configs(named, self.root),
                         {"results/research/issue253/CONTRACT.md": digest})
        # A digest that matches no contract document is not a binding.
        with self.assertRaisesRegex(ValueError, "contract"):
            archive_audit._repository_configs({"mathematical_contract_sha256": "0"*64}, self.root)
        for unsafe in ("/etc/passwd", "../escape/CONTRACT.md"):
            with self.subTest(path=unsafe), self.assertRaisesRegex(ValueError, "contract"):
                archive_audit._repository_configs(
                    {"mathematical_contract_path": unsafe,
                     "mathematical_contract_sha256": digest}, self.root)

    def test_binding_refuses_crossed_export_and_missing_run_identity(self):
        import torch
        manifest = json.loads(self.export.read_text())
        manifest["checkpoint_sha256"] = "0" * 64
        self.export.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            archive_audit.create_binding(self.source, self.checkpoint, self.sid,
                                         self.export, repo=self.root)
        self.payload.pop("run_id")
        torch.save(self.payload, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "run"):
            archive_audit.create_binding(self.source, self.checkpoint, self.sid,
                                         self.export, repo=self.root)

    def test_imported_repo_source_and_config_are_archived_and_checked(self):
        import torch
        (self.root / "tools/demo").mkdir(parents=True)
        external = self.root / "tools/demo/dependency.py"
        external.write_text("VALUE=8\n")
        (self.source / "model/__init__.py").write_text("from tools.demo.dependency import VALUE\n")
        self.payload["provenance"] = {
            "candidate_sources": archive_audit.source_hashes(self.source),
            "infrastructure_sources": archive_audit.imported_repo_sources(self.source, self.root),
        }
        self.assertEqual(set(self.payload["provenance"]["infrastructure_sources"]),
                         {"tools/demo/dependency.py"})
        torch.save(self.payload, self.checkpoint)
        manifest = json.loads(self.export.read_text())
        manifest["checkpoint_sha256"] = archive_audit.sha256(self.checkpoint)
        self.export.write_text(json.dumps(manifest))
        archive_audit.create_binding(self.source, self.checkpoint, self.sid,
                                     self.export, output=self.binding, repo=self.root)
        self.update_final(stage2_checkpoint_sha256=archive_audit.sha256(self.checkpoint),
                          archive_binding_sha256=archive_audit.sha256(self.binding))
        self.stage()
        with tarfile.open(self.root / "staging/Arm/stage2_source.tar.gz") as bundle:
            self.assertEqual(bundle.extractfile("tools/demo/dependency.py").read(),
                             external.read_bytes())
        external.write_text("VALUE=9\n")
        with self.assertRaisesRegex(ValueError, "source"):
            self.stage("mutated")


    def test_source_import_closure_handles_relative_imports_cycles_and_missing_modules(self):
        directory = self.root / "tools/demo"
        directory.mkdir(parents=True)
        (directory / "__init__.py").write_text("from .a import VALUE\n")
        (directory / "a.py").write_text("from . import b\nVALUE=1\n")
        (directory / "b.py").write_text("from . import a\n")
        (self.source / "model/__init__.py").write_text("from tools.demo import VALUE\n")
        self.assertEqual(set(archive_audit.imported_repo_sources(self.source, self.root)),
                         {"tools/demo/__init__.py", "tools/demo/a.py", "tools/demo/b.py"})
        (directory / "a.py").write_text("from tools.demo.nonexistent import VALUE\n")
        with self.assertRaisesRegex(ValueError, "missing required"):
            archive_audit.imported_repo_sources(self.source, self.root)

    def test_binding_requires_every_token_to_match_export(self):
        manifest = json.loads(self.export.read_text())
        manifest.update(same_l3=.99999, differing_items=1)
        self.export.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "every checkpoint token"):
            archive_audit.create_binding(self.source, self.checkpoint, self.sid,
                                         self.export, repo=self.root)

    def test_checkpoint_bound_asset_configuration_travels_with_source(self):
        import torch
        manifest = self.root / "assets/manifest.json"
        manifest.parent.mkdir()
        manifest.write_text('{"assets":[]}\n')
        self.payload["provenance"]["asset_manifest_sha256"] = archive_audit.sha256(manifest)
        torch.save(self.payload, self.checkpoint)
        export = json.loads(self.export.read_text())
        export["checkpoint_sha256"] = archive_audit.sha256(self.checkpoint)
        self.export.write_text(json.dumps(export))
        archive_audit.create_binding(self.source, self.checkpoint, self.sid,
                                     self.export, output=self.binding, repo=self.root)
        self.update_final(stage2_checkpoint_sha256=archive_audit.sha256(self.checkpoint),
                          archive_binding_sha256=archive_audit.sha256(self.binding))
        self.stage()
        with tarfile.open(self.root / "staging/Arm/stage2_source.tar.gz") as bundle:
            self.assertEqual(bundle.extractfile("assets/manifest.json").read(), manifest.read_bytes())
        manifest.write_text('{"assets":["changed"]}\n')
        with self.assertRaisesRegex(ValueError, "config"):
            self.stage("changed-config")

    def test_incomplete_receipt_is_a_value_error_not_an_unbound_archive(self):
        receipt = json.loads(self.binding.read_text())
        for field in ("run_id", "repository", "source_tree"):
            with self.subTest(field=field):
                damaged = dict(receipt)
                damaged.pop(field)
                self.binding.write_text(json.dumps(damaged))
                with self.assertRaises(ValueError):
                    archive_audit.validate_binding(self.binding, self.sid, self.checkpoint)

    def test_ambiguous_stage3_generations_are_not_selected_by_mtime(self):
        logs = self.root / "stage3/Arm/logs/dataset"
        for run in ("first", "newer"):
            directory = logs / run
            directory.mkdir(parents=True)
            (directory / "test_final.json").write_text("{}")
        with patch.object(self.module, "STAGE3_LOGS", self.root / "stage3"):
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                self.module._latest_test_final("Arm")

    def test_incomplete_log_stream_is_not_archived(self):
        (self.stage3 / "training_metrics.jsonl").write_text('{"event":"test"}\n')
        with self.assertRaisesRegex(ValueError, "train_end"):
            self.stage()
        self.assertFalse((self.root / "staging").exists())

    def test_stage3_captures_actual_resolved_input_identity(self):
        # Execute the provenance helpers without importing the GPU training module.
        path = ROOT / "stage3_T5Train/train_HG-Rec.py"
        syntax = ast.parse(path.read_text())
        functions = [node for node in syntax.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("_sha256_file", "_stage2_input_provenance")]
        namespace = {"Path": Path, "hashlib": hashlib, "json": json,
                     "SCRIPT_DIR": ROOT / "stage3_T5Train"}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
        record = namespace["_stage2_input_provenance"](self.sid)
        self.assertEqual(record["sid_sha256"], archive_audit.sha256(self.sid))
        self.assertEqual(record["stage2_checkpoint_sha256"], archive_audit.sha256(self.checkpoint))
        self.assertEqual(record["archive_binding_sha256"], archive_audit.sha256(self.binding))
        self.checkpoint.write_bytes(b"another checkpoint")
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            namespace["_stage2_input_provenance"](self.sid)


if __name__ == "__main__":
    unittest.main()
