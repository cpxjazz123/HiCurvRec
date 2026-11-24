"""Remote archive bytes must pass before the normal uploader persists its ledger."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "stage2_RQ-VAE").is_dir())


def uploader():
    spec = importlib.util.spec_from_file_location(
        "issue252_verified_uploader", ROOT / "results/research/issue139/upload_stage3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RemoteCommand:
    """Only the rclone command boundary is replaced; verification is production code."""

    def __init__(self, objects, failure=None):
        self.objects = dict(objects)
        self.failure = failure
        self.calls = []
        self.reads = []

    def __call__(self, args, timeout=3600, *, text=True, consume=None):
        self.calls.append(list(args))
        operation = args[0]
        if operation == self.failure:
            return subprocess.CompletedProcess(args, 1, "" if text else b"",
                                               "credential-must-not-be-logged")
        if operation == "copy":
            # Preserve the specified remote fixture, including stale/missing bytes.
            return subprocess.CompletedProcess(args, 0, "", "")
        if operation == "lsf":
            return subprocess.CompletedProcess(args, 0,
                                               "\n".join(sorted(self.objects)) + "\n", "")
        if operation == "cat":
            relative = args[1].split("/51/", 1)[1]
            self.reads.append(relative)
            if consume is None or text:
                raise AssertionError("verification must stream binary cat output into its hasher")
            if relative not in self.objects:
                return subprocess.CompletedProcess(args, 1, b"", b"missing")
            data = self.objects[relative]
            if consume is not None:
                for start in range(0, len(data), 3):
                    consume(data[start:start + 3])
                data = b""
            return subprocess.CompletedProcess(args, 0, data, b"")
        raise AssertionError("unexpected command: " + repr(args))


class RemoteVerificationTests(unittest.TestCase):
    def setUp(self):
        self.module = uploader()
        self.layout = sys.modules["r2_layout"]
        temporary = tempfile.TemporaryDirectory(dir=ROOT / "results/research/issue252")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.objects = {
            "Arm/train_rqvae.py": b"ARM_NAME = 'Arm'\n",
            "Arm/model/layer.py": b"bound model bytes\n",
            "Arm/stage2_source.tar.gz": b"complete source archive",
            "item_sids.json": b'{"0":[1,2,3,0]}',
            "rqvae_best.pth": bytes(range(256)) * 17,
            "recall.json": b'{"test_R@10":0.07}',
        }
        self.files = {name: hashlib.sha256(data).hexdigest()
                      for name, data in self.objects.items()}
        self.ledger_path = self.root / "ledger.json"
        self.old_entry = {"key": "old-arm/old-sid", "files": {"old/file": "old-digest"}}
        self.ledger_path.write_text(json.dumps({"50": self.old_entry}))
        self.original_ledger = self.ledger_path.read_bytes()

    def stage(self, arm, staging, tree=None):
        self.assertEqual(arm, "Arm")
        for name, data in self.objects.items():
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return dict(self.files), "Arm/sid-digest"

    def run_upload(self, remote, *, failure=None):
        command = RemoteCommand(remote, failure)
        ledger = self.layout.Ledger(self.ledger_path)
        output = io.StringIO()
        original_record, original_save = ledger.record, ledger.save

        def record_verified(*args, **kwargs):
            self.assertEqual(sorted(command.reads), sorted(self.files))
            return original_record(*args, **kwargs)

        def save_verified():
            self.assertEqual(sorted(command.reads), sorted(self.files))
            return original_save()

        with patch.object(self.module, "RESEARCH", self.root), \
                patch.object(self.module, "stage", side_effect=self.stage), \
                patch.object(self.module, "Ledger", return_value=ledger), \
                patch.object(self.layout, "rclone", side_effect=command), \
                patch.object(ledger, "record", side_effect=record_verified) as record, \
                patch.object(ledger, "save", side_effect=save_verified) as save, \
                patch.object(sys, "argv", ["upload_stage3.py", "Arm"]), \
                contextlib.redirect_stdout(output):
            try:
                result = self.module.main()
            except (RuntimeError, ValueError, SystemExit) as error:
                self.assertEqual(self.ledger_path.read_bytes(), self.original_ledger)
                record.assert_not_called()
                save.assert_not_called()
                self.assertNotIn("uploaded to ", output.getvalue())
                self.assertNotIn("credential-must-not-be-logged", str(error))
                return command, error
        self.assertEqual(result, 0)
        save.assert_called_once_with()
        self.assertIn("uploaded to r2:genrec/51/", output.getvalue())
        return command, None

    def test_valid_remote_streams_every_object_before_ledger_save(self):
        command, error = self.run_upload(self.objects)
        self.assertIsNone(error)
        self.assertEqual(sorted(command.reads), sorted(self.files))
        ledger = json.loads(self.ledger_path.read_text())
        self.assertEqual(ledger["50"], self.old_entry)
        self.assertEqual(ledger["51"]["files"], self.files)
        self.assertFalse(any("/50" in arg for call in command.calls for arg in call))

    def test_explicit_verifier_consumes_all_remote_bytes(self):
        command = RemoteCommand(self.objects)
        self.layout.verify_upload("51", self.files, runner=command)
        self.assertEqual(sorted(command.reads), sorted(self.files))
        self.assertEqual({name.split("/", 1)[0] for name in self.files},
                         {"Arm", "item_sids.json", "rqvae_best.pth", "recall.json"})

    def test_corrupt_checkpoint_never_records_or_saves(self):
        remote = dict(self.objects)
        remote["rqvae_best.pth"] = remote["rqvae_best.pth"][:-1] + b"X"
        _, error = self.run_upload(remote)
        self.assertIsNotNone(error)

    def test_stale_source_bytes_never_record_or_save(self):
        remote = dict(self.objects)
        remote["Arm/model/layer.py"] = b"older source generation\n"
        _, error = self.run_upload(remote)
        self.assertIsNotNone(error)

    def test_missing_object_never_records_or_saves(self):
        remote = dict(self.objects)
        del remote["Arm/model/layer.py"]
        _, error = self.run_upload(remote)
        self.assertIsNotNone(error)

    def test_extra_fifth_entry_never_records_or_saves(self):
        remote = dict(self.objects, **{"HG_Rec_best.pth": b"unwanted fifth entry"})
        _, error = self.run_upload(remote)
        self.assertIsNotNone(error)

    def test_extra_nested_object_never_records_or_saves(self):
        remote = dict(self.objects, **{"Arm/model/obsolete.py": b"stale object"})
        _, error = self.run_upload(remote)
        self.assertIsNotNone(error)

    def test_failed_cat_never_records_or_saves(self):
        _, error = self.run_upload(self.objects, failure="cat")
        self.assertIsNotNone(error)

    def test_failed_listing_never_records_or_saves(self):
        _, error = self.run_upload(self.objects, failure="lsf")
        self.assertIsNotNone(error)

    def test_failed_upload_never_records_or_saves(self):
        _, error = self.run_upload(self.objects, failure="copy")
        self.assertIsNotNone(error)


if __name__ == "__main__":
    unittest.main()
