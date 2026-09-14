"""Demo input boundaries, artifact pins and embedding cache behavior."""

import ast
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy" / "space"))
from demo_settings import ARTIFACTS, MAX_QUERY_CHARS, artifact_location, validate_query


class DemoInputTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse((ROOT / "deploy/space/app.py").read_text(encoding="utf-8"))
        callback = next(n for n in ast.walk(tree)
                        if isinstance(n, ast.FunctionDef) and n.name == "run")
        self.retrieve = Mock(return_value=[])
        self.scope = {
            "validate_query": validate_query,
            "gr": types.SimpleNamespace(Error=ValueError),
            "label_to_arm": {"dense": "bge"}, "ARM_BGE": "bge",
            "by_label": {}, "_retrieve_and_rank": self.retrieve,
            "DEMO_CONFIGS": (), "ARM_LABELS": {"bge": "dense"},
        }
        exec(compile(ast.Module(body=[callback], type_ignores=[]),
                     "app.py:run", "exec"), self.scope)

    def test_overlong_input_never_reaches_inference(self):
        for text in ("x" * (MAX_QUERY_CHARS + 1),
                     " " * MAX_QUERY_CHARS + "x",
                     "問" * (MAX_QUERY_CHARS + 1)):
            with self.subTest(length=len(text)), self.assertRaisesRegex(ValueError, "2,000"):
                self.scope["run"](None, text, "dense")
        self.retrieve.assert_not_called()

    def test_boundary_length_reaches_inference_unchanged(self):
        self.retrieve.side_effect = RuntimeError("inference reached")
        text = "問" * MAX_QUERY_CHARS
        with self.assertRaisesRegex(RuntimeError, "inference reached"):
            self.scope["run"](None, text, "dense")
        self.retrieve.assert_called_once_with(text, "bge")

    def test_empty_input_preserves_selection_prompt(self):
        self.assertEqual(self.scope["run"](None, "   ", "dense"),
                         ("Pick a bench question or type your own.", "", ""))
        self.retrieve.assert_not_called()

    def test_query_normalization_and_type(self):
        self.assertEqual(validate_query("  hello  "), "hello")
        self.assertEqual(validate_query(None), "")
        for value in (1, [], {}):
            with self.assertRaisesRegex(ValueError, "must be text"):
                validate_query(value)


class ArtifactTests(unittest.TestCase):
    def test_default_artifacts_use_full_commit_pins(self):
        for name in ARTIFACTS:
            repo, revision = artifact_location(name)
            self.assertIn("/", repo)
            self.assertRegex(revision, r"^[0-9a-f]{40}$")

    def test_custom_remote_requires_its_own_revision(self):
        with patch.dict(os.environ, {"ASSETS_REPO": "example/data"}, clear=True):
            with self.assertRaisesRegex(ValueError, "ASSETS_REVISION is required"):
                artifact_location("assets", "ASSETS_REPO", "ASSETS_REVISION")
            os.environ["ASSETS_REVISION"] = "main"
            with self.assertRaisesRegex(ValueError, "full commit"):
                artifact_location("assets", "ASSETS_REPO", "ASSETS_REVISION")
            os.environ["ASSETS_REVISION"] = "a" * 40
            self.assertEqual(artifact_location("assets", "ASSETS_REPO", "ASSETS_REVISION"),
                             ("example/data", "a" * 40))

    def test_local_finetuned_smoke_override_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict(os.environ, {"FT_REPO": folder}, clear=True):
                self.assertEqual(artifact_location("finetuned", "FT_REPO", "FT_REVISION"),
                                 (folder, None))

    def test_space_payload_contains_runtime_settings(self):
        spec = importlib.util.spec_from_file_location("space_packager", ROOT / "deploy/create_space.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            payload = module.assemble(Path(folder))
            self.assertEqual((payload / "demo_settings.py").read_bytes(),
                             (ROOT / "deploy/space/demo_settings.py").read_bytes())


class EmbeddingRevisionTests(unittest.TestCase):
    def test_revision_reaches_loader_and_separates_cache(self):
        config = types.SimpleNamespace(EMBED_MODEL="boundary", BOUNDARY_EMBED_MODEL="boundary",
                                       RETRIEVAL_EMBED_MODEL="retrieval", RETRIEVAL_EMBED_REVISION="a" * 40)
        constructor = Mock(side_effect=lambda *a, **k: object())
        modules = {"config": config,
                   "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
                   "sentence_transformers": types.SimpleNamespace(SentenceTransformer=constructor)}
        spec = importlib.util.spec_from_file_location("embedding_under_test", ROOT / "rag_chunk/embedding.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(module)
            first = module.get_embedder("retrieval")
            self.assertIs(first, module.get_embedder("retrieval"))
            constructor.assert_called_once_with("retrieval", device="cpu", revision="a" * 40)
            config.RETRIEVAL_EMBED_REVISION = "b" * 40
            self.assertIsNot(first, module.get_embedder("retrieval"))
            constructor.assert_called_with("retrieval", device="cpu", revision="b" * 40)
            module.get_embedder("boundary")
            constructor.assert_called_with("boundary", device="cpu")
            module.get_embedder("retrieval", model_name="different-model")
            constructor.assert_called_with("different-model", device="cpu")


if __name__ == "__main__":
    unittest.main()
