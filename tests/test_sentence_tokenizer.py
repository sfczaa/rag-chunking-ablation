"""Punkt inference and verified parameter-cache boundaries."""

import hashlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag_chunk import sentence_tokenizer as tokenizer
from rag_chunk.wiki_data import split_sentences


def fixture_archive():
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        for name, content in {
            "abbrev_types.txt": "dr\nmr",
            "sent_starters.txt": "he\nshe\nthen",
            "collocations.tab": "s\tbach",
            "ortho_context.tab": "smith\t4\narrived\t32",
        }.items():
            archive.writestr(f"punkt_tab/english/{name}", content)
    return buffer.getvalue()


class SentenceTokenizerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = fixture_archive()
        self.sha = hashlib.sha256(self.data).hexdigest()
        self.addCleanup(patch.stopall)
        patch.object(tokenizer, "ARCHIVE_SHA256", self.sha).start()
        patch.object(tokenizer.C, "DATA_DIR", self.root).start()
        tokenizer._tokenizer.cache_clear()
        self.addCleanup(tokenizer._tokenizer.cache_clear)

    def seed_cache(self):
        cache = self.root / "tokenizers" / f"punkt_tab-{self.sha}.zip"
        cache.parent.mkdir(exist_ok=True)
        cache.write_bytes(self.data)
        return cache

    def test_abbreviations_quotes_and_paragraphs(self):
        self.seed_cache()
        for text, expected in [
            ('Dr. Smith arrived. He said, "Go now." Then he left.',
             ['Dr. Smith arrived.', 'He said, "Go now."', 'Then he left.']),
            ('First paragraph.\r\n\r\nSecond paragraph!', ['First paragraph.', 'Second paragraph!']),
            ('The value is 3.14. Another value is 2.71.', ['The value is 3.14.', 'Another value is 2.71.']),
            ('No terminal punctuation', ['No terminal punctuation']),
        ]:
            with self.subTest(text=text):
                self.assertEqual(tokenizer.sent_tokenize(text), expected)

    def test_empty_text_needs_no_download(self):
        with patch.object(tokenizer, "urlopen") as request:
            for text in ('', ' ', '\n\r\n'):
                self.assertEqual(tokenizer.sent_tokenize(text), [])
        request.assert_not_called()

    def test_nontext_rejected(self):
        with self.assertRaises(TypeError):
            tokenizer.sent_tokenize(None)

    def test_strip_then_minimum_length_filter(self):
        self.seed_cache()
        self.assertEqual(split_sentences('  Short. This sentence is long enough to retain.  '),
                         ['This sentence is long enough to retain.'])

    def test_cached_tables_work_offline(self):
        self.seed_cache()
        with patch.object(tokenizer, "urlopen", side_effect=AssertionError("network used")):
            self.assertEqual(tokenizer.sent_tokenize('A sentence. Another sentence.'),
                             ['A sentence.', 'Another sentence.'])

    def test_corrupt_cache_rejected_before_zip_parsing(self):
        self.seed_cache().write_bytes(b'not the pinned archive')
        with patch.object(tokenizer, "ZipFile") as archive:
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                tokenizer.sent_tokenize('A sentence.')
        archive.assert_not_called()

    def test_download_is_bounded_and_verified_before_caching(self):
        response = Mock()
        response.read.return_value = self.data
        context = Mock()
        context.__enter__ = Mock(return_value=response)
        context.__exit__ = Mock(return_value=False)
        with patch.object(tokenizer, "urlopen", return_value=context) as request:
            tokenizer.sent_tokenize('A sentence.')
        request.assert_called_once_with(tokenizer.ARCHIVE_URL, timeout=60)
        response.read.assert_called_once_with(tokenizer.MAX_ARCHIVE_BYTES + 1)
        self.assertEqual(list((self.root / 'tokenizers').iterdir())[0].read_bytes(), self.data)

    def test_bad_download_is_not_cached(self):
        response = Mock()
        response.read.return_value = b'changed upstream content'
        context = Mock()
        context.__enter__ = Mock(return_value=response)
        context.__exit__ = Mock(return_value=False)
        with patch.object(tokenizer, "urlopen", return_value=context):
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                tokenizer.sent_tokenize('A sentence.')
        self.assertFalse((self.root / 'tokenizers').exists())

    def test_failed_write_removes_temporary_file(self):
        pending = self.root / 'pending.zip'
        pending.write_bytes(b'partial download')
        response = Mock()
        response.__enter__ = Mock(return_value=io.BytesIO(self.data))
        response.__exit__ = Mock(return_value=False)
        handle = Mock(name=str(pending))
        handle.name = str(pending)
        handle.write.side_effect = OSError('disk full')
        context = Mock()
        context.__enter__ = Mock(return_value=handle)
        context.__exit__ = Mock(return_value=False)
        with patch.object(tokenizer, 'urlopen', return_value=response), \
             patch.object(tokenizer.tempfile, 'NamedTemporaryFile', return_value=context):
            with self.assertRaisesRegex(OSError, 'disk full'):
                tokenizer.sent_tokenize('A sentence.')
        self.assertFalse(pending.exists())

    def test_corrupt_cache_error_identifies_file(self):
        path = self.seed_cache()
        path.write_bytes(b'corrupt')
        with self.assertRaises(ValueError) as caught:
            tokenizer.sent_tokenize('A sentence.')
        self.assertIn(str(path), str(caught.exception))

    def test_preprocessing_runs_without_site_packages(self):
        self.seed_cache()
        code = '''import sys
from pathlib import Path
from rag_chunk import sentence_tokenizer as t
from rag_chunk.wiki_data import split_sentences
t.C.DATA_DIR = Path(sys.argv[1])
t.ARCHIVE_SHA256 = sys.argv[2]
assert split_sentences("Short. This sentence is long enough to retain.") == ["This sentence is long enough to retain."]
assert not any(name == "nltk" or name.startswith("nltk.") for name in sys.modules)
'''
        subprocess.run([sys.executable, '-S', '-c', code, str(self.root), self.sha],
                       cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True)


if __name__ == '__main__':
    unittest.main()
