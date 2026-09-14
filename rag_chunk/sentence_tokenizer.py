"""English Punkt inference using a verified, immutable parameter archive."""

from collections import defaultdict
from functools import lru_cache
import hashlib
import io
from pathlib import Path
import tempfile
from urllib.request import urlopen
from zipfile import ZipFile

import config as C
from rag_chunk._vendor.punkt import PunktParameters, PunktSentenceTokenizer

DATA_REVISION = "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"
ARCHIVE_SHA256 = "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106"
ARCHIVE_URL = (f"https://raw.githubusercontent.com/nltk/nltk_data/{DATA_REVISION}"
               "/packages/tokenizers/punkt_tab.zip")
MAX_ARCHIVE_BYTES = 8 * 1024 * 1024


def _verify_archive(data: bytes) -> None:
    if hashlib.sha256(data).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Punkt archive checksum mismatch.")


def _load_params(data: bytes) -> PunktParameters:
    _verify_archive(data)
    with ZipFile(io.BytesIO(data)) as archive:
        def lines(name):
            content = archive.read(f"punkt_tab/english/{name}").decode("utf-8")
            return [line.removesuffix("\n") for line in io.StringIO(content)]

        params = PunktParameters()
        params.collocations = {tuple(line.split("\t")) for line in lines("collocations.tab")}
        params.sent_starters = set(lines("sent_starters.txt"))
        params.abbrev_types = set(lines("abbrev_types.txt"))
        params.ortho_context = defaultdict(int, {
            word: int(value) for word, value in
            (line.split("\t") for line in lines("ortho_context.tab"))})
    return params


@lru_cache(maxsize=4)
def _tokenizer(cache_dir: Path) -> PunktSentenceTokenizer:
    cache = cache_dir / f"punkt_tab-{ARCHIVE_SHA256}.zip"
    if cache.exists():
        with cache.open("rb") as handle:
            data = handle.read(MAX_ARCHIVE_BYTES + 1)
        try:
            _verify_archive(data)
        except ValueError as exc:
            raise ValueError(f"Punkt archive checksum mismatch: {cache}. "
                             "Remove this cached file and retry.") from exc
    else:
        with urlopen(ARCHIVE_URL, timeout=60) as response:
            data = response.read(MAX_ARCHIVE_BYTES + 1)
        _verify_archive(data)
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(data)
            temporary.replace(cache)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return PunktSentenceTokenizer(params=_load_params(data))


def sent_tokenize(text: str) -> list[str]:
    """Split English text; download verified tables on the first uncached use."""
    if not isinstance(text, str):
        raise TypeError("Sentence tokenizer input must be text.")
    if not text.strip():
        return []
    return _tokenizer(C.DATA_DIR / "tokenizers").tokenize(text)
