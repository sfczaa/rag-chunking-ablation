# Punkt inference

`punkt.py` contains the inference subset of `nltk/tokenize/punkt.py` from the
official NLTK 3.10.3 PyPI wheel, under Apache-2.0. Copyright and author notices
remain in the source; the full license is in `LICENSE-NLTK.txt`.
Source archive and file hashes are recorded in `punkt-source.json`.

The retained classes, constants and methods are copied without algorithm
changes. `PunktSentenceTokenizer` inherits `PunktBaseClass` directly and uses
its constructor with preloaded parameters. Training, pickle loading, model
export, debug file writes, the NLTK loader and the unused `TokenizerI` base
are excluded. The retained sentence methods are listed in the manifest.

English parameter tables are downloaded separately from the pinned official
NLTK data archive by `rag_chunk/sentence_tokenizer.py`. They are not bundled
in this repository. No NLTK package import is required.
