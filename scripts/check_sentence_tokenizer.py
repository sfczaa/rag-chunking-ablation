"""Check the pinned English tables against recorded NLTK 3.10.3 outputs."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag_chunk.sentence_tokenizer import sent_tokenize


def main():
    fixture = json.loads((ROOT / 'tests/fixtures/punkt_english.json').read_text(encoding='utf-8'))
    for index, case in enumerate(fixture['cases']):
        actual = sent_tokenize(case['text'])
        if actual != case['sentences']:
            raise AssertionError(f'English Punkt regression in case {index}: {actual!r}')
    print(f"{len(fixture['cases'])} English Punkt cases passed.")


if __name__ == '__main__':
    main()
