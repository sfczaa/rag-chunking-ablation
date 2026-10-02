"""Stage 17 answer scoring, prompt text and verdict rules."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag_chunk import answer_eval as AE


class WordTokenizer:
    """Whitespace stand-in for a reader tokenizer."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}

    def decode(self, ids):
        return " ".join(ids)


class ScoringTest(unittest.TestCase):
    def test_normalization_follows_squad(self):
        self.assertEqual(AE.normalize_answer("The  U.S.A. , an Ally"), "usa ally")
        self.assertEqual(AE.normalize_answer(")"), "")

    def test_prediction_keeps_first_line_without_think_block(self):
        self.assertEqual(AE.prediction_from_output("<think>x\ny</think>\n Paris \nmore"),
                         "Paris")
        self.assertEqual(AE.prediction_from_output("  \n"), "")

    def test_containment_is_token_aligned(self):
        self.assertTrue(AE.contains_answer("It aired on March 18, 2018.", "March 18 , 2018"))
        self.assertTrue(AE.contains_answer("the Beatles", "Beatles"))
        self.assertFalse(AE.contains_answer("19901", "1990"))
        self.assertFalse(AE.contains_answer("March 2018", "March 18 , 2018"))
        self.assertFalse(AE.contains_answer("anything", ")"))

    def test_exact_match_and_f1(self):
        self.assertTrue(AE.exact_match("The Beatles.", "beatles"))
        self.assertAlmostEqual(AE.token_f1("john paul george", "paul george ringo"), 2 / 3)
        self.assertEqual(AE.token_f1("", "paris"), 0.0)


class PromptTest(unittest.TestCase):
    def test_messages_match_the_registered_text(self):
        self.assertEqual(
            AE.user_message("who wrote it", [("Title A", "text a"), ("Title B", "text b")]),
            "Answer the question using the passages below. Reply with the answer only, as "
            "a short phrase, without explanation.\n\n[1] Title A\ntext a\n\n[2] Title B\n"
            "text b\n\nQuestion: who wrote it")
        self.assertEqual(
            AE.user_message("who wrote it", None),
            "Answer the question. Reply with the answer only, as a short phrase, without "
            "explanation.\n\nQuestion: who wrote it")

    def test_passage_cap_keeps_the_head(self):
        tok = WordTokenizer()
        self.assertEqual(AE.cut_passage("a b c", tok, 3), ("a b c", False))
        self.assertEqual(AE.cut_passage("a b c d", tok, 3), ("a b c", True))


class VerdictTest(unittest.TestCase):
    def test_mean_ci(self):
        m, lo, hi = AE.mean_ci([1, 0, 1, 0])
        self.assertAlmostEqual(m, 0.5)
        self.assertAlmostEqual(hi - m, 1.96 * (1 / 3) ** 0.5 / 2)

    def test_size_rule(self):
        def diffs(n_pos, n_neg, n=1000):
            return [1] * n_pos + [-1] * n_neg + [0] * (n - n_pos - n_neg)

        self.assertEqual(AE.size_verdict(diffs(80, 20), 0.02)[0], "SIZE-CARRIES")
        self.assertEqual(AE.size_verdict(diffs(1100, 900, 20000), 0.02)[0], "SIZE-SMALL")
        self.assertEqual(AE.size_verdict(diffs(20, 80), 0.02)[0], "SIZE-REVERSED")
        self.assertEqual(AE.size_verdict(diffs(55, 50), 0.02)[0], "SIZE-NOT-DETECTED")

    def test_placement_rule(self):
        def diffs(n_pos, n_neg, n=1031):
            return [1] * n_pos + [-1] * n_neg + [0] * (n - n_pos - n_neg)

        self.assertEqual(AE.placement_verdict(diffs(80, 20), 0.03)[0], "BILSTM-BETTER")
        self.assertEqual(AE.placement_verdict(diffs(20, 80), 0.03)[0], "BILSTM-WORSE")
        self.assertEqual(AE.placement_verdict(diffs(40, 40), 0.03)[0], "EQUIVALENT")
        self.assertEqual(AE.placement_verdict(diffs(250, 250), 0.03)[0], "INCONCLUSIVE")

    def test_reader_check(self):
        self.assertTrue(AE.reader_check([1] * 300 + [0] * 700)[0])
        self.assertFalse(AE.reader_check([1, -1] * 50)[0])


if __name__ == "__main__":
    unittest.main()
