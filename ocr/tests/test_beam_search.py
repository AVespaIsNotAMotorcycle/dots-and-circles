import itertools
import math

import numpy as np
import pytest

from ocr.beam_search import Decoder, align, beam_search, log_softmax
from ocr.char_lm import CharLM

# Three letters and a blank, class 3
LETTERS = ["a", "b", "c", None]
BLANK = 3
A, B, C, _ = [[9 if i == j else 0 for j in range(4)] for i in range(4)]
GAP = [0, 0, 0, 9]


def read(logits, **kwargs):
    log_probs = log_softmax(np.array(logits, dtype=float))
    return beam_search(log_probs, BLANK, LETTERS, **kwargs)


def greedy(logits):
    out, previous = "", None
    for index in np.array(logits).argmax(axis=-1):
        if index != previous and index != BLANK:
            out += LETTERS[index]
        previous = index
    return out


def likeliest(log_probs):
    # The reading with the most probability over every alignment: only for
    # tiny inputs
    totals = {}
    for path in itertools.product(range(len(LETTERS)),
                                  repeat=len(log_probs)):
        reading, previous = "", None
        for index in path:
            if index != previous and index != BLANK:
                reading += LETTERS[index]
            previous = index
        p = math.exp(sum(log_probs[t, c] for t, c in enumerate(path)))
        totals[reading] = totals.get(reading, 0) + p
    return max(totals, key=totals.get)


def test_matches_greedy_when_sure():
    logits = [A, GAP, B, B, GAP, C]
    assert read(logits) == greedy(logits) == "abc"


def test_doubled_letter_needs_a_blank():
    assert read([A, A]) == "a"
    assert read([A, GAP, A]) == "aa"


@pytest.mark.parametrize("seed", range(20))
def test_finds_the_likeliest_reading(seed):
    logits = np.random.default_rng(seed).normal(0, 1.5, size=(4, 4))
    log_probs = log_softmax(logits)
    assert (beam_search(log_probs, BLANK, LETTERS, width=64, prune=1e-9)
            == likeliest(log_probs))


def test_sums_alignments_unlike_greedy():
    # The single best path is two blanks (P 0.16), reading nothing, but
    # "a" has three paths (a a, a -, - a) worth 0.40 in all
    logits = np.log([[0.35, 0.25, 1e-9, 0.4]] * 2)
    assert greedy(logits) == ""
    assert read(logits) == likeliest(log_softmax(logits)) == "a"


@pytest.fixture
def lm():
    return CharLM(["ab"] * 5 + ["ac"], ["a", "b", "c"], order=3)


def test_lm_breaks_a_near_tie(lm):
    logits = [A, GAP, [0, 4.0, 4.1, 0]]
    assert read(logits) == "ac"
    assert read(logits, lm=lm, weight=1.0) == "ab"


def test_lm_leaves_a_clear_image_alone(lm):
    assert read([A, GAP, C], lm=lm, weight=1.0) == "ac"


def test_bonus_keeps_letters(lm):
    # Unsure whether there's a second letter: the bonus tips it to yes
    logits = [A, GAP, [0, 4.0, 0, 4.2]]
    assert read(logits) == "a"
    assert read(logits, bonus=1.0) == "ab"


# --- align and Decoder, over a small alphabet with a space and a mark ------


SMALL = ["a", "b", " ", ".", None]
SMALL_BLANK = 4


def one_hot(*letters, sure=9.0):
    # One token per letter (None for the blank)
    rows = []
    for letter in letters:
        row = [0.0] * len(SMALL)
        row[SMALL.index(letter)] = sure
        rows.append(row)
    return log_softmax(np.array(rows))


def test_align_finds_each_letters_tokens():
    log_probs = one_hot("a", "a", None, "b", None, None, "a")
    assert align(log_probs, "aba", SMALL_BLANK, SMALL) == [(0, 1), (3, 3),
                                                           (6, 6)]


def test_align_doubled_letter_uses_its_blank():
    log_probs = one_hot("a", None, "a")
    assert align(log_probs, "aa", SMALL_BLANK, SMALL) == [(0, 0), (2, 2)]


def test_align_empty():
    assert align(one_hot(None, None), "", SMALL_BLANK, SMALL) == []


def decoder(weight=0.0):
    lm = CharLM(["ab", "ba", "a b"], ["a", "b", " ", "."], order=3)
    return Decoder(lm, weight, 0.0, SMALL, marks=".")


def test_decoder_keeps_a_mark_at_a_words_end():
    output = one_hot("a", None, "b", None, ".", None, " ", None, "a")
    text, spaces = decoder(0.5).read(output, len(output), SMALL_BLANK)
    assert text == "ab. a"
    assert spaces == [6]


def test_decoder_keeps_a_mark_at_a_words_start():
    output = one_hot(".", None, "a", None, "b")
    assert decoder(0.5).read(output, len(output), SMALL_BLANK)[0] == ".ab"


def test_decoder_drops_a_mark_inside_a_word():
    output = one_hot("a", None, ".", None, "b")
    assert decoder(0.5).read(output, len(output), SMALL_BLANK)[0] == "ab"


def test_decoder_without_marks_matches_the_beam_search():
    output = one_hot("a", None, "b", "b", None, "a")
    assert (decoder().read(output, len(output), SMALL_BLANK)[0]
            == beam_search(output, SMALL_BLANK, SMALL) == "aba")
