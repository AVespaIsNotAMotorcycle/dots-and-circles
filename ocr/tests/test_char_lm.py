import pytest

from ocr.char_lm import END, CharLM

WORDS = ["abka", "abkai", "baba", "bahambi", "ehe", "erebe", "gemu",
         "gebu", "hala", "halambi", "nadan", "dade"]
SYMBOLS = sorted(set("".join(WORDS))) + ["x"]


@pytest.fixture(scope="module")
def lm():
    return CharLM(WORDS, SYMBOLS, order=4)


@pytest.mark.parametrize("context", ["", "a", "ab", "bah", "xyz", "ere"])
def test_sums_to_one(lm, context):
    total = sum(lm.prob(context, symbol) for symbol in lm.symbols)
    assert total == pytest.approx(1)


def test_never_zero(lm):
    assert lm.prob("xxx", "x") > 0
    assert lm.prob("", "x") > 0


def test_seen_spellings_beat_unseen(lm):
    assert lm.word_log_prob("baba") > lm.word_log_prob("babe")
    assert lm.word_log_prob("erebe") > lm.word_log_prob("ereba")


def test_ends_words(lm):
    # Training words end after "bi", never after "bah"
    assert lm.prob("bahambi", END) > lm.prob("bah", END)


def test_skips_words_with_other_characters():
    assert CharLM(["abc", "a?c"], ["a", "b", "c"], order=3).words == 1
