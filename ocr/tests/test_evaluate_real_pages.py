import pytest

pytest.importorskip("matplotlib")  # ocr.train imports it

from ocr.evaluate_real_pages import clean, match_lines, score


def test_clean_drops_punctuation():
    assert clean("genefi , jugvn asuru .") == "genefi jugvn asuru"
    assert clean("fonjime [1] , tiyan de [2]") == "fonjime tiyan de"

def test_match_lines_by_closeness():
    found = ["boo horonggo", "junk □ □", "genefi jugvn"]
    pairs, extra = match_lines(found, ["genefi jugvn", "boo horonggo"])
    assert pairs == [("genefi jugvn", "genefi jugvn"),
                     ("boo horonggo", "boo horonggo")]
    assert extra == ["junk □ □"]

def test_missing_lines_count_as_errors():
    pairs, extra = match_lines([], ["abka"])
    assert pairs == [("abka", "")] and extra == []
    assert score(pairs) == (4, 4, 1, 1)

def test_score_counts_word_errors():
    assert score([("hoton de", "hotonde")]) == (8, 1, 2, 2)
