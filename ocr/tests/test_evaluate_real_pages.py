import pytest

pytest.importorskip("matplotlib")  # ocr.train imports it

from ocr.evaluate_real_pages import (clean, match_lines, punctuation_counts,
                                     score)


def test_clean_drops_punctuation():
    assert clean("genefi , jugvn asuru .") == "genefi jugvn asuru"
    assert clean("fonjime [1] , tiyan de [2]") == "fonjime tiyan de"
    assert clean("han i Araha Mujilen") == "han i araha mujilen"

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

def test_punctuation_counted_by_kind_anywhere_in_line():
    assert punctuation_counts("be, tere. ombi,", "be tere, ombi") == (3, 1, 1)
    # The Manchu marks count as their Latin ones
    assert punctuation_counts("be, tere.", "be\u1808 tere\u1809") == (2, 2, 2)
    assert punctuation_counts("abka", "abka.") == (0, 1, 0)
