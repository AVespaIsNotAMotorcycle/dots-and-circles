import pytest

from ocr.training_data import GENITIVE_WORD, fonts, lexicon
from ocr.training_data.compose import (COMMA, FULL_STOP, HOOK, ISOLATED,
                                       Composite, fix_mark_order,
                                       generate_composite, label,
                                       make_composites, single_words)


def index_of(word):
    return lexicon.index(word)

def test_label_spaces_words_and_attaches_marks():
    be, niyalma = index_of("ᠪᡝ"), index_of("ᠨᡳᠶᠠᠯᠮᠠ")
    assert label([be, COMMA, niyalma, FULL_STOP]) == "ᠪᡝ᠈ ᠨᡳᠶᠠᠯᠮᠠ᠉"
    assert label([niyalma, HOOK]) == "ᠨᡳᠶᠠᠯᠮᠠ ᡳ"
    assert label([ISOLATED]) == "ᡳ"

def test_marks_follow_the_genitive_and_never_start():
    assert fix_mark_order([1, COMMA, HOOK, 2]) == [1, HOOK, COMMA, 2]
    assert fix_mark_order([FULL_STOP, 1]) == [1]

def test_composites_keep_to_their_words_and_repeat():
    indices = list(range(0, len(lexicon), 7))
    first = make_composites(indices, 2000, seed=5)
    assert first == make_composites(indices, 2000, seed=5)
    allowed = set(single_words(indices))
    words = {part for composite in first for part in composite.parts
             if isinstance(part, int)}
    assert words and words <= allowed
    # 15% phrases, 3% lone i, 10% punctuated words
    assert len(first) == 300 + 60 + 200
    assert len({composite.number for composite in first}) == len(first)

def test_genitive_drawn_as_a_suffix_only_as_its_own_word():
    assert GENITIVE_WORD.sub("+", "ᡥᠣᡨᠣᠨ ᡳ ᡩᡝ") == "ᡥᠣᡨᠣᠨ+ᡳ ᡩᡝ"
    # ᡳᠨᡠ starts with i but isn't the genitive
    assert GENITIVE_WORD.sub("+", "ᠨᡳᠶᠠᠯᠮᠠ ᡳᠨᡠ") == "ᠨᡳᠶᠠᠯᠮᠠ ᡳᠨᡠ"
    assert GENITIVE_WORD.sub("+", "ᠨᡳᠶᠠᠯᠮᠠ ᡳ") == "ᠨᡳᠶᠠᠯᠮᠠ+ᡳ"

@pytest.mark.skipif(not fonts, reason="the fonts aren't here")
def test_composite_renders_to_the_models_width():
    be, niyalma = index_of("ᠪᡝ"), index_of("ᠨᡳᠶᠠᠯᠮᠠ")
    for spacing in ("tight", "close", "normal"):
        composite = Composite((be, COMMA, niyalma, HOOK), 0, spacing, 3)
        image, text = generate_composite(composite)
        assert image.width == 32 and text == "ᠪᡝ᠈ ᠨᡳᠶᠠᠯᠮᠠ ᡳ"
        augmented, same = generate_composite(composite, epoch=1)
        assert augmented.width == 32 and same == text
