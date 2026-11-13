from ocr.affixes import VARIANTS, ending_breaks, endings, harmony


def test_harmony_classes():
    # Porter's rules: a, e or o alone take their own vowel...
    assert harmony("ara") == "a" and harmony("gene") == "e"
    assert harmony("o") == "o" and harmony("bodo") == "o"
    # ...e, i and u take e...
    assert harmony("ji") == harmony("tuci") == harmony("gisure") == "e"
    # ...and i or u with a, o or ū take a
    assert harmony("donji") == harmony("jabu") == harmony("hvla") == "a"
    assert harmony("dosi") == "a"

def test_variants_are_grouped_by_vowel():
    assert VARIANTS["-ha"] == {"a": "-ha", "e": "-he", "o": "-ho"}
    assert VARIANTS["-nje-"] == {"a": "-nja-", "e": "-nje-", "o": "-njo-"}
    # Affixes with one spelling, even with a, e or o in them
    for affix in ("-me", "-cina", "-cibe", "-mbihe", "-bu-", "-nji-"):
        assert affix not in VARIANTS

def harmony_break(affix, expected):
    return [{"affix": affix, "expected": expected, "reason": "harmony"}]

def test_harmony_breaks():
    assert ending_breaks("ara", ["-ha"]) == []
    assert ending_breaks("ara", ["-ho"]) == harmony_break("-ho", ["-ha"])
    assert ending_breaks("o", ["-ho"]) == ending_breaks("dosi", ["-ka"]) \
        == []
    # Stems of e, i and u take e
    assert ending_breaks("ji", ["-he"]) == []
    assert ending_breaks("ji", ["-ha"]) == harmony_break("-ha", ["-he"])
    # Infixes count: an e infix makes -he right after an e stem
    assert ending_breaks("gene", ["-ne-", "-he"]) == []
    assert ending_breaks("o", ["-bu-", "-ho"]) == harmony_break(
        "-ho", ["-ha"])
    # A wrong infix is flagged once, and what follows is judged as if it
    # had been right
    assert ending_breaks("gene", ["-na-", "-he"]) == harmony_break(
        "-na-", ["-ne-"])
    # Invariant affixes never break it
    assert ending_breaks("ara", ["-me"]) == []

def test_re_follows_i_and_u():
    # -re after i, u or ū, whatever the stem's vowels; -ha as usual
    assert ending_breaks("ali", ["-re"]) == ending_breaks(
        "aca", ["-bu-", "-re"]) == ending_breaks("gashv", ["-re"]) == []
    assert ending_breaks("aca", ["-bu-", "-ha"]) == []
    assert ending_breaks("ali", ["-ra"]) == harmony_break("-ra", ["-re"])

def test_norman_notes():
    # ILIMBI (-ha): iliha, not the e Porter's rules give, and on -habi too
    assert ending_breaks("ili", ["-ha"], ["-ha"]) == []
    assert ending_breaks("ili", ["-he"], ["-ha"]) == [
        {"affix": "-he", "expected": ["-ha"], "reason": "norman"}]
    assert ending_breaks("ili", ["-hebi"], ["-ha"]) == [
        {"affix": "-hebi", "expected": ["-habi"], "reason": "norman"}]
    # DOSIMBI (-ka)
    assert ending_breaks("dosi", ["-ka"], ["-ka"]) == []
    assert ending_breaks("dosi", ["-ha"], ["-ka"]) == [
        {"affix": "-ha", "expected": ["-ka"], "reason": "norman"}]
    # A note giving only a perfective that isn't a suffix here (JEMBI (2)
    # (-ngke))
    assert ending_breaks("je", ["-ke"], ["-ngke"]) == [
        {"affix": "-ke", "expected": ["-ngke"], "reason": "norman"}]
    # A note with no perfective (OMBI's gives ojoro and oso) leaves it to
    # the rules
    assert ending_breaks("o", ["-ho"], []) == []

def test_k_only_where_norman_notes_it():
    unmarked = [{"affix": "-ka", "expected": ["-ha"], "reason": "unmarked"}]
    # A verb in Norman with no note takes -ha...
    assert ending_breaks("ara", ["-ka"], []) == unmarked
    # ...but one only in Wiktionary isn't checked
    assert ending_breaks("ara", ["-ka"]) == []
    # The vowel is still the rules': araka, of an e stem
    assert ending_breaks("gene", ["-ka"], []) == [
        {"affix": "-ka", "expected": ["-he"], "reason": "unmarked"}]

def test_noted_vowels_carry_through_invariant_infixes():
    # ili-bu-ha, of ILIMBI (-ha), and of ilibumbi, derived from it
    assert ending_breaks("ili", ["-bu-", "-ha"], ["-ha"]) == []
    assert ending_breaks("ilibu", ["-ha"], [], ["-ha"]) == []
    # Only the vowel: the k of a note isn't inherited
    assert ending_breaks("isibu", ["-ka"], [], ["-ka"]) == [
        {"affix": "-ka", "expected": ["-ha"], "reason": "unmarked"}]
    # An infix with variants chooses the vowel afresh: gene-ne-he, even
    # if the verb's own perfective were -ka
    assert ending_breaks("gene", ["-ne-", "-he"], ["-ka"]) == []

def test_endings():
    found = dict(endings())
    assert found["ha"] == ["-ha"] and found["buha"] == ["-bu-", "-ha"]
    assert found["rahv"] == ["-rahū"]
    assert "mbi" not in found and found["bumbi"] == ["-bu-", "-mbi"]
