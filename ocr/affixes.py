'''
Verb suffixes and derivational infixes, for matching inflected readings
to their verbs' dictionary entries (ocr.dictionary). A verb's entry is its
stem with -mbi (arambi); its other forms put up to MAX_INFIXES infixes
then one suffix on the stem (ara-bu-ha). Vowel harmony isn't enforced:
any of -ha, -he and -ho may follow any stem, but ending_breaks says which
affixes break it, or the forms Norman's dictionary gives.

From Mark Elliott's Manchu Studies Group handouts, CC BY-NC-SA 4.0:
https://www.manchustudiesgroup.org/wp-content/uploads/2020/07/Handout-3-Verbs.pdf
https://www.manchustudiesgroup.org/wp-content/uploads/2020/07/Handout-4-Derivational-suffixes.pdf

They're in Norman's romanization, as in the handouts: norman_to_roman
gives the photo dataset's (ū is v, š is x).
'''

import re
from itertools import product

SUFFIXES = [
    "-mbi", "-ra", "-re", "-ro", "-habi", "-hebi", "-hobi", "-ha", "-he",
    "-ho", "-ka", "-ke", "-ko", "-mbihe", "-ki", "-kini", "-cina", "-rahū",
    "-me", "-fi", "-pi", "-ci", "-tala", "-tele", "-tolo", "-nggala",
    "-nggele", "-nggolo", "-cibe", "-hai", "-hei", "-hoi", "-ralame",
    "-relame", "-rolame", "-tai", "-tei", "-toi",
]

INFIXES = [
    "-bu-", "-mbu-", "-ndu-", "-ca-", "-ce-", "-co-", "-nji-", "-na-",
    "-ne-", "-no-", "-ta-", "-te-", "-to-", "-nja-", "-nje-", "-njo-",
    "-nu-", "-nggi-",
]

# Infixes allowed between a stem and its suffix
MAX_INFIXES = 2

# The verb ending of dictionary entries, in the dataset's romanization
LEMMA_ENDING = "mbi"


def norman_to_roman(affix):
    # An affix's letters in the photo dataset's romanization
    return affix.strip("-").replace("ū", "v").replace("š", "x")


def endings():
    '''
    (roman, affixes) for every ending a verb stem can take, roman being
    its letters in the dataset's romanization and affixes the Norman ones
    making it up, fewest infixes first. -mbi alone is left out, as that's
    the dictionary entry itself. A spelling made up two ways (ne-bu, say,
    if both were affixes) is kept once, with its first, fewest infixes.
    '''
    found = {}
    for count in range(MAX_INFIXES + 1):
        for infixes in product(INFIXES, repeat=count):
            for suffix in SUFFIXES:
                affixes = list(infixes) + [suffix]
                roman = "".join(norman_to_roman(affix) for affix in affixes)
                if roman != LEMMA_ENDING:
                    found.setdefault(roman, affixes)
    return list(found.items())


# Vowels in the dataset's romanization (v is ū)
VOWELS = "aeiouv"


def harmony(letters):
    '''
    The vowel class of letters (in the dataset's romanization), which picks
    the variant of the affix after them, as in David C. Porter's rules for
    the perfective (-ha): "a" if they have a or ū, else "o" if all their
    vowels are o, else "a" if they have o (with i or u: dosi-ka), else
    "e" (with e, i and u only: gene-he, tuci-ke, ji-he).
    '''
    vowels = [letter for letter in letters if letter in VOWELS]
    if any(vowel in "av" for vowel in vowels):
        return "a"
    if "o" in vowels:
        return "o" if all(vowel == "o" for vowel in vowels) else "a"
    return "e"


def variant_groups():
    '''
    Affix -> {vowel: affix} for each affix with a, e and o variants (-ha,
    -he, -ho), which are spelled alike but for those vowels. Affixes with
    one spelling (-me, -cina, -bu-) aren't in it.
    '''
    groups = {}
    for affix in SUFFIXES + INFIXES:
        groups.setdefault(re.sub("[aeo]", "*", affix), []).append(affix)
    variants = {}
    for group in groups.values():
        if len(group) > 1:
            by_vowel = {vowel(affix): affix for affix in group}
            variants.update((affix, by_vowel) for affix in group)
    return variants


def vowel(affix):
    # The a, e or o choosing an affix's variant
    return re.search("[aeo]", affix).group()


VARIANTS = variant_groups()

# Affixes whose e variant follows any i, u or ū (ali-re, aca-bu-re, not
# aca-bu-ra; gashū-re): in all 56 such words of the photo dataset. -ralame
# is built on -ra, but no word there shows it.
E_AFTER_I_U = {"-ra", "-re", "-ro", "-ralame", "-relame", "-rolame"}

# The perfective participle, and the suffixes built on it, which take
# the vowel Norman gives a verb's perfective (ILIMBI (-ha): ili-ha-bi)
PERFECTIVE = {"-ha", "-he", "-ho", "-ka", "-ke", "-ko"}
ON_PERFECTIVE = {"-habi", "-hebi", "-hobi", "-hai", "-hei", "-hoi"}
# The perfective with k, which only verbs noted as taking it in Norman's
# dictionary do
K_PERFECTIVE = {"-ka", "-ke", "-ko"}


def ending_breaks(stem, affixes, perfective=None, inherited=None):
    '''
    The affixes (Norman) on stem (the dataset's romanization) that are
    wrong, each as {"affix": affix, "expected": [affixes it should have
    been], "reason": why}. perfective is None for a verb not in Norman's
    dictionary, else the perfective endings its note there gives (["-ka"],
    or [] for none). inherited are those of the verb it's derived from by
    an infix with one spelling (ilibumbi, of ILIMBI (-ha)), if it has none
    of its own: their vowel carries over (ili-bu-ha), as does a verb's own
    through such infixes. The reasons:

    - "norman": Norman's note gives other perfective endings (ILIMBI
      (-ha), so ilihe is wrong), or another vowel for a suffix built on
      it (ilihebi). Only for affixes straight after the stem: a note is
      about the verb, not its derived verbs.
    - "unmarked": a perfective with k (-ka) on a verb in Norman's
      dictionary not noted as taking one (araka for araha).
    - "harmony": the affix breaks vowel harmony (harmony) with the letters
      before it, infixes included (ara-na-ha), but -re follows i, u or ū
      whatever the vowels (E_AFTER_I_U), and the perfective and suffixes
      on it take the vowel carried over from a note, if any.

    A wrong affix is taken as its first expected one for the affixes after
    it, so one break isn't flagged twice (gene-na-he has -na- for -ne-,
    but -he is right). Empty if none are wrong. Irregular verbs not noted
    in Norman can break harmony anyway (kiri-ha).
    '''
    breaks = []
    before = stem
    carried = {vowel(ending) for ending in perfective or inherited or []}
    for position, affix in enumerate(affixes):
        noted = perfective if position == 0 else None
        expected, reason = None, None
        if noted and affix in PERFECTIVE:
            if affix not in noted:
                expected, reason = noted, "norman"
        elif noted and affix in ON_PERFECTIVE:
            vowels = sorted({vowel(ending) for ending in noted})
            if vowel(affix) not in vowels:
                expected, reason = [VARIANTS[affix][v] for v in vowels], \
                    "norman"
        elif affix in VARIANTS:
            allowed = harmony(before)
            if affix in E_AFTER_I_U and before[-1:] in ("i", "u", "v"):
                allowed = "e"
            if carried and affix in PERFECTIVE | ON_PERFECTIVE:
                allowed = "".join(sorted(carried))
            k_unmarked = perfective is not None and affix in K_PERFECTIVE
            if vowel(affix) not in allowed or k_unmarked:
                expected = [VARIANTS[affix][v] for v in allowed]
                reason = "harmony"
                if k_unmarked:
                    expected = ["-h" + ending[2:] for ending in expected]
                    reason = "unmarked"
        if expected:
            breaks.append({"affix": affix, "expected": expected,
                           "reason": reason})
            affix = expected[0]
        if affix in VARIANTS:
            carried = set()  # The vowel's chosen afresh after -na- (say)
        before += norman_to_roman(affix)
    return breaks
