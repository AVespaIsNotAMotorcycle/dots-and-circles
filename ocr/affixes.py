'''
Verb suffixes and derivational infixes, for matching inflected readings
to their verbs' dictionary entries (ocr.dictionary). A verb's entry is its
stem with -mbi (arambi); its other forms put up to MAX_INFIXES infixes
then one suffix on the stem (ara-bu-ha). Vowel harmony isn't enforced:
any of -ha, -he and -ho may follow any stem.

From Mark Elliott's Manchu Studies Group handouts, CC BY-NC-SA 4.0:
https://www.manchustudiesgroup.org/wp-content/uploads/2020/07/Handout-3-Verbs.pdf
https://www.manchustudiesgroup.org/wp-content/uploads/2020/07/Handout-4-Derivational-suffixes.pdf

They're in Norman's romanization, as in the handouts: NORMAN_TO_ROMAN
gives the photo dataset's (ū is v, š is x).
'''

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
