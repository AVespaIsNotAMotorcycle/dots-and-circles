'''
Adds the headwords of Jerry Norman's Comprehensive Manchu-English Dictionary
(Jerry_Norman_Dict.txt, OCR'd text in Norman transliteration) to
lexicon.json, in Manchu script. Run from this folder after reformat.py:
reformat.py rewrites lexicon.json from scratch, which drops these words.

Headwords are the runs of all-caps words in the text. The OCR is imperfect,
so a run is only kept if every word in it converts to Manchu letters and
looks like a Manchu word; "A/B" gives both A and B. A word already in the
lexicon (compared in Abkai, so ignoring variation selectors) isn't added,
and neither is one too tall for a training image (see too_tall).
'''

import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))
from ocr.photos import manchu_to_abkai
from ocr.training_data import HEIGHT, WIDTH, fonts, render

VOWELS = {"a": "ᠠ", "e": "ᡝ", "i": "ᡳ", "o": "ᠣ", "u": "ᡠ", "ū": "ᡡ"}
CONSONANTS = {
    "k'": "ᠺ", "g'": "ᡬ", "h'": "ᡭ", "c'": "ᡱ", "j'": "ᡷ",
    "ts": "ᡮ", "dz": "ᡯ", "ng": "ᠩ",
    "n": "ᠨ", "k": "ᡴ", "g": "ᡤ", "h": "ᡥ", "b": "ᠪ", "p": "ᡦ", "s": "ᠰ",
    "š": "ᡧ", "t": "ᡨ", "d": "ᡩ", "l": "ᠯ", "m": "ᠮ", "c": "ᠴ", "j": "ᠵ",
    "y": "ᠶ", "r": "ᡵ", "f": "ᡶ", "w": "ᠸ", "ž": "ᡰ",
    # The OCR sometimes loses the caron, as in ZIN for žin
    "z": "ᡰ",
}
# Syllables with the apical vowel, spelled as in the table at
# https://en.wikipedia.org/wiki/Transliterations_of_Manchu#Romanization.
# Norman writes them with y, or with no vowel at all (ts, dz). They only
# count as these when no vowel follows: tsa is ᡮ + a.
APICAL = {"tsy": "ᡮᡟ", "dzy": "ᡯᡳ᠌", "sy": "ᠰᡟ", "cy": "ᡱᡳ", "jy": "ᡷᡳ",
          "ts": "ᡮᡟ", "dz": "ᡯᡳ᠌"}
# Each table's spellings longest first, so that ng beats n and k' beats k
LETTERS = {**CONSONANTS, **VOWELS}
APICAL_SPELLINGS = sorted(APICAL, key=len, reverse=True)
SPELLINGS = sorted(LETTERS, key=len, reverse=True)
MANCHU_VOWELS = set("ᠠᡝᡳᠣᡠᡡᡟ")

def to_manchu(word):
    '''
    One lower-case Norman word in Manchu script, or None if it doesn't
    split into Manchu letters.
    '''
    letters = []
    i = 0
    while i < len(word):
        apical = [spelling for spelling in APICAL_SPELLINGS
                  if word.startswith(spelling, i)
                  and word[i + len(spelling):i + len(spelling) + 1]
                  not in VOWELS]
        plain = [spelling for spelling in SPELLINGS
                 if word.startswith(spelling, i)]
        if apical:
            letters.append(APICAL[apical[0]])
            i += len(apical[0])
        elif plain:
            letters.append(LETTERS[plain[0]])
            i += len(plain[0])
        else:
            return None
    return "".join(letters)

def looks_manchu(word):
    '''
    Whether a Norman word converts and is shaped like a Manchu word: it has
    a vowel, starts and ends with at most one consonant (ng + g aside) and
    never has three in a row. This catches most OCR errors, like ŠBTEMBI.
    '''
    manchu = to_manchu(word)
    if manchu is None or not MANCHU_VOWELS & set(manchu):
        return False
    consonants = re.split("[" + "".join(MANCHU_VOWELS) + "]", manchu)
    ending = consonants[-1].replace("ᠩᡤ", "ᠩ")
    return (len(consonants[0]) <= 1 and len(ending) <= 1
            and all(len(run) <= 2 for run in consonants))

def too_tall(word):
    '''
    Whether word, rendered and scaled to WIDTH, is taller than HEIGHT in
    the median font. Training images that tall get squashed to fit, far
    denser than any photographed word. Fonts run up to about 24 px a letter
    at WIDTH, so shorter words can't be too tall and aren't rendered.
    '''
    if len(word) * 24 <= HEIGHT:
        return False
    images = [render(word, font) for font in fonts]
    heights = sorted(image.height * WIDTH / image.width for image in images)
    return heights[len(heights) // 2] > HEIGHT

def headwords(text):
    '''
    Every headword in the dictionary's text, in lower-case Norman.
    '''
    caps = re.compile(r"^[A-ZŠŪŽ'/]+$")
    runs = []
    for line in text.split("\n"):
        run = []
        for token in line.split() + [""]:
            if caps.match(token):
                run.append(token.lower())
            elif run:
                runs.append(run)
                run = []

    found = []
    for run in runs:
        good = [all(looks_manchu(word) for word in token.split("/"))
                for token in run]
        # Page references like BH 629B sit next to headwords, so drop bad
        # words at the ends but skip runs with one in the middle
        while run and not good[0]:
            run, good = run[1:], good[1:]
        while run and not good[-1]:
            run, good = run[:-1], good[:-1]
        if not run or not all(good):
            continue
        phrases = [[]]
        for token in run:
            phrases = [phrase + [word] for phrase in phrases
                       for word in token.split("/")]
        found.extend(" ".join(phrase) for phrase in phrases)
    return found

with open("Jerry_Norman_Dict.txt", "r", encoding="utf-8") as file:
    norman = headwords(file.read())
with open("lexicon.json", "r", encoding="utf-8") as file:
    lexicon = json.load(file)

known = {manchu_to_abkai(word) for word in lexicon}
added = tall = 0
for phrase in norman:
    manchu = " ".join(to_manchu(word) for word in phrase.split())
    abkai = manchu_to_abkai(manchu)
    if abkai in known:
        continue
    if too_tall(manchu):
        tall += 1
        continue
    known.add(abkai)
    lexicon.append(manchu)
    added += 1

print(f"{len(norman)} headwords in the dictionary, {added} of them new. "
      f"{tall} more were left out as too tall.")
print(f"{len(lexicon)} words in lexicon.")

lexicon = sorted(lexicon)
with open("lexicon.json", "w", encoding="utf-8") as file:
    json.dump(lexicon, file, ensure_ascii=False, indent=4)
    file.write('\n')
