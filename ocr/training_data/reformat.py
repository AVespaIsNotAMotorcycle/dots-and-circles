'''
Rebuilds lexicon.json and alphabet.json from scratch from the Wiktionary
dump (kaikki.org-dictionary-Manchu.jsonl). Run from this folder, then
add_norman.py. Other scripts get the dump's words from kaikki_words.
'''

import json
import os
import unicodedata

directory = os.path.dirname(os.path.abspath(__file__))
filename = "kaikki.org-dictionary-Manchu.jsonl"

def kaikki_words(path=os.path.join(directory, filename)):
    '''
    The dump's words and the letters in them, as two sets: the word of
    every entry with inflected forms, unless it has a Latin letter.
    '''
    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    lexicon = set()
    alphabet = set()
    for line in lines:
        line = json.loads(line)
        if "forms" in line.keys():
            word = line["word"]
            has_latin = False
            for letter in word:
                if "LATIN" in unicodedata.name(letter):
                    has_latin = True
                    break
                alphabet.add(letter)
            if not has_latin:
                lexicon.add(word)
    return lexicon, alphabet

if __name__ == "__main__":
    lexicon, alphabet = kaikki_words(filename)

    print(f"{len(lexicon)} unique words in lexicon.")
    print(f"{len(alphabet)} unique letters in alphabet.")

    # Punctuation isn't in the dictionary's words, but the models write it.
    # It goes last, after the sorted letters, so adding it left every
    # letter's index (and so older models' outputs) as it was.
    PUNCTUATION = ["᠈", "᠉"]   # Manchu comma and full stop
    alphabet = sorted(alphabet - set(PUNCTUATION)) + PUNCTUATION
    with open("alphabet.json", "w", encoding="utf-8") as file:
        json.dump(alphabet, file, ensure_ascii=False, indent=4)
        file.write('\n')

    lexicon = sorted(list(lexicon))
    with open("lexicon.json", "w", encoding="utf-8") as file:
        json.dump(lexicon, file, ensure_ascii=False, indent=4)
        file.write('\n')
