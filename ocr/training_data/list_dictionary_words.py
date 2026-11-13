'''
Writes dictionary_words.json: every word and phrase of the Wiktionary dump
(as reformat.py reads it) and of Norman's dictionary (as add_norman.py
reads it), each with the dictionaries it's in, for the server's closest
words (ocr.dictionary). Run it from the repository root after either
dictionary changes:

  python -m ocr.training_data.list_dictionary_words

Entries are matched across the two dictionaries in the photo dataset's
romanization (ocr.photos.manchu_to_roman), so spellings differing only in
variation selectors are one entry, written as Wiktionary spells it. Unlike
lexicon.json, it keeps headwords too tall for a training image, and leaves
out affixes (entries with ᠊, such as ᠊ᠨ), which aren't words on their own.
Verbs Norman notes irregular forms of (DOSIMBI (-ka), see
add_norman.verb_notes) have them as a third item, for ocr.dictionary.
'''

import json

from ocr.dictionary import PATH
from ocr.photos import manchu_to_roman
from ocr.training_data.add_norman import norman_words, to_manchu, verb_notes
from ocr.training_data.reformat import kaikki_words

# The hyphen-like mark Manchu affixes are written with
AFFIX_MARK = "\u180a"

def dictionary_words():
    '''
    [manchu, sources] for each entry, sorted by romanization, sources being
    a list of "kaikki" and "norman". A verb with a note in Norman has it
    as a third item, {"perfective": its perfective endings (Norman, as
    ["-ka"]), "forms": [[manchu, affixes]]}, its forms in Manchu script.
    '''
    kaikki, _ = kaikki_words()
    entries = {}
    for source, words in (("kaikki", sorted(kaikki)),
                          ("norman", norman_words())):
        for word in words:
            if AFFIX_MARK in word:
                continue
            entry = entries.setdefault(manchu_to_roman(word), [word, []])
            if source not in entry[1]:
                entry[1].append(source)
    for verb, note in verb_notes().items():
        entry = entries.get(manchu_to_roman(to_manchu(verb)))
        if entry is not None:
            entry.append({"perfective": note["perfective"], "forms": [
                [to_manchu(form), affixes] for form, affixes in note["forms"]]})
    return [entries[roman] for roman in sorted(entries)]

def write(entries, path=PATH):
    # One entry a line, so changes diff well
    with open(path, "w", encoding="utf-8") as file:
        file.write("[\n")
        file.write(",\n".join(json.dumps(entry, ensure_ascii=False)
                              for entry in entries))
        file.write("\n]\n")

if __name__ == "__main__":
    entries = dictionary_words()
    write(entries)
    count = lambda source: sum(source in entry[1] for entry in entries)
    print(f"{len(entries)} entries: {count('kaikki')} in Wiktionary, "
          f"{count('norman')} in Norman, "
          f"{sum(len(entry[1]) == 2 for entry in entries)} in both, "
          f"{sum(len(entry) == 3 for entry in entries)} with verb notes.")
