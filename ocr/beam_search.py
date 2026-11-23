'''
Reads a CTC model's output with a beam search, optionally weighing in a
character language model (ocr/char_lm.py) as a tiebreaker.

ctc_decode in train.py takes the likeliest class at each token. This
instead keeps the WIDTH likeliest readings so far, token by token, so a
token that is nearly a tie (a or e) can be settled by what follows it and
by how likely the whole spelling is. Each reading is scored

    log P(reading | image) + weight * log P_LM(reading) + bonus * letters

- P(reading | image) sums over every way CTC can spell it (where the
  blanks and repeated tokens fall), not just the single best path.
- weight is how much the language model counts; 0 reads the image alone.
- bonus is added per letter: a language model likes short readings (each
  letter multiplies in a probability below 1), and without it the search
  drops letters, the model's commonest mistake already.

The prefix beam search keeps two probabilities per reading, of the image
so far ending in a blank or in its last letter: CTC spells a doubled
letter only with a blank between its two halves, and merges a letter
repeated without one (Hannun et al., "First-Pass Large Vocabulary
Continuous Speech Recognition using Bidirectional Recurrent DNNs", 2014).
'''
import math
from collections import defaultdict

import numpy as np

from ocr.char_lm import END

WIDTH = 16
# A token's classes below this probability aren't tried, which keeps the
# search fast: most tokens are nearly certain blanks
PRUNE = 1e-3
NEGATIVE_INFINITY = float("-inf")


def log_softmax(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def add_logs(*values):
    # log(sum(exp(values))), safely
    top = max(values)
    if top == NEGATIVE_INFINITY:
        return top
    return top + math.log(sum(math.exp(value - top) for value in values))


def beam_search(log_probs, blank, alphabet, lm=None, weight=0.0, bonus=0.0,
                width=WIDTH, prune=PRUNE):
    '''
    The best reading of one image. log_probs is the model's log softmax
    output for the image's real tokens, (tokens, classes); alphabet maps a
    class to its letter. Returns the reading as a string.
    '''
    use_lm = lm is not None and weight != 0
    # Each reading's (log P ending in blank, log P ending in its last letter)
    beams = {"": (0.0, NEGATIVE_INFINITY)}
    # Each reading's language model score plus its letter bonuses, kept as
    # readings grow so each letter's is looked up once
    extra = {"": 0.0}
    threshold = math.log(prune)

    def letter_score(reading, letter):
        score = bonus
        if use_lm:
            score += weight * lm.log_prob(reading, letter)
        return score

    for token in log_probs:
        tried = [index for index in np.flatnonzero(token >= threshold)
                 if index != blank]
        if not tried:
            tried = [int(token.argmax())] if token.argmax() != blank else []
        token_blank = float(token[blank])
        grown = defaultdict(lambda: [NEGATIVE_INFINITY, NEGATIVE_INFINITY])
        for reading, (ends_blank, ends_letter) in beams.items():
            # A blank: the reading stays as it is
            entry = grown[reading]
            entry[0] = add_logs(entry[0], ends_blank + token_blank,
                                ends_letter + token_blank)
            for index in tried:
                letter = alphabet[index]
                p = float(token[index])
                longer = reading + letter
                if reading and letter == reading[-1]:
                    # The same letter again, no blank between: merged into
                    # the reading as it is. Only after a blank does it
                    # start a second, doubled letter.
                    entry = grown[reading]
                    entry[1] = add_logs(entry[1], ends_letter + p)
                    entry = grown[longer]
                    entry[1] = add_logs(entry[1], ends_blank + p)
                else:
                    entry = grown[longer]
                    entry[1] = add_logs(entry[1], ends_blank + p,
                                        ends_letter + p)
                if longer not in extra:
                    extra[longer] = extra[reading] + letter_score(reading,
                                                                  letter)
        ranked = sorted(grown.items(), key=lambda item: add_logs(*item[1])
                        + extra[item[0]], reverse=True)
        beams = {reading: tuple(probs) for reading, probs in ranked[:width]}

    def final(reading):
        score = add_logs(*beams[reading]) + extra[reading]
        if use_lm:
            score += weight * lm.log_prob(reading, END)
        return score

    return max(beams, key=final)


def align(log_probs, text, blank, alphabet):
    '''
    The likeliest CTC path through log_probs that spells text (Viterbi):
    for each letter of text, its (first, last) token. text must be
    spellable in that many tokens.
    '''
    if not text:
        return []
    # The path's states: a blank before, between and after the letters
    states = [blank]
    for letter in text:
        states += [alphabet.index(letter), blank]
    states = np.array(states)
    count = len(states)
    emit = log_probs[:, states]
    # A letter state may skip the blank before it, unless the letter
    # before is the same (a doubled letter needs its blank)
    can_skip = np.zeros(count, dtype=bool)
    can_skip[2:] = (states[2:] != blank) & (states[2:] != states[:-2])
    score = np.full(count, -np.inf)
    score[:2] = emit[0, :2]
    back = np.zeros((len(log_probs), count), dtype=int)
    for t in range(1, len(log_probs)):
        stay = score
        step = np.concatenate([[-np.inf], score[:-1]])
        skip = np.where(can_skip, np.concatenate([[-np.inf] * 2,
                                                  score[:-2]]), -np.inf)
        options = np.stack([stay, step, skip])
        best = options.argmax(axis=0)
        back[t] = np.arange(count) - best
        score = options.max(axis=0) + emit[t]
    state = count - 1 if score[-1] >= score[-2] else count - 2
    spans = {}
    for t in range(len(log_probs) - 1, -1, -1):
        if state % 2:
            first, last = spans.get(state // 2, (t, t))
            spans[state // 2] = (t, last)
        state = back[t, state]
    return [spans[k] for k in range(len(text))]


class Decoder:
    '''
    Reads model outputs with the beam search and language model, keeping
    the model's own punctuation where it marks a word's start or end:

    1. The letters are read with the marks taken out of the model's
       output, since the language model, trained on lexicon words, has
       seen no punctuation and would drop every mark.
    2. Each mark the greedy reading has at a word's start or end goes back
       in, after the letters the model saw above it. Marks inside a word
       are dropped.

    On the real pages this read as many marks as greedy decoding and the
    letters as well as the language model alone.
    '''
    def __init__(self, lm, weight, bonus, alphabet, marks="᠈᠉",
                 width=WIDTH):
        self.lm, self.weight, self.bonus = lm, weight, bonus
        self.alphabet, self.marks, self.width = alphabet, marks, width

    def read(self, output, length, blank):
        '''
        The reading of one image's output (its first length tokens), and
        for each space in it, the token in its middle, as
        page_reader.decode_words gives them.
        '''
        log_probs = log_softmax(np.asarray(output[:length], dtype=float))
        mark_classes = [self.alphabet.index(mark) for mark in self.marks
                        if self.alphabet.index(mark) < log_probs.shape[1]]
        kept = self.edge_marks(log_probs, blank)
        letters_only = log_probs.copy()
        letters_only[:, mark_classes] = -np.inf
        letters_only = log_softmax(letters_only)
        text = beam_search(letters_only, blank, self.alphabet, self.lm,
                           self.weight, self.bonus, self.width)
        starts = [first for first, _ in align(letters_only, text, blank,
                                               self.alphabet)]
        # From the last mark back, so earlier places don't move
        for place, mark in sorted(((sum(s < token for s in starts), mark)
                                   for token, mark in kept), reverse=True):
            text = text[:place] + mark + text[place:]
        spaces = [(first + last) / 2 for letter, (first, last) in
                  zip(text, align(log_probs, text, blank, self.alphabet))
                  if letter == " "]
        return text, spaces

    def edge_marks(self, log_probs, blank):
        # (middle token, mark) for each mark the greedy reading has at a
        # word's start or end
        classes = log_probs.argmax(axis=-1)
        greedy, middles, start = "", [], 0
        for token in range(1, len(classes) + 1):
            if token < len(classes) and classes[token] == classes[start]:
                continue
            if classes[start] != blank:
                greedy += self.alphabet[classes[start]]
                middles.append((start + token - 1) / 2)
            start = token
        kept = []
        for i, letter in enumerate(greedy):
            if letter not in self.marks:
                continue
            before = greedy[:i].rstrip(self.marks)
            after = greedy[i + 1:].lstrip(self.marks)
            if (not before or before.endswith(" ") or not after
                    or after.startswith(" ")):
                kept.append((middles[i], letter))
        return kept


# The language model, built once however many decoders use it
_lm = None


def load_decoder(settings):
    '''
    A Decoder from a server/config.json entry's "language_model" settings,
    {"weight": ..., "bonus": ...}.
    '''
    global _lm
    from ocr.char_lm import load_lm
    from ocr.train import alphabet
    if _lm is None:
        _lm = load_lm()
    return Decoder(_lm, settings["weight"], settings.get("bonus", 0),
                   alphabet)


def read_words(model, y, entries, lm=None, weight=0.0, bonus=0.0,
               width=WIDTH):
    '''
    Decodes the model's outputs y for entries (as train.read_words does)
    with the beam search.
    '''
    from ocr.train import alphabet
    return [beam_search(log_softmax(output[:length]), model.blank, alphabet,
                        lm, weight, bonus, width)
            for output, (_, _, length) in zip(y, entries)]
