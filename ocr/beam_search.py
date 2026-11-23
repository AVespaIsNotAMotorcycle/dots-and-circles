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
