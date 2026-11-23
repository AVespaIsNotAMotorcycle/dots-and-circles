'''
A character language model of Manchu spelling: how likely each next letter
is, given the letters before it in the word. The beam search decoder
(ocr/beam_search.py) weighs it against what the image shows, as a
tiebreaker where the image is unclear (a or e, o or u, a tooth or none):
it never rules a spelling out, so words in no lexicon can still be read.

It counts which letters follow which in the lexicon's training words
(the same split as train.py, so no benchmark word is counted), and
smooths the counts with interpolated Kneser-Ney (Chen and Goodman, "An
Empirical Study of Smoothing Techniques for Language Modeling", 1998):

- Each count is lowered by a discount D, and the probability taken off
  is handed to the next shorter context. A context seen often keeps
  nearly all its probability; one seen once or twice leans on the
  shorter contexts.
- The shorter contexts count *in how many different longer contexts* a
  letter follows them, not how often. A letter that is common only after
  one particular context shouldn't look common everywhere.
- Below the one-letter-free context sits a uniform share over every
  symbol, so no spelling ever has probability 0.

Contexts are the ORDER - 1 letters before: five, by default, about half
a word, so a suffix's letter is predicted from the stem vowel before it.
'''
import math
from collections import Counter, defaultdict

ORDER = 6
# Pads the start of every word, so its first letters have full contexts
START = "<"
# Ends every word: the model predicts it like a letter
END = ">"


class CharLM:
    def __init__(self, words, symbols, order=ORDER):
        '''
        words are the training text, each a string of symbols; symbols are
        the letters it can predict (END is added). Words with any other
        character are left out.
        '''
        self.order = order
        self.symbols = list(symbols) + [END]
        allowed = set(symbols)
        top = order - 1
        # counts[k][context][letter] for contexts of k letters: plain
        # counts at the top order, continuation counts below
        counts = [defaultdict(Counter) for _ in range(order)]
        self.words = 0
        for word in words:
            if not word or not set(word) <= allowed:
                continue
            self.words += 1
            text = START * top + word + END
            for i in range(top, len(text)):
                counts[top][text[i - top:i]][text[i]] += 1
        # A (context, letter) at order k counts once for each different
        # letter seen before it, i.e. once per (longer context, letter)
        # type at order k + 1
        for k in range(top - 1, -1, -1):
            for context, following in counts[k + 1].items():
                for letter in following:
                    counts[k][context[1:]][letter] += 1
        self.counts = [dict(level) for level in counts]
        self.totals = [{context: sum(following.values())
                        for context, following in level.items()}
                       for level in self.counts]
        self.discounts = [discount(level) for level in self.counts]
        self.cache = {}

    def prob(self, context, letter):
        '''
        P(letter | the word so far is context). letter is a symbol or END.
        '''
        top = self.order - 1
        context = (START * top + context)[-top:] if top else ""
        key = (context, letter)
        if key not in self.cache:
            self.cache[key] = self._prob(context, letter)
        return self.cache[key]

    def log_prob(self, context, letter):
        return math.log(self.prob(context, letter))

    def _prob(self, context, letter):
        # The next shorter context's probability, or uniform below them all
        lower = (self._prob(context[1:], letter) if context
                 else 1 / len(self.symbols))
        k = len(context)
        following = self.counts[k].get(context)
        if not following:
            return lower
        d = self.discounts[k]
        kept = max(following.get(letter, 0) - d, 0)
        # What the discounts took off, for the shorter context to share
        handed_down = d * len(following)
        return (kept + handed_down * lower) / self.totals[k][context]

    def word_log_prob(self, word):
        # log P(word), its END included
        return sum(self.log_prob(word[:i], letter)
                   for i, letter in enumerate(word + END))


def discount(level):
    '''
    Kneser-Ney's discount for one order, from how many (context, letter)
    pairs were seen once (n1) and twice (n2): D = n1 / (n1 + 2 n2).
    '''
    seen = Counter(count for following in level.values()
                   for count in following.values())
    n1, n2 = seen[1], seen[2]
    if not n1:
        return 0.5
    return min(0.95, max(0.05, n1 / (n1 + 2 * n2)))


def training_words():
    '''
    The lexicon's training words: train.py's split (ocr.split.split_words,
    15% test, no validation, words too long for the model left out), so
    the photo, cursive and lexicon test words are never counted.
    '''
    from ocr.split import split_words
    from ocr.train import CONTEXT_LENGTH, lexicon, tokens_needed
    kept = [word for word in lexicon if tokens_needed(word) <= CONTEXT_LENGTH]
    train, _, _ = split_words(kept, 0.15, 0)
    return [kept[index] for index in train]


def load_lm(order=ORDER):
    # The model over the alphabet, trained on the lexicon's training words
    from ocr.train import alphabet
    return CharLM(training_words(), alphabet, order)
