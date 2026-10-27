# Run 27 evaluation

Run 27 is run 26's setup (D-8/64, Adam 3e-3, print augmentation) trained on
the expanded lexicon: 21,532 entries after Norman's dictionary was added and
over-tall phrases removed, against 1,989 for run 26. It ran for 3 epochs, still
with the old 70/30 split and the old batch schedule.

## Comparing like with like

An epoch of run 27 is 11 times larger than one of run 26, so epochs aren't a
fair unit. Run 27's 3 epochs are about 497,000 training images, which is about
32 of run 26's epochs. Run 26 kept a snapshot every 5 epochs, so the main
comparison is with its **epoch-35 snapshot** (a little more training than run
27). Run 26's final weights (90 epochs) are shown as the target. The learning
rate is constant in both runs, so the schedule doesn't differ.

## Test sets

| Set | What it is |
|---|---|
| Photos | The 5,000 photographed words (seed 0) used for runs 25 and 26, whitened + cropped unless stated |
| Synthetic photo words | The same 5,000 words, rendered in a random font and augmented as in training, with no top padding |
| Synthetic lexicon | 5,000 random entries from the current lexicon, rendered the same way |

The synthetic photo words hold the words fixed, so any difference from the
photos comes from the images themselves. 2,669 of the photographed words are in
run 27's lexicon: 1,243 from the old Wiktionary lexicon and 1,426 added from
Norman. The other 2,331 are in neither.

## Headline

Each cell is exact match / character accuracy.

| | Photos | Synthetic photo words | Synthetic lexicon |
|---|---|---|---|
| **Run 27** (3 epochs) | 30.8% / 76.4% | 68.5% / 92.0% | 57.0% / 91.3% |
| Run 26, epoch 35 | 34.7% / 78.5% | 66.8% / 91.4% | 41.7% / 85.7% |
| Run 26, final (epoch 90) | 47.0% / 84.8% | 83.1% / 96.2% | 62.4% / 92.6% |

At about equal training, **run 27 is slightly behind on photos** (−3.9 points
exact), **slightly ahead on synthetic images of the same words** (+1.7), and
**far ahead on its own lexicon** (+15.3, mostly on Norman words and phrases,
which epoch 35 never saw). So the bigger lexicon has improved synthetic reading
but hasn't yet carried over to photos. The rest of this report looks at why.

All four preprocessing passes rank the same way as for run 26:

| Pass | Run 27 | Run 26 (final) |
|---|---|---|
| plain | 19.8% / 66.5% | 27.0% / 72.4% |
| cropped | 24.0% / 71.1% | 35.5% / 78.9% |
| whitened | 28.0% / 74.9% | 41.7% / 81.9% |
| whitened + cropped | 30.8% / 76.4% | 47.0% / 84.8% |

## 1. Training set makeup vs real text

The lexicon is now a dictionary's headword list, and that is quite unlike running
text:

| | Current lexicon | Old lexicon | Photographed words |
|---|---|---|---|
| Entries | 21,532 | 1,989 | 5,000 |
| Median length (letters) | 9 | 6 | 6 |
| 90th percentile length | 15 | 10 | 9 |
| Multi-word phrases | 37.7% | 14.9% | 0% |
| 1–3 letter words | 1.6% | 7.8% | 8.9% |
| Ending in **-mbi** (dictionary form of verbs) | **29.2%** | 10.9% | 5.2% |
| Ending in -me | 2.4% | 0.6% | 7.5% |
| Ending in -fi | **0.1%** | 0.4% | 5.1% |
| Ending in -re / -ra | 0.8% | 1.7% | 6.7% |
| Ending in -ci | 0.8% | 2.2% | 4.3% |
| Ending in -ngge | 1.1% | 0.5% | 5.5% |

- **Verbs are almost all in the -mbi form.** Real text mostly uses inflected
  forms (-me, -fi, -re, -ha, -ci, -ngge), which together are about 30% of the
  photographed words but under 10% of the lexicon. -fi is 50 times rarer in
  training than in the photos.
- **Short words have nearly disappeared:** 1.6% of entries against 8.9% of the
  photos. These are the particles (*i*, *be*, *de*, *ni*) that make up much of
  any page.
- **Phrases take a large share of training.** They are 38% of entries, and
  spaces alone are 5% of all training letters, while the photos are single
  words.

This shows up directly in the length results below: run 27 lost the most on
short words.

## 2. Word length

| Letters | Photos (n) | Photos, run 27 | Photos, run 26 ep. 35 | Synthetic photo words, run 27 | Synthetic photo words, run 26 ep. 35 |
|---|---|---|---|---|---|
| 1–3 | 549 | **42.8%** | **56.3%** | 66.8% | 77.6% |
| 4–5 | 1,511 | 42.0% | 47.8% | 74.5% | 76.6% |
| 6–7 | 1,850 | 26.1% | 28.9% | 68.9% | 68.0% |
| 8–9 | 1,024 | 19.0% | 18.8% | 63.1% | 50.4% |
| 10–12 | 1,198 | **11.5%** | **5.5%** | 51.8% | 32.0% |
| 13–16 | 697 | 9.1% | 9.1% | 36.4% | 9.1% |

There is a clear trade-off. Run 27 is **better on long words** (10–12 letters:
twice epoch 35's photo score, and +20 points on synthetic), and **worse on
short ones** (1–3 letters: −13.5 points on photos). This matches the shift in
the lexicon's length distribution. The particle *i* is a good example: run 27
gives an empty reading for both of its photos in this sample.

## 3. Seen vs unseen words

Photos only, exact match:

| Word | n | Run 27 | Run 26 ep. 35 | Run 26 final |
|---|---|---|---|---|
| In the old lexicon (both runs trained on it) | 1,243 | 37.8% | 45.5% | 59.1% |
| Added from Norman (only run 27 trained on it) | 1,426 | 32.3% | 35.1% | 48.4% |
| In neither lexicon | 2,331 | 26.1% | 28.7% | 39.6% |

Training on the Norman words hasn't yet made run 27 better at photographing
them than epoch 35, which never saw them (32.3% vs 35.1%). On synthetic images
of the same words it is ahead (71.7% vs 68.2%). Seeing a word in training helps
on renders, but on photos the image domain is the bigger factor so far.

Run 27 also has a smaller gap between seen and unseen words: 37.8% → 26.1%
(−11.7) against 45.5% → 28.7% (−16.8) for epoch 35. It relies less on whole-word
memory, which is the direction you want for unknown words.

## 4. Vowel harmony

Words are grouped by their vowels: **masculine** (a, o, ū with no e),
**feminine** (e with no a, o, ū), **neutral only** (just i and u), or **mixed**
(both masculine and feminine vowels, as in loanwords and compounds).

| Class | Share of lexicon | Share of photos | Photos, run 27 | Photos, run 26 ep. 35 | Synthetic lexicon, run 27 | Synthetic lexicon, run 26 ep. 35 |
|---|---|---|---|---|---|---|
| masculine | 52.9% | 48.9% | 26.4% | 30.7% | 60.5% | 43.8% |
| feminine | 21.9% | 29.9% | 32.8% | 39.8% | 62.4% | 51.8% |
| neutral only | 5.6% | 9.5% | 53.3% | 60.4% | 78.0% | 72.5% |
| mixed | **19.6%** | 11.7% | **25.3%** | **17.4%** | 35.5% | 15.9% |

- **Mixed-harmony words are the one group where run 27 beats epoch 35 on
  photos** (25.3% vs 17.4%). It even matches run 26's final weights (24.8%). The
  lexicon's mixed share went from 7.5% to 19.6%, mostly through phrases whose
  words differ in harmony. That gives the model practice reading vowels without
  relying on harmony, which matters for transliterations.
- Feminine and neutral words are underrepresented in the lexicon compared with
  the photos, and are where run 27 is furthest behind.
- *a*/*e* confusions are still among the most common errors (268 a→e and 214
  e→a on photos). They are exactly the errors harmony could resolve, and the
  model doesn't fully use it yet.

## 5. Letters

**Letter distribution.** Shares are mostly close. The biggest gaps between the
current lexicon and the photos are *e* (8.0% vs 10.9%), *m* (5.6% vs 3.8%) and
*b* (6.1% vs 5.2%). The last two come from -mbi. The transliteration letters are
still rare: k‘ 7, h‘ 2, c‘ 8, ts 22 and ž 10 occurrences in the whole lexicon.

**Per-letter recall on photos** (share of a label's letters read correctly):

| Letter | n | Run 27 | Run 26 final |
|---|---|---|---|
| w | 257 | **56%** | 78% |
| g | 1,081 | 59% | 63% |
| x (š) | 167 | **66%** | 90% |
| v (ū) | 451 | 68% | 64% |
| d | 832 | **70%** | 86% |
| a | 4,107 | 72% | 81% |
| f | 575 | **74%** | 93% |
| e | 3,288 | 75% | 81% |
| dz | 21 | **90%** | 52% |
| l | 849 | 96% | 97% |

Run 27 is notably behind on **w, š, d and f**. It now confuses **d → t** 143
times, which isn't among run 26's dozen most common confusions. The same letters are weak on synthetic
images too (f 82%, w 84%, against 96% and 91% for run 26), with **w → f** as a
new confusion. So these are undertrained letter shapes, not a photo problem.
**dz** improved a lot (52% → 90%), as you'd expect from the lexicon now having
Chinese loanwords.

**Position in the word.** On photos, run 27 reads 79.6% of first letters
correctly against 88.0% for run 26. Middle letters are 80.4% vs 85.7%, and last
letters 88.0% vs 90.0%. The **start of the word** is where it has lost the most.
Initial forms of letters look different from medial ones, and short words and
inflected forms (which start with common stems) are now rarer in training.

**Error types.** Dropping an *a* is still the most common error on every set.
Run 27 also inserts *l* far more often (131 vs 41 for run 26 on photos).

## 6. Image domain gap

For the same 5,000 words, run 27 reads 68.5% of synthetic images exactly but
30.8% of photos. Pairing each photo with the synthetic image of its word:

| | Synthetic right | Synthetic wrong |
|---|---|---|
| **Photo right** | 1,155 | 358 |
| **Photo wrong** | 2,268 | 1,219 |

In 2,268 cases the model can read the word from a render but not from the
photo. The photo-to-synthetic gap (about 38 points) is much larger than any
lexicon effect, and it's similar for epoch 35 (32 points) and run 26's final
weights (36 points). The augmentation still doesn't cover something about real
prints. Run 27's synthetic results also vary a lot by font: ShuKai is at 50.9%
and LiuYe at 59.4%, while YaBai is at 83.7%.

### Clean renders vs augmented vs photos

To see how much of the gap augmentation accounts for, the same 10,000 items
(same word and font) were also scored as clean renders, made as `generate()`
makes them but without the top padding. Each cell is exact match / character
accuracy:

| Set | Model | Clean | Augmented | Photo |
|---|---|---|---|---|
| Photo words | Run 27 | 77.1% / 94.5% | 68.5% / 92.0% | 30.8% / 76.7% |
| Photo words | Run 26, ep. 35 | 75.4% / 94.6% | 66.8% / 91.4% | 34.7% / 78.6% |
| Photo words | Run 26, final | 90.6% / 98.1% | 83.1% / 96.2% | 47.0% / 84.9% |
| Lexicon | Run 27 | 68.5% / 95.2% | 57.0% / 91.3% | – |
| Lexicon | Run 26, ep. 35 | 50.1% / 90.5% | 41.7% / 85.7% | – |
| Lexicon | Run 26, final | 74.2% / 96.3% | 62.4% / 92.6% | – |

For run 27 on the photographed words, the steps break down like this:

- **Clean → augmented: −8.6 points.** Augmentation makes renders harder, as
  intended.
- **Augmented → photo: −37.7 points**, more than four times as large. Run 26's
  final weights show the same pattern (−7.5 then −36.1).
- **Scale doesn't explain it.** Clean renders, augmented renders and photos all
  average about 14–15 px per letter.

Splitting each word by whether the model reads it exactly on the clean render,
the augmented render and the photo (run 27):

| Clean | Augmented | Photo | Words |
|---|---|---|---|
| right | right | **wrong** | **1,937** |
| right | right | right | 1,054 |
| right | wrong | wrong | 645 |
| wrong | wrong | wrong | 562 |
| wrong | right | wrong | 318 |
| right | wrong | right | 219 |
| wrong | wrong | right | 151 |
| wrong | right | right | 114 |

The largest group, 39% of all words, is read correctly on **both** synthetic
versions but not on the photo. That's the real-world gap. Augmentation as it
stands reproduces only a small part of what makes photos hard. What's left
could be the books' typefaces (none of the 11 fonts), print and photo artifacts
the augmentation doesn't imitate, crop problems, or label noise in the photo
dataset. These results can't separate those; looking at a sample of those
1,937 photos by eye would.

Clean renders aren't solved either. Run 27 misses 23% of them, mostly by
dropping or doubling a single letter (*mimbe → mibe*, *dung → dungng*). That's
more evidence that it's undertrained. Run 26's final weights miss 9.4%.

## 7. Image density

Grouping by tokens per letter (image height ÷ 4 ÷ letters):

| Tokens per letter | Photos (n) | Photos, run 27 | Photos, run 26 ep. 35 |
|---|---|---|---|
| < 2 | 453 | 13.0% | 2.2% |
| 2–3 | 1,907 | 29.7% | 30.9% |
| 3–4 | 1,818 | 32.3% | 36.1% |
| 4–5 | 913 | 29.4% | 35.5% |
| 5+ | 636 | 32.2% | 41.6% |

Run 27 copes much better with densely written words (under 2 tokens per letter:
13.0% vs 2.2%), presumably from the long, squashed lexicon entries. It is worse
on sparse, tall images, which are mostly short words.

## Takeaways

1. **Run 27 is undertrained, not worse.** At equal training it's within 4
   points of run 26 on photos, and better on long words, mixed-harmony words,
   dense words and dz. Training it longer is the most informative next step.
   The 85/15 split and even test schedule are now in place for that.
2. **The lexicon is skewed towards dictionary forms.** 29% of entries end in
   -mbi, inflected forms like -fi barely appear, short particles fell from 7.8%
   to 1.6%, and 38% of entries are phrases. Real text looks nothing like that.
   Options include sampling entries so that lengths and endings match running
   text, weighting short words up, or generating inflected forms of the -mbi
   verbs.
3. **The image gap is still the biggest factor:** about 38 points between photos
   and renders of the same words, with no lexicon change touching it. More
   variation in the augmentation (and checking it against the weak fonts and
   real prints) is likely to matter more than words.
4. **Watch w/f, d/t and š.** These letters are weak even on synthetic images,
   so check them again after longer training before assuming the data is the
   cause.
5. **The particle *i* is still misread.** It needs to be read reliably on its
   own, since it is one of the most common words on any page.

Files in this folder: `non_synthetic_eval_*.csv` (run 27 on the photos),
`non_synthetic_eval_whitened_cropped_run026_epoch035.csv` (run 26's epoch-35
snapshot), `synthetic_eval_run027_run026.csv` (both synthetic sets, augmented,
one row per image and model) and `synthetic_clean_eval_run027_run026.csv` (the
same items as clean renders).
