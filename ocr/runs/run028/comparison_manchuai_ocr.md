# Handwritten test sets: run 28 against the ManchuAI-OCR models

Chung & Choi, "Finetuning Vision-Language Models as OCR Systems for
Low-Resource Languages: A Case Study of Manchu" (arXiv 2507.06761v1, July
2025; code at [mic7ch1/ManchuAI-OCR](https://github.com/mic7ch1/ManchuAI-OCR))
fine-tune LLaMA-3.2-11B and Qwen2.5-VL on 60,000 synthetic word images and
compare them with a CRNN ("CRNN-3M") trained on the same data. Both are
tested on handwritten words. A later CRNN by the same authors,
[mic7ch/crnn-manchu-ngsyd](https://huggingface.co/mic7ch/crnn-manchu-ngsyd)
(CC-BY-NC-4.0), is trained on real handwritten data and is also included.

Run 28 (D-8/64, Adam 3e-3, final epoch-10 weights) has never seen
handwriting. It's trained only on rendered fonts with print augmentation.

## Test sets

- **Paper test set (218 words):** handwritten words from the *Neige Cangben
  Manwen Laodang*, published as
  [mic7ch/test_data_long](https://huggingface.co/datasets/mic7ch/test_data_long)
  (train + validation splits, 174 + 44 = 218, the paper's count). The
  images are binarized and black on white, rotated 90° counter-clockwise and
  padded to 480 × 64.
- **Repository test set (753 words):** the `test` split of
  [mic7ch/manchu](https://huggingface.co/datasets/mic7ch/manchu), which the
  repository's current results use. The CRNN model card calls it
  `mic7ch/manchu-2025-0033` test, "real data". The repository publishes
  per-image predictions for CRNN-3M and LLaMA-3.2-11B on it, so it allows
  paired comparisons. The arXiv paper has only v1, so it doesn't report this
  set.

Neither dataset states a license, so this report gives only aggregate
results.

## Method

- **Run 28:** each image is rotated 90° clockwise back upright (checked
  against a rendering of *amala*), then put through all four
  `ocr/evaluate_non_synthetic.py` passes via `ocr.preprocess.prepare`. The
  padding in these images makes `plain` and `whitened` unfair, since the
  word fills only part of the frame. **Whitened + cropped**, the default
  chosen before seeing this data, is the headline figure. Choosing the best
  pass on the test set would be test-set selection.
- **CRNN v3** (`checkpoint-426284.pth`, chosen by validation only): their
  `src/CRNN/inference.py` preprocessing (resize to 64 × 480, ImageNet
  normalization) and greedy CTC decoding. It scores 97.34% on the 753-word
  set in its model card, and 97.3% here, so the setup reproduces it.
- **CRNN-3M and LLaMA-3.2-11B:** the 218-word numbers are from the paper.
  The 753-word numbers are recomputed from the repository's
  `results/predictions/*/best_checkpoint/test.json`. Of those rows, 715 still
  match the current Hugging Face labels; the dataset was updated after the
  run. The paired comparison uses only those 715.
- **Scoring**, identical for every model: Manchu script → Abkai with
  `ocr.photos.MANCHU_TO_ABKAI` (unmapped punctuation dropped), then their
  protocol's w → f before i/u/ū. CER is total edits over total label
  letters.
- **Speed:** run 28 in one process with `OPENBLAS_NUM_THREADS=1` on a Ryzen 7
  7730U laptop CPU, preprocessing included. CRNN v3 on the same CPU,
  preprocessing excluded. CRNN-3M and LLaMA-3.2-11B speeds are from the
  paper, on an NVIDIA RTX 6000 Ada (48 GB).

## Paper test set (218 handwritten words)

| Model | Training data | Exact | CER | Speed |
|---|---|---|---|---|
| CRNN v3 | ~96k real + synthetic images | 97.7% | 0.006 | 4 img/s (1 CPU thread), 16 (8 threads) |
| LLaMA-3.2-11B | 60k synthetic, LoRA | 93.1% | 0.022 | 0.11 img/s (GPU) |
| **Run 28, whitened + cropped** | rendered fonts only | **73.9%** | **0.062** | **~135 img/s (1 CPU core)** |
| CRNN-3M | 60k synthetic | 72.5% | 0.093 | ~270 img/s (GPU) |
| Qwen2.5-VL-7B | 60k synthetic, LoRA | 43.1% | 0.254 | 1.1 img/s (GPU) |
| Qwen2.5-VL-3B | 60k synthetic, LoRA | 23.9% | 0.368 | 1.4 img/s (GPU) |

Run 28's other passes: cropped 76.6% (CER 0.057), plain 50.5%, whitened
50.5%.

With 218 words, the 95% interval on an accuracy near 73% is about ±6
points, so run 28 and CRNN-3M can't be told apart on this set. The paper
also chose the CRNN-3M and LLaMA checkpoints by their score on this same
test set (§5.1), so those two figures are optimistic.

## Repository test set (753 words)

| Model | Exact |
|---|---|
| CRNN v3 | 97.3% |
| LLaMA-3.2-11B | 87.4% |
| **Run 28, whitened + cropped** | **60.4%** (char 88.7%, CER 0.108) |
| CRNN-3M | 56.8% |

Run 28's cropped pass gives the same result. Plain and whitened give 40.8%.

### Paired, on the 715 rows whose labels are unchanged

| | CRNN-3M | LLaMA-3.2-11B |
|---|---|---|
| Their exact | 56.1% | 88.0% |
| Run 28 exact | 60.6% | 60.6% |
| Right only in run 28 | 187 | 50 |
| Right only in theirs | 155 | 246 |
| McNemar z | +1.73 (p ≈ 0.08) | −11.4 |

## What `mic7ch/manchu` contains

The 753-word test set is the `test` split of `mic7ch/manchu`. Profiling all
three splits (labels mapped to Abkai, images hashed) shows the dataset joins
two others:

| Part | Rows | Source |
|---|---|---|
| Synthetic train / validation (rows 0–59,999 / 0–14,999) | 60,000 / 15,000 | `mic7ch/manchu-2025-0033` train / validation |
| Real train / validation (the rest) | 20,306 / 3,359 | `HenryKingCN/scidb_manchu_v2` (`v3` has the same counts) |
| Test | 753 | `mic7ch/manchu-2025-0033` test ("real data", source not stated) |

All images are 480 × 64 RGB PNGs.

### Its real part is our photo set

`ocr/non_synthetic_data/` holds 24,280 photos of 2,428 words from the same
scidb.cn dataset. The real part of `mic7ch/manchu` has 23,665 images of 2,417
words:

- **Labels:** 2,210 labels are identical to our folder names.
- **Counts per word:** for 1,823 of our 2,428 words the image counts match
  exactly. Almost all of the rest have one image fewer (e.g. 10 against 9).
- **Spelling:** the 207 labels that aren't among our folder names (2,026
  rows) are spelling variants. The Manchu labels write ᠸ (w) for f before
  i/u, so `gajifi` appears as `gajiwi`. The two letters look the same in
  that position, and the CRNN v3 protocol normalizes w → f for this reason.

So any model trained on this real part has trained on almost all of our
photos.

### Which models trained on it

No model card names `mic7ch/manchu` itself. The cards name its parts:

| Models | Training data | Seen our photos? |
|---|---|---|
| Paper models: `manchu-ocr-{qwen-25-3b, qwen-25-7b, llama-32-11b}`, CRNN-3M | `mic7ch/manchu-2025-0033` (`configs/base.yaml`), 60k synthetic | no |
| `manchu-ocr-*-step1-syn*` | synthetic only | no |
| `manchu-ocr-crnn-{step1-real, mix, final}` | `manchu-2025-0033` + `scidb_manchu_v2` ("60k synthetic, then 20k real") | yes |
| `manchu-ocr-{llama, qwen, pixtral}-{step1-real, mix, final}*` | cards list only 0033, but are named and selected on `real_val` | almost certainly |
| `crnn-manchu-ngsyd` | real handwriting + `scidb_manchu_v3` | yes |

Our photos therefore can't serve as a held-out comparison with any of the
2026 models. The 753-word test set has no byte-identical images in train or
validation, and run 28 has seen none of it. That makes it the cleanest
shared benchmark.

### Compared with our data

| | Their synthetic | Their real (our photos) | Their test | Our lexicon |
|---|---|---|---|---|
| Images | 75,000, one fixed render each | 23,665 photos | 753 | 21,532 words × 12 fonts, re-augmented each epoch |
| Distinct words | 57,179 | 2,417 | 569 | 21,532 |
| Mean length (Abkai letters) | 10.2 | 6.1 | 5.5 | — |
| 1–3 letter words | 0.5% | 9.0% | 12.6% | 1.6% |
| In our lexicon | 9.3% | 51.6% | 57.4% | 100% |

- **Their synthetic words are long and rare.** They're sampled from Zhang
  Zhuohui's 130,917-word dictionary, so they share our lexicon's skew
  toward long, rare words, only more strongly. Short words are 0.5% of their
  synthetic data, against 9–13% of real text. This may explain much of the
  drop that CRNN-3M shows on real data (99.8% → 72.5%), alongside the visual
  difference between renders and handwriting.
- **The vocabularies barely overlap.** Only 9% of their synthetic words are
  in our lexicon, and only 25% of our lexicon is in their synthetic data.
- **Their real data is split by image, not by word.** Every word in the real
  validation split also appears in the real training split, and 83% of
  test rows are words seen in training. The 96–99% `real_val` and test
  scores in the 2026 model cards therefore mostly measure new images of
  known words. Run 28's photo result on words in no lexicon (58.4% exact)
  measures something harder.

## Conclusions

1. On handwriting it has never seen, run 28 sits between the paper's
   CRNN-3M and LLaMA-3.2-11B on both test sets. It is ahead of CRNN-3M by
   1.4 and 4.5 points, which is borderline significant at best.
2. Run 28 is about 1,200× faster than LLaMA-3.2-11B, on a single laptop core
   instead of a 48 GB GPU. On CPU it is also about 8× faster than CRNN v3
   running on 8 threads.
3. CRNN v3, trained on ~96k real images, beats every other model here, so
   the paper's two models no longer bracket the state of the art. Run 28's
   case rests on needing no real labeled data and no GPU.
4. The real part of `mic7ch/manchu` is our photo set (see above). CRNN v3
   and every 2026 `step1-real`, `mix` and `final` model trained on it, so
   none of them can be compared with run 28 on our photos. The 753-word
   test set is the fair shared benchmark.
5. Cropping matters on padded inputs. Without it run 28 loses about 20
   points.
6. When comparing against their labels, normalize w → f before i/u.
   Otherwise about 9% of their real rows look wrong when they aren't.
