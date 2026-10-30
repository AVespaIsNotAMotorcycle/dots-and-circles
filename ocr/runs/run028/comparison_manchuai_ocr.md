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
4. CRNN v3's training data includes `HenryKingCN/scidb_manchu_v3` (23,665
   images), which is almost certainly the scidb source of
   `ocr/non_synthetic_data/` (24,280 photos). Its accuracy on those photos
   wouldn't be a held-out result.
5. Cropping matters on padded inputs. Without it run 28 loses about 20
   points.
