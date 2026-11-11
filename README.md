# Dots and Circles

## Summary

- Transformer-based OCR for Manchu script
- Neural net code handwritten without autograd, then trained on synthetic data plus photographed print words
- Against whitened and cropped photographs of words it never trained on, guessed 92.9% of words exactly correct (98.5% character accuracy)
- [Try the live demo](https://dots-and-circles.sasha-m-ebersole.com/)

## Results
| Test set | Run 30 | Run 29 | ManchuOCR | CRNN-3M | LLaMA-3.2-11B | CRNN v3 |
|---|---|---|---|---|---|---|
| Photographed words, held-out benchmark (Sun et al. 2026) | 92.9% | 72.1% | 69.2%\* | | | |
| ManchuAI-OCR handwritten (218 words) | 90.4% | 76.1% | 60.6% | 72.5%† | 93.1%† | 97.7% |
| ManchuAI-OCR handwritten (753 words) | 85.5% | 70.3% | 58.0% | 56.8% | 87.4% | 97.3% |

Numbers show percentage of words guessed exactly correctly.

\* With Dots and Circles' preprocessing, since it did better with it on that dataset.

† Checkpoint chosen on this test set, so the figure is optimistic.

Run 30 trained on rendered fonts plus 85% of the photographed words; the benchmark is the other 315 words (3,150 photos), which it never saw. It has never seen handwriting. Run 29 trained only on rendered fonts. CRNN v3 trained on these same photos (split by image, not by word), so it can't be scored on the benchmark. Both runs process ~90 img/s on one CPU core.
Accuracy figures are from [this report](ocr/runs/run030/comparison_epoch018_benchmark_manchuai.md).

## Reading whole pages

`POST /upload/multiple` (see [the server README](server/README.md)) reads a
page of print: it finds the columns and words with classical image
processing (`ocr/layout.py`), tells Manchu from Chinese with a small MLP
(`ocr/box_classifier.py`), and reads each Manchu word with Run 30.

On 100 synthetic pages (`python -m ocr.evaluate_pages`) of words no model
trained on, a third each of all-Manchu, alternating Manchu and Chinese
columns, and Manchu with Chinese characters mixed in, and about a third
with small Chinese glosses between the Manchu columns:

| Pages | Boxes found (precision / recall) | Kind right (Manchu / Chinese) | Manchu words exact | Page text character accuracy |
|---|---|---|---|---|
| Clean | 98.6% / 99.0% | 99.5% / 100% | 98.0% | 98.2% |
| Made to look photographed | 98.7% / 98.9% | 99.6% / 99.7% | 97.6% | 98.2% |

With Run 28 the photographed-looking pages read 93.2% exactly, and with
Run 26 80.6%. A page takes about a second on one CPU core.

On six real pages so far, in [`ocr/real_pages`](ocr/real_pages) and
scored by `python -m ocr.evaluate_real_pages`: five printed (one with six
columns of small Chinese glosses) and one handwritten letter. On the
printed pages the Manchu text reads at 89.1–97.6% character / 71.8–89.5%
word accuracy, with every gloss left out; most word errors are pairs of
words set so close that they're read as one (`hotonde` for `hoton de`).
The handwritten letter reads at 48.3% characters.

## Architecture

Best performance: Run 30
- 1 input token is 4 rows of pixels (32px each)
- Tokens are collapsed down to a 64 dimension embedding vector by a linear layer
- Tokens are then fed through 12 transformer blocks to produce CTC output
- Dropout of 0.1
- Optimized with Adam, with a learning rate of 3e-3 for 10 epochs, then 1e-3 for 8
- Epochs 1–14 (run 29) on entirely synthetic data, generated from 21,532 words (Wiktionary + Norman dictionary) and 12 fonts
- Epochs 15–18 add 20,630 photos of 2,063 printed words (85% of the dataset's words) to each epoch

### What was hand-written

A key aspect of this project was the desire to better understand how machine learning works, with an emphasis on transformers, loss calculation, optimization algorithms, and backprop.
- nn/: everything but PositionalEmbedding, ReadoutTokens, and TakeFirst was hand-written by me, though later bugfixes were done with Claude.
- loss/: MSE and cross-entropy loss were first hand-written in nn/tests, so Claude simply ported them over. CTC was hand-written, with later optimizations done by Claude.
- optim/: initially hand-written by me, but SGD's gradient clipping was later added by Claude.

### What was delegated to Claude

The training loop and training data generation was almost entirely Claude, since the focus of the project as a learning exercise was to write a usable transformer architecture. Training transformer models was covered by my [GPT-2 replication following Sebastian Raschka's book](https://github.com/AVespaIsNotAMotorcycle/Raschka-LLM), and a previous non-transformer-based iteration of this project (everything before [this commit](https://github.com/AVespaIsNotAMotorcycle/dots-and-circles/commit/579f91d6ff230e1b7ebd4d3c4aa09e90075fbf77)) also had a hand-written synthetic training data generator, training loop, and Flask server.
- ocr/: outlined by me, but almost all actual code was written by Claude.
- server/: written by Claude.

## Resources Used

Training data:
- [Tatu Ylonen: Wiktextract: Wiktionary as Machine-Readable Structured Data, Proceedings of the 13th Conference on Language Resources and Evaluation (LREC), pp. 1317-1325, Marseille, 20-25 June 2022.](https://kaikki.org/dictionary/Manchu/index.html)
- [The Jerry Norman dictionary](https://github.com/purobaburi/manchu-resources/blob/main/Dictionaries/Jerry_Norman_Dict.txt)

Test data:
- [Haipeng SUN, Wenhao TAO, Xiaojun BI. A dataset of Manchu ancient book words for OCR[J]. China Scientific Data, 2026, 11(1): 1-11. DOI: 10.11922/11-6035.csd.2025.0069.zh.](https://www.scidb.cn/en/detail?dataSetId=b45491b63d694534a9323acf14846586)

Fonts:
- [A set of 11 from Manchu Studies Group](https://www.manchustudiesgroup.org/typing-manchu/)
- [Abkai Xanyan from abkai.net](https://abkai.net/zh/manchu/manchu-fonts/)

## Limitations
- Poor performance on free variation selectors
- Poor performance on non-preprocessed (whitened+cropped) images
- Poor performance on long words
- Page reading is tested on only one real page, and doesn't handle curved
  pages, perspective, two-line small Chinese glosses or Mongolian; Manchu
  punctuation is left out of the text
