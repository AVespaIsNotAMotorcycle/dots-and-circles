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
(`ocr/box_classifier.py`), and reads each Manchu word with Run 31.

On 60 synthetic pages (`python -m ocr.evaluate_pages`) of words no model
trained on, a third each of all-Manchu, alternating Manchu and Chinese
columns, and Manchu with Chinese characters mixed in, and about a third
with small Chinese glosses between the Manchu columns. Some words are
followed by the genitive particle *i*, printed apart as a hook, and about
a third of pages are set tight, each word's tail running alongside the
next word's start, as in much real print:

| Pages | Boxes found (precision / recall) | Kind right (Manchu / Chinese) | Manchu words exact | Page text character accuracy |
|---|---|---|---|---|
| Clean | 96.9% / 97.8% | 99.5% / 100% | 90.0% | 95.6% |
| Made to look photographed | 96.9% / 97.6% | 99.5% / 100% | 89.7% | 95.5% |

Words set that tight often share a box. Run 31 was trained on tight
phrases, so it writes the space between such words itself, and the
reader splits the box there. Run 30 wasn't. For it, the reader splits
such a box where its stem breaks, if the model reads the parts more
confidently than the whole (92.4% page text on the clean pages). Run 31
doesn't need that, and it only cut lone letters off its words, so its
`config.json` entry turns it off (`stem_breaks`). A page takes about
0.6 s on one CPU core.

On seven real pages so far, in [`ocr/real_pages`](ocr/real_pages) and
scored by `python -m ocr.evaluate_real_pages`: six printed (one with six
columns of small Chinese glosses) and one handwritten letter. With Run
31, on five of the printed pages the Manchu text reads at 90.7–99.0%
character / 80.3–94.7% word accuracy, with every gloss left out. The
sixth, set tightest, reads at 90.2% (Run 30: 86.3%). Together the
printed pages read at 95.5% / 86.7% (Run 30: 94.9% / 83.1%). The
handwritten letter reads at 43.4% characters (Run 30: 48.8%).

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
- [Haipeng SUN, Wenhao TAO, Xiaojun BI. A dataset of Manchu ancient book words for OCR[J]. China Scientific Data, 2026, 11(1): 1-11. DOI: 10.11922/11-6035.csd.2025.0069.zh.](https://www.scidb.cn/en/detail?dataSetId=b45491b63d694534a9323acf14846586)

Fonts:
- [A set of 11 from Manchu Studies Group](https://www.manchustudiesgroup.org/typing-manchu/)
- [Abkai Xanyan from abkai.net](https://abkai.net/zh/manchu/manchu-fonts/)

## Limitations
- It's made for printed Manchu, and struggles with other styles, such as manuscripts.
- It struggles with words set close together.
- It sometimes takes decorations or other text (such as Chinese) for Manchu, or Manchu for other text.
- It may flag words that are right as possible misreadings: its dictionaries are fairly small, and don't know every suffix.
