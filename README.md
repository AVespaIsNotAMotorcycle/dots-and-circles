# Dots and Circles

## Summary

- Transformer-based OCR for Manchu script
- Neural net code handwritten without autograd, then trained on all-synthetic data
- Against real-world data, guessed 62.3% of words exactly correct (90.5% character accuracy)

## Architecture

Best performance: Run 28
- 1 input token is 4 rows of pixels (32px each)
- Tokens are collapsed down to a 64 dimension embedding vector by a linear layer
- Tokens are then fed through 8 transformer blocks to produce CTC output
- Optimized with Adam, with a learning rate of 3e-3
- Entirely synthetic data, generated from 21,532 words (Wiktionary + Norman dictionary) and 12 fonts

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
[Tatu Ylonen: Wiktextract: Wiktionary as Machine-Readable Structured Data, Proceedings of the 13th Conference on Language Resources and Evaluation (LREC), pp. 1317-1325, Marseille, 20-25 June 2022.](https://kaikki.org/dictionary/Manchu/index.html)

- [The Jerry Norman dictionary](https://github.com/purobaburi/manchu-resources/blob/main/Dictionaries/Jerry_Norman_Dict.txt)

Test data:
[Haipeng SUN, Wenhao TAO, Xiaojun BI. A dataset of Manchu ancient book words for OCR[J]. China Scientific Data, 2026, 11(1): 1-11. DOI: 10.11922/11-6035.csd.2025.0069.zh.](https://www.scidb.cn/en/detail?dataSetId=b45491b63d694534a9323acf14846586)

Fonts:
- [A set of 11 from Manchu Studies Group](https://www.manchustudiesgroup.org/typing-manchu/)
- [Abkai Xanyan from abkai.net](https://abkai.net/zh/manchu/manchu-fonts/)


