# Real pages

Photographed or scanned pages of print, with their transcriptions, for
scoring the page reader (`python -m ocr.evaluate_real_pages`). Each page is
an image `pageNNN.png` (or `.jpg`) and a transcription `pageNNN.txt`: one
line per column of Manchu, left to right, in Abkai. Columns of Chinese
aren't transcribed; punctuation may be, and is ignored when scoring.

## page001

- **Image:** a crop of a page of *Sonjofi ubaliyambuha Liyoo jai j'i i
  bithe* (擇繙聊齋志異), Jakdan's (扎克丹) Manchu translation of Pu
  Songling's *Liaozhai zhiyi*: Stabi Berlin, 4° 41322 ROA, page
  PHYS_0041,
  <http://resolver.staatsbibliothek-berlin.de/SBB0000862300000000>.
  The scan is the Staatsbibliothek zu Berlin's own, of a public-domain
  original, and under its
  [terms of use](https://digital.staatsbibliothek-berlin.de/nutzungsbedingungen)
  is public domain:
  [Public Domain Mark 1.0](https://creativecommons.org/publicdomain/mark/1.0/).
- **Transcription:** from [manc.hu](https://manc.hu/) (© Fresco Sam-Sin,
  Léon Rodenburg).

## page002 to page004

Sources not yet recorded:

- **page002:** apparently the same book as page001 (the story 考城隍).
- **page003:** a woodblock print of military regulations (qoohai fafun),
  opening with an edict of Yongzheng 9 (1731).
- **page004:** a handwritten letter from the Daoguang emperor, dated
  Daoguang 24 (1844).

Transcriptions supplied with the images; page004's footnote marks ([1],
[2]) are ignored when scoring.

## Results

Characters / words, as of 2026-11-10:

| Model | page001 | page002 | page003 | page004 (handwritten) |
|---|---|---|---|---|
| Run 30 | 97.6% / 87.3% | 93.0% / 73.6% | 90.1% / 76.3% | 48.3% / 3.4% |
| Run 28 | 88.2% / 51.9% | 86.0% / 50.6% | 88.0% / 55.3% | 42.2% / 1.0% |
| Run 26 | 82.0% / 44.3% | 77.0% / 34.5% | 80.7% / 44.7% | 37.8% / 2.4% |

On the printed pages, most of Run 30's word errors are pairs of words set
so close that they're read as one (`hotonde` for `hoton de`). The
handwritten letter's columns are found, but the models, trained on
printed fonts and photographed print, read its hand poorly.
