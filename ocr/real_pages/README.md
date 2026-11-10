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

## Results

| Model | page001 characters | page001 words |
|---|---|---|
| Run 30 | 95.7% | 81.0% |
| Run 28 | 89.8% | 55.7% |
| Run 26 | 82.0% | 44.3% |

As of 2026-11-10. Most of Run 30's word errors are pairs of words set so
close that they're read as one (`hotonde` for `hoton de`).
