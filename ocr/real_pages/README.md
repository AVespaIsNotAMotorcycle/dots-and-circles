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

## page002

- **Image:** a crop of another page of the same book as page001, from the
  same story (考城隍), by way of [manc.hu](https://manc.hu/): Stabi Berlin,
  <http://resolver.staatsbibliothek-berlin.de/SBB0000862300000000>, page
  [PHYS_0043](https://digital.staatsbibliothek-berlin.de/werkansicht?PPN=PPN3308101262&PHYSID=PHYS_0043&DMDID=DMDLOG_0011).
  Public domain, as for page001.
- **Transcription:** from [manc.hu](https://manc.hu/) (© Fresco Sam-Sin,
  Léon Rodenburg).

## page003

- **Image:** a crop of a Stabi Berlin scan, by way of
  [manc.hu](https://manc.hu/): a woodblock print of military regulations
  (*qoohai fafun*, 軍令), opening with an edict of Yongzheng 9 (1731):
  Stabi Berlin, PPN3306131504, page
  [PHYS_0003](https://digital.staatsbibliothek-berlin.de/werkansicht?PPN=PPN3306131504&PHYSID=PHYS_0003&DMDID=DMDLOG_0002).
  As a Staatsbibliothek scan of a public-domain original, it's public
  domain under its terms of use, as for page001.
- **Transcription:** from [manc.hu](https://manc.hu/) (© Fresco Sam-Sin,
  Léon Rodenburg).

## page004

- **Image:** from [manc.hu](https://manc.hu/): a handwritten letter from
  the Daoguang emperor to President John Tyler (*be lii hi tiyan*,
  伯理璽天德), dated Daoguang 24 (1844). The letter itself is public
  domain; where manc.hu's reproduction comes from isn't recorded. On the
  letter, see CHU Shih-Chia, "Tao-Kuang to President Tyler", *Harvard
  Journal of Asiatic Studies* 7, no. 3 (1943): 169–173.
- **Transcription:** from [manc.hu](https://manc.hu/), whose
  transliteration was shared with them by Guillaume Lescuyer. Its footnote
  marks ([1], [2]) are ignored when scoring.

## page005 and page006

- **Images:** two pages of the Fodo Mama prayers (*Fodo Mama court*), from
  [manc.hu](https://manc.hu/en/tools/reader/text/36648/Fodo-Mama-court);
  page005 is cropped to leave out the margin. manc.hu doesn't say where
  its scans come from; probably Tatiana Pang, "The worship of Fodo-mama:
  progenitor of the Manchus".
- **Transcriptions:** from the same manc.hu text (© Fresco Sam-Sin, Léon
  Rodenburg). page005's second line holds a note printed small in two
  narrow columns; the transcription gives it inline in parentheses, which
  are ignored when scoring like other punctuation. manc.hu's `hvisun` and
  `hvituri` (twice) in page005 are corrected to `hvsun` and `hvturi`, as
  the print has them.

## page007

- **Image:** the preface (*bithei sioi*) of *Han i araha ambasai mujilen
  be targabure bithe*, from
  [manc.hu](https://manc.hu/en/tools/reader/text/123759/00%20Bithei%20sioi),
  which doesn't say where the scan comes from.
- **Transcription:** from the same manc.hu text (© Fresco Sam-Sin, Léon
  Rodenburg). It capitalizes the title's words; scoring is in lower case.

`sources.json` gives each page's credit and link in short, for the
server (`/random?source=page`) to show beside it.

## Results

Characters / words, as of 2026-11-10:

| Model | page001 | page002 | page003 | page004 (handwritten) | page005 | page006 | page007 |
|---|---|---|---|---|---|---|---|
| Run 30 | 97.9% / 88.6% | 93.8% / 75.9% | 99.1% / 97.4% | 47.4% / 3.9% | 90.5% / 78.9% | 98.6% / 92.1% | 78.4% / 26.3% |
| Run 28 | 87.9% / 51.9% | 86.8% / 52.9% | 94.0% / 68.4% | 39.2% / 1.9% | 85.6% / 59.2% | 94.2% / 73.7% | 70.6% / 26.3% |
| Run 26 | 82.0% / 44.3% | 77.7% / 36.8% | 89.3% / 63.2% | 35.9% / 2.9% | 80.7% / 47.9% | 88.5% / 63.2% | 21.6% / 10.5% |

On the printed pages, most of Run 30's word errors are pairs of words set
so close that they're read as one (`hotonde` for `hoton de`); page007 is
the worst case, its words' tails running alongside the next word's start
(`beyebeilibure` for `beye be ilibure`). The
handwritten letter's columns are found, but the models, trained on
printed fonts and photographed print, read its hand poorly. On page005
the small two-column note is read, but as a line of its own. Of the 14
genitive *i* on the printed pages, Run 30's reading has 9; most of the
rest are joined onto the word before them (`hotoni` for `hoton i`).
Readings with a confidence under 0.65 are left out of the text, which
takes most junk (a page's edge read as Manchu) out of it; on the
handwritten letter it also drops some uncertain words.
