# Dots and Circles

## Architecture

### Server

A flask server with at least one endpoint:
GET /ocr
Requires:
- an attached image
Returns:
- an array of bounding boxes
- the text that was found

### Bounding Box Generator

Draws bounding boxes around all words on a page (but *not* around anything
else) and labels the boxes with the script the word is written in.

Requirements:
- Draws boxes around each word
- Does not draw boxes around non-word elements
- Does not overlap bounding boxes
- Labels word script correctly

Manchu-script words are passed on the the Lexigraph OCR.

### Lexigraph OCR

Given a lexigraph (an image of a word, i.e. the contents of a bounding box)
returns the Unicode characters representing the word. Composed of one or more
transformers, using CTC for loss calculation.

#### Training Data

A set of Manchu words was pulled from
[kaikki](https://kaikki.org/dictionary/Manchu/index.html).
