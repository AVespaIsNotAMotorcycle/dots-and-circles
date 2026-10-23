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

## Roadmap

- [x] **Under the hood**:
  - [x] **nn module**: provides the basis for the implementation of ANNs
  - [x] **loss module**: calculates loss and gradients
  - [x] **optim module**: updates parameters based on gradients
- [ ] **Basic OCR**:
  - [x] **Simple synthetic training data generation**: black text on white background
  - [ ] **Transformer OCR Model**: reads image line by line, outputs CTC string
  - [ ] **CTC loss function**: generates loss values for TrOCR output
  - [ ] **Simple training loop**: train TrOCR on synthetic data, graph results


# Notes

There's no automatic differentiation. This means that backpropogation must be manually implemented for each Module, and has resulted in some errors. For example, I forgot to account for shortcut layers around the attention and MLP blocks of the Transformer, so the gradients were off. Claude caught that in a later pass.

Another problem downstream of this is that, to make things marginally easier, I averaged out loss and gradients when doing backprop. Tests still showed decreasing loss, so I figured it was fine. However, when trying to train the actual OCR model, accuracy stalled out quickly, so I had to undo it. Compare before and after this change:

| Before                                                              | After                                                               |
| ------------------------------------------------------------------- | ------------------------------------------------------------------- |
| <img width="640" height="480" alt="image" src="./ocr/run001.png" /> | <img width="640" height="480" alt="image" src="./ocr/run002.png" /> |
