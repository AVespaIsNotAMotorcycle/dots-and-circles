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


# Progress

I didn't start using Claude code until after the `nn` module was fully implemented, since I wanted to work through it myself.

There's no automatic differentiation. This means that backpropogation must be manually implemented for each Module, and has resulted in some errors. For example, I forgot to account for shortcut layers around the attention and MLP blocks of the Transformer, so the gradients were off. Claude caught that in a later pass.

Another problem downstream of this is that, to make things marginally easier, I averaged out loss and gradients when doing backprop.
Tests still showed decreasing loss, so I figured it was fine.
However, when trying to train the actual OCR model, accuracy stalled out quickly, so I had to undo it.
Accuracy here is measured as the percentage of the first n output characters which are correct, where n is the length of the actual word.
Compare before and after this change:

| Before                                                              | After                                                               |
| ------------------------------------------------------------------- | ------------------------------------------------------------------- |
| <img width="640" height="480" alt="image" src="./ocr/run_graphs/run001.png" /> | <img width="640" height="480" alt="image" src="./ocr/run_graphs/run002.png" /> |

Since I hadn't yet implemented CTC loss calculation and was just using Cross-Entropy Loss, the misalignment between input tokens (rows of pixels) and output (characters) was causing issues. Because there was a causal attention mask in the multi-head attention module, the OCR model was attempting to output characters corresponding to parts of the image it hadn't yet seen. Postponing the inevitable CTC implementation, I added an option to disable the mask, which marginally improved accuracy.

| Before                                                              | After                                                               |
| ------------------------------------------------------------------- | ------------------------------------------------------------------- |
| <img width="640" height="480" alt="image" src="./ocr/run_graphs/run002.png" /> | <img width="640" height="480" alt="image" src="./ocr/run_graphs/run003.png" /> |

I then tried updating the learn rate throughout training, starting with a high value, 0.1, which drops over time so that it's about 0.0027 by the end of the first epoch.

| Before                                                              | After                                                               |
| ------------------------------------------------------------------- | ------------------------------------------------------------------- |
| <img width="640" height="480" alt="image" src="./ocr/run_graphs/run003.png" /> | <img width="640" height="480" alt="image" src="./ocr/run_graphs/run004.png" /> |

Next, tokens were changed from being one row of pixels to four, which are then compressed by a factor of four by being run through a linear layer.

| Before                                                              | After                                                               |
| ------------------------------------------------------------------- | ------------------------------------------------------------------- |
| <img width="640" height="480" alt="image" src="./ocr/run_graphs/run004.png" /> | <img width="640" height="480" alt="image" src="./ocr/run_graphs/run005.png" /> |

Positional encoding (non-learned, sinosoidal) was added.
The model was then split into an A and B variant: A stayed the same, and B got readout tokens.
Model B initially learned faster, but plateaued with an accuracy around .2, and Model A quickly caught up and plateaued at the same point.
<img width="1280" height="960" alt="image" src="./ocr/run_graphs/run007.png" />

CTC loss was implemented, and a CTC model (Model C) was compared against Model B (readout tokens).
Accuracy here is measured as `1 - edit_distance(guess, word) / len(word)`.
<img width="1280" height="960" alt="image" src="./ocr/run_graphs/run009.png" />

Accuracy seemed to be increasing slightly at the end of the 2-epoch run comparing B and C, so I trained a 4-transformer and 8-transformer variant of C for 5 epochs.
<img width="1280" height="960" alt="image" src="./ocr/run_graphs/run010.png" />

Then followed with a 10-epoch run with 4- and 8-transformer variants of Models B and C.
Model C clearly has better accuracy, with the 4-transformer Model C beating the 8-transformer Model B.
There's also a large gap between C-8 and C4.
<img width="1280" height="960" alt="image" src="./ocr/run_graphs/run011.png" />
For comparison, I ran a C-16 version as well:
<img width="1280" height="960" alt="image" src="./ocr/run_graphs/run012.png" />

Model C has a linear layer that compresses the 128 pixel tokens (4 rows of 32px each) down to the transformer's embedding dimension of 32.
Rather than simply increase the number of transformers, I then did a comparison of variants with a larger embedding dimension.
Four variants of Model C were trained for 5 epochs:
- C-8/32: 8 transformers, embedding dimension 32
- C-16/32: 16 transformers, embedding dimension 32
- C-8/64: 8 transformers, embedding dimension 64
- C-8/128: 8 transformers, embedding dimension 128
Those with a larger embedding dimension clearly outperformed those with a smaller one, but the accuracy difference between 64 and 128 was negligable,
and C-8/128 was by far the slowest to train.
<img width="1280" height="960" alt="image" src="./ocr/run_graphs/run015.png" />

The test accuracy seemed to be falling below the train accuracy for C-8/64 and C-8/128 in the previous test, so a fourth model, Model D, was added,
which had dropout enabled. A 20-epoch run comparing Model C and Model D showed that Model D avoided overfitting, with test accuracy still rising
at the end of the run. Model C's test accuracy stopped rising shortly after epoch 10.
<img width="1280" height="960" alt="image" src="./ocr/run_graphs/run019.png" />

Though results were promising, the lengthy training time was quite inconvenient. I implemented Adam (the previous runs had used SGD) and compared
Models C and D again in a 5-epoch run. By the end of 5 epochs, test accuracy for both models was around 0.6, while in the previous (SGD) run,
accuracy was about 0.4 for Model C and 0.3 for Model D at epoch 5.
<img width="1280" height="960" alt="image" src="./ocr/run_graphs/run020.png" />
