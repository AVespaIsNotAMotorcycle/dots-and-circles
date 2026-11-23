# Diagnosis of ocr/runs/run032/weights_D-12-64_Adam_3e-3.npz

## Robustness

Words read exactly. All but plain scale the crop to 24px wide in the 32px strip, so compare them with centred; the half-pixel shift shows what the sub-pixel blurring of the slants and bends costs alone. A slant drifts the given pixels from top to bottom; a bend weaves them either side, 48 rows per wave.

| set | crops | plain | centred | half pixel | moved 4px | slant ±2 | slant ±4 | slant ±8 | bend ±1 | bend ±2 | bend ±3 | bend ±4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| photo | 3150 | 94.9% | 89.6% | 87.9% | 89.6% | 87.7% | 87.7% | 87.7% | 87.1% | 84.4% | 74.4% | 55.1% |
| cursive_v2 | 191 | 48.2% | 37.2% | 37.7% | 37.2% | 33.0% | 36.1% | 29.8% | 34.0% | 31.4% | 23.6% | 11.0% |
| cursive_v1 | 322 | 35.1% | 26.1% | 25.8% | 26.7% | 26.1% | 27.6% | 27.0% | 24.5% | 23.3% | 19.6% | 14.9% |
| manchuai_218 | 218 | 91.7% | 78.9% | 76.6% | 78.0% | 76.6% | 77.1% | 69.7% | 78.0% | 76.1% | 67.9% | 52.3% |
| manchuai_753 | 753 | 89.6% | 80.2% | 79.4% | 77.7% | 79.2% | 76.9% | 76.1% | 76.5% | 74.1% | 65.1% | 48.6% |

## Letter confusions

### Print

186 letter errors over 18920 letters (∅ on the right: dropped; on the left: inserted).

| true → read | count | share |
|---|---|---|
| a → ∅ | 21 | 11.3% |
| ∅ → n | 19 | 10.2% |
| e → a | 9 | 4.8% |
| n → ∅ | 9 | 4.8% |
| a → e | 8 | 4.3% |
| u → o | 7 | 3.8% |
| c → j | 6 | 3.2% |
| g → k | 5 | 2.7% |
| ∅ → o | 5 | 2.7% |
| g → h | 5 | 2.7% |
| ∅ → k | 4 | 2.2% |
| e → ∅ | 4 | 2.2% |
| v → i | 4 | 2.2% |
| a → n | 3 | 1.6% |
| o → u | 3 | 1.6% |

| letter | seen | misread |
|---|---|---|
| y | 210 | 3.3% |
| w | 210 | 2.4% |
| x | 180 | 1.7% |
| f | 420 | 1.7% |
| g | 670 | 1.6% |
| v | 250 | 1.6% |
| n | 920 | 1.5% |
| c | 440 | 1.4% |
| a | 2950 | 1.3% |
| e | 1890 | 0.8% |

### cursive_v2

184 letter errors over 1004 letters (∅ on the right: dropped; on the left: inserted).

| true → read | count | share |
|---|---|---|
| a → e | 16 | 8.7% |
| n → ∅ | 13 | 7.1% |
| a → ∅ | 11 | 6.0% |
| ∅ → , | 6 | 3.3% |
| g → k | 6 | 3.3% |
| a → i | 5 | 2.7% |
| e → ∅ | 5 | 2.7% |
| e → i | 5 | 2.7% |
| e → a | 4 | 2.2% |
| n → e | 4 | 2.2% |
| d → t | 4 | 2.2% |
| a → n | 3 | 1.6% |
| y → j | 3 | 1.6% |
| ∅ → n | 3 | 1.6% |
| t → d | 3 | 1.6% |

| letter | seen | misread |
|---|---|---|
| g | 45 | 31.1% |
| n | 95 | 25.3% |
| a | 169 | 23.1% |
| t | 23 | 21.7% |
| s | 23 | 21.7% |
| e | 103 | 18.4% |
| m | 24 | 16.7% |
| d | 46 | 15.2% |
| r | 33 | 15.2% |
| c | 28 | 14.3% |

### cursive_v1

480 letter errors over 1500 letters (∅ on the right: dropped; on the left: inserted).

| true → read | count | share |
|---|---|---|
| n → ∅ | 21 | 4.4% |
| a → ∅ | 17 | 3.5% |
| ∅ → e | 13 | 2.7% |
| a → e | 12 | 2.5% |
| ∅ → i | 11 | 2.3% |
| i → a | 10 | 2.1% |
| e → ∅ | 9 | 1.9% |
| e → a | 9 | 1.9% |
| g → k | 9 | 1.9% |
| ∅ → a | 8 | 1.7% |
| y → j | 8 | 1.7% |
| ∅ → k | 7 | 1.5% |
| u → i | 7 | 1.5% |
| ∅ → n | 7 | 1.5% |
| y → ∅ | 6 | 1.2% |

| letter | seen | misread |
|---|---|---|
| v | 26 | 69.2% |
| y | 49 | 63.3% |
| n | 105 | 42.9% |
| w | 27 | 37.0% |
| g | 65 | 36.9% |
| r | 44 | 36.4% |
| o | 62 | 32.3% |
| ng | 53 | 30.2% |
| u | 79 | 29.1% |
| j | 46 | 28.3% |

### manchuai_218

20 letter errors over 1168 letters (∅ on the right: dropped; on the left: inserted).

| true → read | count | share |
|---|---|---|
| e → a | 4 | 20.0% |
| a → ∅ | 3 | 15.0% |
| d → t | 3 | 15.0% |
| w → f | 2 | 10.0% |
| l → g | 1 | 5.0% |
| u → o | 1 | 5.0% |
| b → p | 1 | 5.0% |
| m → n | 1 | 5.0% |
| m → h | 1 | 5.0% |
| n → k | 1 | 5.0% |
| t → d | 1 | 5.0% |
| g → k | 1 | 5.0% |

| letter | seen | misread |
|---|---|---|
| d | 65 | 4.6% |
| e | 94 | 4.3% |
| m | 49 | 4.1% |
| u | 34 | 2.9% |
| l | 39 | 2.6% |
| n | 57 | 1.8% |
| b | 96 | 1.0% |
| a | 291 | 1.0% |
| s | 28 | 0.0% |
| r | 35 | 0.0% |

### manchuai_753

100 letter errors over 4079 letters (∅ on the right: dropped; on the left: inserted).

| true → read | count | share |
|---|---|---|
| w → f | 13 | 13.0% |
| e → a | 8 | 8.0% |
| d → t | 7 | 7.0% |
| a → ∅ | 6 | 6.0% |
| a → e | 4 | 4.0% |
| y → j | 4 | 4.0% |
| b → p | 3 | 3.0% |
| e → ∅ | 3 | 3.0% |
| j → c | 3 | 3.0% |
| ∅ → n | 3 | 3.0% |
| w → j | 3 | 3.0% |
| n → ∅ | 2 | 2.0% |
| ∅ → e | 2 | 2.0% |
| u → o | 2 | 2.0% |
| g → x | 2 | 2.0% |

| letter | seen | misread |
|---|---|---|
| w | 60 | 30.0% |
| y | 71 | 7.0% |
| d | 108 | 6.5% |
| j | 75 | 5.3% |
| b | 155 | 3.2% |
| e | 498 | 2.8% |
| g | 162 | 2.5% |
| k | 96 | 2.1% |
| n | 258 | 1.9% |
| a | 544 | 1.8% |

## Crop shape

Letter error rate by fifths of each set's crops, sorted by height ÷ width ÷ letters before scaling: the widest crops for their length first.

| set | widest | 2 | 3 | 4 | narrowest |
|---|---|---|---|---|---|
| photo | 1.3% | 1.1% | 0.8% | 0.9% | 0.6% |
| cursive_v2 | 38.5% | 19.4% | 12.6% | 7.5% | 10.4% |
| cursive_v1 | 32.4% | 30.3% | 37.5% | 25.2% | 33.9% |
| manchuai_218 | 3.9% | 2.8% | 0.0% | 0.9% | 0.0% |
| manchuai_753 | 4.0% | 2.0% | 1.8% | 2.0% | 1.8% |
