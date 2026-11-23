# Diagnosis of ocr/runs/run034/weights_D-12-64_Adam_3e-3.npz

## Robustness

Words read exactly. All but plain scale the crop to 24px wide in the 32px strip, so compare them with centred; the half-pixel shift shows what the sub-pixel blurring of the slants and bends costs alone. A slant drifts the given pixels from top to bottom; a bend weaves them either side, 48 rows per wave.

| set | crops | plain | centred | half pixel | moved 4px | slant ±2 | slant ±4 | slant ±8 | bend ±1 | bend ±2 | bend ±3 | bend ±4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| photo | 3150 | 94.8% | 90.9% | 89.8% | 90.4% | 89.1% | 88.9% | 88.8% | 88.6% | 85.7% | 75.4% | 55.0% |
| cursive_v2 | 191 | 52.4% | 44.0% | 40.8% | 41.4% | 40.8% | 38.2% | 36.1% | 40.3% | 36.1% | 27.2% | 17.3% |
| cursive_v1 | 322 | 36.0% | 30.4% | 30.1% | 29.5% | 32.0% | 31.1% | 28.6% | 30.4% | 27.0% | 21.4% | 17.7% |
| manchuai_218 | 218 | 90.8% | 84.4% | 83.9% | 83.5% | 83.9% | 83.9% | 75.7% | 80.7% | 74.8% | 63.8% | 47.2% |
| manchuai_753 | 753 | 90.6% | 82.2% | 80.2% | 80.3% | 81.1% | 79.9% | 78.9% | 79.7% | 75.0% | 64.4% | 46.1% |

## Letter confusions

### Print

189 letter errors over 18920 letters (∅ on the right: dropped; on the left: inserted).

| true → read | count | share |
|---|---|---|
| ∅ → n | 19 | 10.1% |
| e → a | 15 | 7.9% |
| a → ∅ | 14 | 7.4% |
| a → e | 14 | 7.4% |
| n → ∅ | 12 | 6.3% |
| g → h | 8 | 4.2% |
| m → h | 7 | 3.7% |
| t → d | 5 | 2.6% |
| n → e | 5 | 2.6% |
| u → o | 5 | 2.6% |
| g → k | 4 | 2.1% |
| o → u | 4 | 2.1% |
| a → n | 3 | 1.6% |
| i → o | 3 | 1.6% |
| e → u | 3 | 1.6% |

| letter | seen | misread |
|---|---|---|
| y | 210 | 2.9% |
| n | 920 | 2.4% |
| g | 670 | 1.8% |
| x | 180 | 1.7% |
| t | 300 | 1.7% |
| w | 210 | 1.4% |
| e | 1890 | 1.3% |
| v | 250 | 1.2% |
| a | 2950 | 1.2% |
| j | 370 | 1.1% |

### cursive_v2

168 letter errors over 1004 letters (∅ on the right: dropped; on the left: inserted).

| true → read | count | share |
|---|---|---|
| a → e | 15 | 8.9% |
| n → ∅ | 13 | 7.7% |
| ∅ → n | 8 | 4.8% |
| a → ∅ | 8 | 4.8% |
| n → e | 6 | 3.6% |
| e → i | 5 | 3.0% |
| e → ∅ | 4 | 2.4% |
| ∅ → , | 4 | 2.4% |
| a → n | 3 | 1.8% |
| i → e | 3 | 1.8% |
| a → u | 3 | 1.8% |
| e → a | 3 | 1.8% |
| ∅ → e | 3 | 1.8% |
| d → t | 3 | 1.8% |
| s → l | 2 | 1.2% |

| letter | seen | misread |
|---|---|---|
| n | 95 | 26.3% |
| a | 169 | 23.7% |
| t | 23 | 21.7% |
| d | 46 | 19.6% |
| j | 30 | 16.7% |
| e | 103 | 14.6% |
| c | 28 | 14.3% |
| g | 45 | 13.3% |
| s | 23 | 13.0% |
| r | 33 | 9.1% |

### cursive_v1

424 letter errors over 1500 letters (∅ on the right: dropped; on the left: inserted).

| true → read | count | share |
|---|---|---|
| n → ∅ | 20 | 4.7% |
| a → e | 13 | 3.1% |
| ∅ → a | 12 | 2.8% |
| ∅ → n | 11 | 2.6% |
| g → k | 9 | 2.1% |
| i → a | 9 | 2.1% |
| o → u | 8 | 1.9% |
| ∅ → i | 8 | 1.9% |
| i → ∅ | 8 | 1.9% |
| a → n | 7 | 1.7% |
| ∅ → e | 7 | 1.7% |
| y → j | 6 | 1.4% |
| ∅ → g | 6 | 1.4% |
| a → ∅ | 6 | 1.4% |
| u → o | 6 | 1.4% |

| letter | seen | misread |
|---|---|---|
| v | 26 | 73.1% |
| y | 49 | 49.0% |
| n | 105 | 42.9% |
| w | 27 | 40.7% |
| o | 62 | 40.3% |
| c | 24 | 29.2% |
| g | 65 | 27.7% |
| u | 79 | 25.3% |
| j | 46 | 21.7% |
| r | 44 | 20.5% |

### manchuai_218

26 letter errors over 1168 letters (∅ on the right: dropped; on the left: inserted).

| true → read | count | share |
|---|---|---|
| e → a | 6 | 23.1% |
| d → t | 2 | 7.7% |
| m → h | 2 | 7.7% |
| w → f | 2 | 7.7% |
| u → o | 1 | 3.8% |
| ∅ → n | 1 | 3.8% |
| b → p | 1 | 3.8% |
| m → n | 1 | 3.8% |
| a → ∅ | 1 | 3.8% |
| n → k | 1 | 3.8% |
| ∅ → l | 1 | 3.8% |
| i → e | 1 | 3.8% |
| n → ∅ | 1 | 3.8% |
| d → ∅ | 1 | 3.8% |
| t → d | 1 | 3.8% |

| letter | seen | misread |
|---|---|---|
| e | 94 | 7.4% |
| m | 49 | 6.1% |
| n | 57 | 5.3% |
| d | 65 | 4.6% |
| u | 34 | 2.9% |
| h | 54 | 1.9% |
| b | 96 | 1.0% |
| i | 135 | 0.7% |
| a | 291 | 0.3% |
| s | 28 | 0.0% |

### manchuai_753

87 letter errors over 4079 letters (∅ on the right: dropped; on the left: inserted).

| true → read | count | share |
|---|---|---|
| w → f | 14 | 16.1% |
| e → a | 7 | 8.0% |
| a → ∅ | 6 | 6.9% |
| a → e | 4 | 4.6% |
| ∅ → n | 4 | 4.6% |
| n → ∅ | 3 | 3.4% |
| d → t | 3 | 3.4% |
| e → ∅ | 3 | 3.4% |
| i → ∅ | 2 | 2.3% |
| ∅ → m | 2 | 2.3% |
| y → j | 2 | 2.3% |
| u → o | 2 | 2.3% |
| dz → ts' | 2 | 2.3% |
| g → ∅ | 2 | 2.3% |
| n → x | 2 | 2.3% |

| letter | seen | misread |
|---|---|---|
| w | 60 | 28.3% |
| y | 71 | 4.2% |
| t | 81 | 3.7% |
| k | 96 | 3.1% |
| d | 108 | 2.8% |
| g | 162 | 2.5% |
| e | 498 | 2.4% |
| n | 258 | 1.9% |
| a | 544 | 1.8% |
| o | 140 | 1.4% |

## Crop shape

Letter error rate by fifths of each set's crops, sorted by height ÷ width ÷ letters before scaling: the widest crops for their length first.

| set | widest | 2 | 3 | 4 | narrowest |
|---|---|---|---|---|---|
| photo | 1.5% | 1.1% | 0.7% | 0.8% | 0.6% |
| cursive_v2 | 32.1% | 17.1% | 13.5% | 9.4% | 8.3% |
| cursive_v1 | 27.8% | 30.6% | 33.9% | 16.0% | 31.6% |
| manchuai_218 | 4.9% | 2.8% | 0.0% | 2.1% | 0.0% |
| manchuai_753 | 3.6% | 1.4% | 1.1% | 1.7% | 2.5% |
