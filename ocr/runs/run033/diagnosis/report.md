# Diagnosis of ocr/runs/run033/weights_D-12-64_Adam_3e-3.npz

## Robustness

Words read exactly. All but plain scale the crop to 24px wide in the 32px strip, so compare them with centred; the half-pixel shift shows what the sub-pixel blurring of the slants and bends costs alone. A slant drifts the given pixels from top to bottom; a bend weaves them either side, 48 rows per wave.

| set | crops | plain | centred | half pixel | moved 4px | slant ±2 | slant ±4 | slant ±8 | bend ±1 | bend ±2 | bend ±3 | bend ±4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| photo | 3150 | 94.8% | 91.4% | 89.9% | 91.2% | 90.1% | 90.1% | 89.9% | 89.5% | 88.4% | 86.1% | 81.1% |
| cursive_v2 | 191 | 68.1% | 52.4% | 52.9% | 56.0% | 55.5% | 53.9% | 51.8% | 53.4% | 54.5% | 49.2% | 44.0% |
| cursive_v1 | 322 | 50.9% | 44.1% | 41.3% | 41.0% | 41.0% | 42.5% | 38.5% | 41.3% | 39.4% | 35.4% | 28.6% |
| manchuai_218 | 218 | 94.5% | 84.4% | 83.9% | 81.2% | 83.0% | 82.6% | 81.2% | 83.9% | 83.5% | 78.0% | 72.0% |
| manchuai_753 | 753 | 91.0% | 84.3% | 83.4% | 82.1% | 82.5% | 82.5% | 81.4% | 81.1% | 80.2% | 78.5% | 72.6% |

## Letter confusions

### Print

189 letter errors over 18920 letters (∅ on the right: dropped; on the left: inserted).

| true → read | count | share |
|---|---|---|
| ∅ → n | 18 | 9.5% |
| a → e | 14 | 7.4% |
| a → ∅ | 11 | 5.8% |
| n → ∅ | 9 | 4.8% |
| g → h | 6 | 3.2% |
| e → a | 6 | 3.2% |
| u → o | 6 | 3.2% |
| d → t | 5 | 2.6% |
| t → d | 5 | 2.6% |
| ∅ → g | 4 | 2.1% |
| l → h | 3 | 1.6% |
| l → m | 3 | 1.6% |
| r → y | 3 | 1.6% |
| g → k | 3 | 1.6% |
| x → s | 3 | 1.6% |

| letter | seen | misread |
|---|---|---|
| x | 180 | 4.4% |
| y | 210 | 2.4% |
| j | 370 | 1.9% |
| t | 300 | 1.7% |
| n | 920 | 1.6% |
| w | 210 | 1.4% |
| l | 590 | 1.4% |
| g | 670 | 1.3% |
| a | 2950 | 1.1% |
| f | 420 | 1.0% |

### cursive_v2

109 letter errors over 1004 letters (∅ on the right: dropped; on the left: inserted).

| true → read | count | share |
|---|---|---|
| n → ∅ | 11 | 10.1% |
| a → ∅ | 9 | 8.3% |
| a → e | 8 | 7.3% |
| ∅ → n | 4 | 3.7% |
| ∅ → e | 4 | 3.7% |
| x → s | 3 | 2.8% |
| e → i | 3 | 2.8% |
| i → a | 3 | 2.8% |
| n → e | 3 | 2.8% |
| j → w | 3 | 2.8% |
| a → n | 2 | 1.8% |
| i → e | 2 | 1.8% |
| i → u | 2 | 1.8% |
| ∅ → , | 2 | 1.8% |
| t → d | 2 | 1.8% |

| letter | seen | misread |
|---|---|---|
| j | 30 | 20.0% |
| n | 95 | 17.9% |
| t | 23 | 17.4% |
| a | 169 | 14.8% |
| d | 46 | 13.0% |
| e | 103 | 8.7% |
| h | 37 | 8.1% |
| i | 122 | 7.4% |
| g | 45 | 6.7% |
| r | 33 | 6.1% |

### cursive_v1

320 letter errors over 1500 letters (∅ on the right: dropped; on the left: inserted).

| true → read | count | share |
|---|---|---|
| n → ∅ | 16 | 5.0% |
| a → e | 13 | 4.1% |
| o → u | 8 | 2.5% |
| a → ∅ | 7 | 2.2% |
| i → ∅ | 7 | 2.2% |
| j → ∅ | 7 | 2.2% |
| ∅ → t | 6 | 1.9% |
| ∅ → e | 6 | 1.9% |
| ng → ∅ | 6 | 1.9% |
| e → a | 6 | 1.9% |
| y → i | 5 | 1.6% |
| i → a | 5 | 1.6% |
| i → e | 5 | 1.6% |
| r → ∅ | 4 | 1.2% |
| ∅ → i | 4 | 1.2% |

| letter | seen | misread |
|---|---|---|
| v | 26 | 50.0% |
| w | 27 | 33.3% |
| o | 62 | 30.6% |
| j | 46 | 30.4% |
| g | 65 | 29.2% |
| y | 49 | 28.6% |
| n | 105 | 28.6% |
| r | 44 | 22.7% |
| c | 24 | 20.8% |
| ng | 53 | 20.8% |

### manchuai_218

14 letter errors over 1168 letters (∅ on the right: dropped; on the left: inserted).

| true → read | count | share |
|---|---|---|
| d → t | 2 | 14.3% |
| e → a | 2 | 14.3% |
| w → f | 2 | 14.3% |
| u → o | 1 | 7.1% |
| m → n | 1 | 7.1% |
| m → h | 1 | 7.1% |
| n → k | 1 | 7.1% |
| a → e | 1 | 7.1% |
| a → ∅ | 1 | 7.1% |
| ∅ → n | 1 | 7.1% |
| g → h | 1 | 7.1% |

| letter | seen | misread |
|---|---|---|
| m | 49 | 4.1% |
| d | 65 | 3.1% |
| u | 34 | 2.9% |
| e | 94 | 2.1% |
| n | 57 | 1.8% |
| a | 291 | 0.7% |
| s | 28 | 0.0% |
| r | 35 | 0.0% |
| o | 35 | 0.0% |
| l | 39 | 0.0% |

### manchuai_753

77 letter errors over 4079 letters (∅ on the right: dropped; on the left: inserted).

| true → read | count | share |
|---|---|---|
| w → f | 14 | 18.2% |
| a → ∅ | 6 | 7.8% |
| d → t | 6 | 7.8% |
| n → ∅ | 5 | 6.5% |
| a → e | 4 | 5.2% |
| e → a | 4 | 5.2% |
| j → c | 3 | 3.9% |
| k → ∅ | 3 | 3.9% |
| i → e | 2 | 2.6% |
| o → u | 2 | 2.6% |
| e → ∅ | 2 | 2.6% |
| ∅ → n | 2 | 2.6% |
| w → j | 2 | 2.6% |
| i → ∅ | 1 | 1.3% |
| ∅ → m | 1 | 1.3% |

| letter | seen | misread |
|---|---|---|
| w | 60 | 26.7% |
| d | 108 | 6.5% |
| j | 75 | 5.3% |
| k | 96 | 3.1% |
| n | 258 | 2.7% |
| a | 544 | 1.8% |
| o | 140 | 1.4% |
| y | 71 | 1.4% |
| e | 498 | 1.4% |
| t | 81 | 1.2% |

## Crop shape

Letter error rate by fifths of each set's crops, sorted by height ÷ width ÷ letters before scaling: the widest crops for their length first.

| set | widest | 2 | 3 | 4 | narrowest |
|---|---|---|---|---|---|
| photo | 1.3% | 0.9% | 0.7% | 1.1% | 0.9% |
| cursive_v2 | 24.0% | 11.4% | 9.8% | 2.3% | 4.2% |
| cursive_v1 | 22.3% | 20.2% | 26.7% | 15.1% | 20.1% |
| manchuai_218 | 2.1% | 2.0% | 0.0% | 1.3% | 0.0% |
| manchuai_753 | 2.5% | 1.7% | 1.4% | 1.3% | 2.5% |
