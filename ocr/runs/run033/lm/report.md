# Language model decoding: ocr/runs/run033/weights_D-12-64_Adam_3e-3.npz

## Tuning on the monitor sets

Words exact on photo_monitor / cursive_monitor. Greedy decoding: 95.4% / 76.0%. Best: weight 0.6, bonus 1.

| weight \ bonus | 0 | 0.5 | 1 | 1.5 | 2 | 3 |
|---|---|---|---|---|---|---|
| 0 | 94.8% / 77.0% |  |  |  |  |  |
| 0.1 | 95.8% / 77.0% | 95.8% / 77.0% | 95.2% / 76.0% | 95.0% / 76.0% | 93.8% / 75.0% | 92.4% / 73.0% |
| 0.2 | 95.8% / 79.0% | 96.0% / 78.0% | 96.0% / 77.0% | 96.0% / 78.0% | 95.2% / 78.0% | 94.0% / 76.0% |
| 0.3 | 96.4% / 79.0% | 96.6% / 78.0% | 96.6% / 78.0% | 96.8% / 78.0% | 96.8% / 78.0% | 95.2% / 78.0% |
| 0.4 | 96.6% / 79.0% | 96.6% / 79.0% | 97.0% / 79.0% | 97.0% / 80.0% | 97.0% / 79.0% | 96.6% / 79.0% |
| 0.6 | 97.0% / 80.0% | 97.0% / 80.0% | 97.6% / 80.0% | 97.6% / 80.0% | 97.6% / 80.0% | 97.6% / 80.0% |
| 0.8 | 96.6% / 79.0% | 96.8% / 80.0% | 97.0% / 80.0% | 97.2% / 80.0% | 97.4% / 80.0% | 97.4% / 80.0% |
| 1.0 | 96.8% / 78.0% | 97.0% / 78.0% | 97.2% / 78.0% | 97.6% / 79.0% | 97.6% / 79.0% | 97.6% / 79.0% |

## Benchmarks

Words exact (letters right). Beam alone is weight 0; with the language model, weight 0.6 and bonus 1. Flips count words the language model turns right → wrong and wrong → right, against the beam alone. *LM never saw* is the words in no training entry of the language model, nor inside one (the benchmark words are never entries, but many are inside Norman's phrases; cursive_v1's and ManchuAI's are often entries): it mustn't help only words it knows.

| set | words | greedy | beam alone | with LM | right → wrong | wrong → right | LM never saw: words, greedy → with LM |
|---|---|---|---|---|---|---|---|
| photo | 3150 | 94.8% (99.0%) | 94.9% (99.0%) | 96.8% (99.4%) | 11 | 70 | 1670, 95% → 97% |
| cursive_v2 | 191 | 68.1% (89.2%) | 68.6% (89.7%) | 76.4% (92.2%) | 1 | 16 | 91, 65% → 73% |
| cursive_v1 | 322 | 50.6% (78.5%) | 50.9% (78.8%) | 61.5% (83.0%) | 3 | 37 | 71, 34% → 44% |
| manchuai_218 | 218 | 94.5% (98.8%) | 94.5% (98.8%) | 95.9% (99.2%) | 0 | 3 | 76, 92% → 93% |
| manchuai_753 | 753 | 91.0% (98.1%) | 91.1% (98.1%) | 93.1% (98.5%) | 3 | 18 | 251, 85% → 88% |

## Examples

true: beam alone → with the language model

### photo

Made right: acaburengge: eacaburengge → acaburengge, aliyame: amiyame → aliyame, amaga: amga → amaga, amaka: amka → amaka, amaka: amka → amaka, ambula: ambua → ambula, arahabi: arhabi → arahabi, baitalambi: baitalembi → baitalambi, baitalame: bayitalame → baitalame, baitalarangge: baitalayangge → baitalarangge, baitalarangge: baitalarage → baitalarangge, beyebe: benyebe → beyebe, daifasa: daiwasa → daifasa, donjici: donjoci → donjici, donjici: tonjici → donjici

Made wrong: antaka: antaka → antakan, antaka: antaka → antakan, babe: babe → bebe, boigon: boigon → boihon, dembei: dembei → dembi, ergici: ergici → ergeci, ergici: ergici → ergeci, g'ai: g'ai → gai, kafi: kafi → hafi, kiyan: kiyan → giyan, u: u → o

### cursive_v2

Made right: amala: amla → amala, ambula: amula → ambula, aniya: aniyn → aniya, elerakv: elerakv, → elerakv, ganaha: ganaha, → ganaha, gisun: gesun → gisun, hokorakv: hokorakv, → hokorakv, jaluka: jeluka → jaluka, maktafi: maaktafi → maktafi, nadan: nadae → nadan, nadan: nada → nadan, nadan: nadun → nadan, nadan: adan → nadan, tatarakv: tatarakv. → tatarakv, tuwanjimbiheo: tuwanimbiheo → tuwanjimbiheo

Made wrong: jobome: jobome → forome

### cursive_v1

Made right: alixame: aliame → alixame, amban: aman → amban, araha: araha. → araha, babe: babeh → babe, baturu: beturu → baturu, beye: bece → beye, biyai: biyi → biyai, bodohonggo: boduhonggo → bodohonggo, cooha: cotaha → cooha, cuwan: cuyan → cuwan, dahaha: daehaha → dahaha, dailame: deilame → dailame, dasataha: dasateh → dasataha, de: dde → de, deng: deng  → deng

Made wrong: babe: babe → bade, ilan: ilan → alan, ni: ni → ini

### manchuai_218

Made right: adabufi: adabofi → adabufi, dabala: dabla → dabala, medege: mendehe → medege

Made wrong: none

### manchuai_753

Made right: ashan: asahan → ashan, banjinahangge: banjinhangge → banjinahangge, banjinjifi: bancinjifi → banjinjifi, bodome: botome → bodome, deo: deu → deo, ejen: ecen → ejen, gidafi: gidefi → gidafi, hengkile: hengkele → hengkile, inenggi: inanggi → inenggi, jakade: akade → jakade, jirgalang: jirgang → jirgalang, kiceme: kicame → kiceme, kunduleme: kuntuleme → kunduleme, latunahakv: latunkahakv → latunahakv, niyengniyeri: nijengiyeri → niyengniyeri

Made wrong: dureng: dureng → dereng, ede: ede → eden, hesa: hesa → hese
