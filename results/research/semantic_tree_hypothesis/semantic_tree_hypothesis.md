# Single semantic hierarchy tree hypothesis test

Labels: Amazon Reviews 2023 `meta_Musical_Instruments` category paths, matched to
the local items by normalised title. Coarse = category level 2; fine = full
category path (the leaf name alone can repeat under different coarse branches).
Ambiguous and missing titles were dropped; this is an independent semantic hierarchy,
not KMeans on the embedding.

Labeled items: **23697 / 24,587**; coarse categories: **16**; fine categories: **567**.

Ideal tree distance: same fine = 1, same coarse different fine = 2, cross coarse = 3.

## New neighbours by K band (text kNN)

| ranks | same fine | same coarse, diff fine | cross coarse |
|---|---:|---:|---:|
| 1-5 | 0.6333 | 0.2558 | 0.1109 |
| 6-10 | 0.5649 | 0.2979 | 0.1373 |
| 11-15 | 0.5245 | 0.3190 | 0.1565 |
| 16-20 | 0.4983 | 0.3321 | 0.1696 |

## New neighbours by K band (ideal tree)

| ranks | same fine | same coarse, diff fine | cross coarse |
|---|---:|---:|---:|
| 1-5 | 0.9883 | 0.0117 | 0.0001 |
| 6-10 | 0.9651 | 0.0348 | 0.0001 |
| 11-15 | 0.9437 | 0.0561 | 0.0001 |
| 16-20 | 0.9249 | 0.0749 | 0.0001 |

## Hierarchy violations

| graph | cross before same coarse | diff-fine before same-fine | first cross n | first cross median | first cross <=5 |
|---|---:|---:|---:|---:|---:|
| text | 0.1611 | 0.2690 | 11804.0 | 6.0 | 0.4818 |
| ideal_tree | 0.0000 | 0.0000 | 3.0 | 3.0 | 1.0000 |

## Graphlets

| graph | K1,3 | P4 | paw | diamond | K4 | C4 |
|---|---:|---:|---:|---:|---:|---:|
| text k=5 | 0.2738 | 0.5198 | 0.1693 | 0.0257 | 0.0049 | 0.0065 |
| text k=10 | 0.2543 | 0.5107 | 0.1887 | 0.0320 | 0.0073 | 0.0069 |
| text k=15 | 0.2490 | 0.5049 | 0.1958 | 0.0348 | 0.0084 | 0.0071 |
| text k=20 | 0.2448 | 0.5026 | 0.1999 | 0.0364 | 0.0090 | 0.0073 |
| ideal_tree k=5 | 0.2277 | 0.6987 | 0.0514 | 0.0078 | 0.0011 | 0.0133 |
| ideal_tree k=10 | 0.2155 | 0.6633 | 0.0844 | 0.0136 | 0.0023 | 0.0210 |
| ideal_tree k=15 | 0.2081 | 0.6377 | 0.1049 | 0.0200 | 0.0040 | 0.0253 |
| ideal_tree k=20 | 0.2017 | 0.6180 | 0.1207 | 0.0252 | 0.0060 | 0.0284 |

## Cross-coarse top-5 pairs

Total cross-coarse pairs in text top-5: **13142**; sampled **100** for manual audit in `cross_coarse_top5_pairs.csv`.

| category pair | pairs |
|---|---:|
| Amplifiers & Effects <> Instrument Accessories | 1783 |
| Drums & Percussion <> Instrument Accessories | 1370 |
| Guitars <> Instrument Accessories | 1269 |
| Electronic Music, DJ & Karaoke <> Studio Recording Equipment | 853 |
| Live Sound & Stage <> Studio Recording Equipment | 843 |
| Instrument Accessories <> Live Sound & Stage | 646 |
| Electronic Music, DJ & Karaoke <> Live Sound & Stage | 543 |
| Microphones & Accessories <> Studio Recording Equipment | 515 |
| Instrument Accessories <> Microphones & Accessories | 503 |
| Instrument Accessories <> Studio Recording Equipment | 390 |
| Instrument Accessories <> Ukuleles, Mandolins & Banjos | 380 |
| Amplifiers & Effects <> Studio Recording Equipment | 361 |
| Live Sound & Stage <> Microphones & Accessories | 345 |
| Band & Orchestra <> Instrument Accessories | 306 |
| Drum & Percussion Accessories <> Instrument Accessories | 284 |

## Ultrametric / four-point residuals

| test | data | mean | median | p95 | >0.05 | >0.10 |
|---|---|---:|---:|---:|---:|---:|
| four-point | text_euclidean | 0.0291 | 0.0244 | 0.0721 | 0.1742 | 0.0064 |
| four-point | ideal_tree | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ultrametric | text_euclidean | 0.0605 | 0.0508 | 0.1494 | 0.5064 | 0.1906 |
| ultrametric | ideal_tree | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Residuals are relative: (largest two-point sum minus second largest) / largest.
The ideal tree is exact by construction; sampling gives 0.
