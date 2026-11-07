# Issue #139: the five arms

The reference row is the frozen resample baseline (hard ER 16.39, positive hit 0.0794, 16145 second-level prefixes).

| arm | epochs | wall s | collision | L1 hard ER | L1 pos hit | L1 neg hit | L2 prefixes | L3 triples | L3 collision | valid SID | rounds (accepted) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F0 | 3000 | 546 | 0.6307 | 3.691 | 0.1571 | 0.0426 | 2096 | 9081 | 0.7672 | False | - |
| G | 3000 | 729 | 0.5809 | 3.786 | 0.1472 | 0.0389 | 2160 | 10304 | 0.7321 | False | 1500@0:1, 2000@1:0, 2500@2:0, 3000@3:0 |
| L | 310 | - | - | - | - | - | - | - | - | None | - |
| E | - | - | - | - | - | - | - | - | - | None | - |
| A | - | - | - | - | - | - | - | - | - | None | - |
