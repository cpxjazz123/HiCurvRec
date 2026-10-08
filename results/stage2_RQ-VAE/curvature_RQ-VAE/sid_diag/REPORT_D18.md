# D18 — Hyperbolic angular clustering, entailment cones, hierarchy, curvature

Frozen Stage2 checkpoint, 24587 items, curvature 1.0, device cuda:0, 40000 behaviour pairs. No training, no loss change, no Stage2 or Stage3 run is part of this analysis.

## D18-A behaviour neighbours in angle

| representation | AUC positive closer than matched | AUC vs in-sample partners | positive axis angle (deg) | matched (deg) | random (deg) | in-sample partner (deg) | partner-to-partner (deg) | positive concentration | matched concentration | 20deg bootstrap coverage (mean ± SD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Encoder | 0.1610 | 0.5315 | 28.366 | 21.640 | 21.099 | 28.553 | 34.408 | 0.8287 | 0.8830 | 0.0371 ± 0.0003 |
| L1-L1 | 0.2630 | 0.5931 | 32.953 | 25.724 | 25.324 | 34.785 | 38.786 | 0.7272 | 0.8439 | 0.0489 ± 0.0002 |
| L1-L2 | 0.2123 | 0.5581 | 31.298 | 24.037 | 23.607 | 32.357 | 37.504 | 0.7909 | 0.8624 | 0.0258 ± 0.0002 |
| L1-L3 | 0.1991 | 0.5414 | 30.069 | 23.177 | 22.662 | 30.788 | 36.004 | 0.8086 | 0.8706 | 0.0307 ± 0.0002 |

## D18-B entailment cones

| representation | cone | aperture (deg, median) | positive coverage | matched FP | random rate | coverage / FP |
| --- | --- | --- | --- | --- | --- | --- |
| Encoder | angle_2deg|opens_away_from_root | 2.00 | 0.0000 | 0.0000 | 0.0000 | inf |
| Encoder | angle_2deg|opens_toward_root | 2.00 | 0.0000 | 0.0000 | 0.0000 | inf |
| Encoder | angle_5deg|opens_away_from_root | 5.00 | 0.0000 | 0.0000 | 0.0000 | inf |
| Encoder | angle_5deg|opens_toward_root | 5.00 | 0.0000 | 0.0000 | 0.0000 | inf |
| Encoder | angle_10deg|opens_away_from_root | 10.00 | 0.0000 | 0.0001 | 0.0003 | 0.00 |
| Encoder | angle_10deg|opens_toward_root | 10.00 | 0.0000 | 0.0000 | 0.0000 | inf |
| Encoder | angle_20deg|opens_away_from_root | 20.00 | 0.0372 | 0.3289 | 0.3890 | 0.11 |
| Encoder | angle_20deg|opens_toward_root | 20.00 | 0.0000 | 0.0000 | 0.0000 | inf |
| Encoder | angle_30deg|opens_away_from_root | 30.00 | 0.6054 | 0.9447 | 0.9571 | 0.64 |
| Encoder | angle_30deg|opens_toward_root | 30.00 | 0.0000 | 0.0000 | 0.0000 | inf |
| Encoder | angle_45deg|opens_away_from_root | 45.00 | 0.9792 | 0.9995 | 0.9995 | 0.98 |
| Encoder | angle_45deg|opens_toward_root | 45.00 | 0.0000 | 0.0000 | 0.0000 | inf |
| Encoder | metric_reach_0.02|opens_away_from_root | 0.03 | 0.0000 | 0.0000 | 0.0000 | inf |
| Encoder | metric_reach_0.02|opens_toward_root | 0.03 | 0.0000 | 0.0000 | 0.0000 | inf |
| Encoder | metric_reach_0.05|opens_away_from_root | 0.08 | 0.0000 | 0.0000 | 0.0000 | inf |
| Encoder | metric_reach_0.05|opens_toward_root | 0.08 | 0.0000 | 0.0000 | 0.0000 | inf |
| Encoder | metric_reach_0.1|opens_away_from_root | 0.16 | 0.0000 | 0.0000 | 0.0000 | inf |
| Encoder | metric_reach_0.1|opens_toward_root | 0.16 | 0.0000 | 0.0000 | 0.0000 | inf |
| Encoder | metric_reach_0.2|opens_away_from_root | 0.32 | 0.0000 | 0.0000 | 0.0000 | inf |
| Encoder | metric_reach_0.2|opens_toward_root | 0.32 | 0.0000 | 0.0000 | 0.0000 | inf |
| Encoder | metric_reach_0.4|opens_away_from_root | 0.64 | 0.0000 | 0.0000 | 0.0000 | inf |
| Encoder | metric_reach_0.4|opens_toward_root | 0.64 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L1 | angle_2deg|opens_away_from_root | 2.00 | 0.0002 | 0.0009 | 0.0011 | 0.19 |
| L1-L1 | angle_2deg|opens_toward_root | 2.00 | 0.0159 | 0.0000 | 0.0001 | inf |
| L1-L1 | angle_5deg|opens_away_from_root | 5.00 | 0.0057 | 0.0020 | 0.0022 | 2.88 |
| L1-L1 | angle_5deg|opens_toward_root | 5.00 | 0.0268 | 0.0000 | 0.0001 | 1072.00 |
| L1-L1 | angle_10deg|opens_away_from_root | 10.00 | 0.0073 | 0.0021 | 0.0024 | 3.51 |
| L1-L1 | angle_10deg|opens_toward_root | 10.00 | 0.0268 | 0.0000 | 0.0001 | 1074.00 |
| L1-L1 | angle_20deg|opens_away_from_root | 20.00 | 0.0219 | 0.1085 | 0.1373 | 0.20 |
| L1-L1 | angle_20deg|opens_toward_root | 20.00 | 0.0268 | 0.0046 | 0.0056 | 5.87 |
| L1-L1 | angle_30deg|opens_away_from_root | 30.00 | 0.3472 | 0.7076 | 0.7259 | 0.49 |
| L1-L1 | angle_30deg|opens_toward_root | 30.00 | 0.0268 | 0.0214 | 0.0224 | 1.26 |
| L1-L1 | angle_45deg|opens_away_from_root | 45.00 | 0.8272 | 0.9520 | 0.9551 | 0.87 |
| L1-L1 | angle_45deg|opens_toward_root | 45.00 | 0.0268 | 0.0264 | 0.0267 | 1.02 |
| L1-L1 | metric_reach_0.02|opens_away_from_root | 0.03 | 0.0000 | 0.0004 | 0.0004 | 0.00 |
| L1-L1 | metric_reach_0.02|opens_toward_root | 0.03 | 0.0082 | 0.0000 | 0.0000 | inf |
| L1-L1 | metric_reach_0.05|opens_away_from_root | 0.08 | 0.0000 | 0.0005 | 0.0004 | 0.00 |
| L1-L1 | metric_reach_0.05|opens_toward_root | 0.08 | 0.0082 | 0.0000 | 0.0000 | inf |
| L1-L1 | metric_reach_0.1|opens_away_from_root | 0.16 | 0.0000 | 0.0005 | 0.0004 | 0.00 |
| L1-L1 | metric_reach_0.1|opens_toward_root | 0.16 | 0.0083 | 0.0000 | 0.0000 | inf |
| L1-L1 | metric_reach_0.2|opens_away_from_root | 0.32 | 0.0000 | 0.0005 | 0.0004 | 0.00 |
| L1-L1 | metric_reach_0.2|opens_toward_root | 0.32 | 0.0083 | 0.0000 | 0.0000 | inf |
| L1-L1 | metric_reach_0.4|opens_away_from_root | 0.64 | 0.0000 | 0.0005 | 0.0005 | 0.00 |
| L1-L1 | metric_reach_0.4|opens_toward_root | 0.64 | 0.0088 | 0.0000 | 0.0000 | inf |
| L1-L2 | angle_2deg|opens_away_from_root | 2.00 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L2 | angle_2deg|opens_toward_root | 2.00 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L2 | angle_5deg|opens_away_from_root | 5.00 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L2 | angle_5deg|opens_toward_root | 5.00 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L2 | angle_10deg|opens_away_from_root | 10.00 | 0.0000 | 0.0001 | 0.0003 | 0.00 |
| L1-L2 | angle_10deg|opens_toward_root | 10.00 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L2 | angle_20deg|opens_away_from_root | 20.00 | 0.0258 | 0.1865 | 0.2213 | 0.14 |
| L1-L2 | angle_20deg|opens_toward_root | 20.00 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L2 | angle_30deg|opens_away_from_root | 30.00 | 0.4348 | 0.8234 | 0.8423 | 0.53 |
| L1-L2 | angle_30deg|opens_toward_root | 30.00 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L2 | angle_45deg|opens_away_from_root | 45.00 | 0.8952 | 0.9911 | 0.9927 | 0.90 |
| L1-L2 | angle_45deg|opens_toward_root | 45.00 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L2 | metric_reach_0.02|opens_away_from_root | 0.03 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L2 | metric_reach_0.02|opens_toward_root | 0.03 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L2 | metric_reach_0.05|opens_away_from_root | 0.08 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L2 | metric_reach_0.05|opens_toward_root | 0.08 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L2 | metric_reach_0.1|opens_away_from_root | 0.16 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L2 | metric_reach_0.1|opens_toward_root | 0.16 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L2 | metric_reach_0.2|opens_away_from_root | 0.32 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L2 | metric_reach_0.2|opens_toward_root | 0.32 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L2 | metric_reach_0.4|opens_away_from_root | 0.64 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L2 | metric_reach_0.4|opens_toward_root | 0.64 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L3 | angle_2deg|opens_away_from_root | 2.00 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L3 | angle_2deg|opens_toward_root | 2.00 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L3 | angle_5deg|opens_away_from_root | 5.00 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L3 | angle_5deg|opens_toward_root | 5.00 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L3 | angle_10deg|opens_away_from_root | 10.00 | 0.0000 | 0.0001 | 0.0003 | 0.00 |
| L1-L3 | angle_10deg|opens_toward_root | 10.00 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L3 | angle_20deg|opens_away_from_root | 20.00 | 0.0306 | 0.2291 | 0.2712 | 0.13 |
| L1-L3 | angle_20deg|opens_toward_root | 20.00 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L3 | angle_30deg|opens_away_from_root | 30.00 | 0.4958 | 0.8728 | 0.8898 | 0.57 |
| L1-L3 | angle_30deg|opens_toward_root | 30.00 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L3 | angle_45deg|opens_away_from_root | 45.00 | 0.9389 | 0.9960 | 0.9971 | 0.94 |
| L1-L3 | angle_45deg|opens_toward_root | 45.00 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L3 | metric_reach_0.02|opens_away_from_root | 0.03 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L3 | metric_reach_0.02|opens_toward_root | 0.03 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L3 | metric_reach_0.05|opens_away_from_root | 0.08 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L3 | metric_reach_0.05|opens_toward_root | 0.08 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L3 | metric_reach_0.1|opens_away_from_root | 0.16 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L3 | metric_reach_0.1|opens_toward_root | 0.16 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L3 | metric_reach_0.2|opens_away_from_root | 0.32 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L3 | metric_reach_0.2|opens_toward_root | 0.32 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L3 | metric_reach_0.4|opens_away_from_root | 0.64 | 0.0000 | 0.0000 | 0.0000 | inf |
| L1-L3 | metric_reach_0.4|opens_toward_root | 0.64 | 0.0000 | 0.0000 | 0.0000 | inf |

## D18-C hierarchy against radial position

| representation | metric | Spearman rho vs radius |
| --- | --- | --- |
| Encoder | in_degree | 0.2105 |
| Encoder | out_degree | 0.2661 |
| Encoder | pagerank | 0.1168 |
| Encoder | k_core | 0.2856 |
| Encoder | depth_to_sink | 0.0410 |
| L1-L1 | in_degree | 0.2106 |
| L1-L1 | out_degree | 0.2651 |
| L1-L1 | pagerank | 0.1187 |
| L1-L1 | k_core | 0.2842 |
| L1-L1 | depth_to_sink | 0.0411 |
| L1-L2 | in_degree | 0.2106 |
| L1-L2 | out_degree | 0.2663 |
| L1-L2 | pagerank | 0.1181 |
| L1-L2 | k_core | 0.2849 |
| L1-L2 | depth_to_sink | 0.0414 |
| L1-L3 | in_degree | 0.2090 |
| L1-L3 | out_degree | 0.2654 |
| L1-L3 | pagerank | 0.1161 |
| L1-L3 | k_core | 0.2837 |
| L1-L3 | depth_to_sink | 0.0411 |

Metric cross-correlations (popularity family vs structural family):

| metric A | metric B | Spearman rho |
| --- | --- | --- |
| in_degree | out_degree | 0.9262 |
| in_degree | pagerank | 0.9458 |
| in_degree | k_core | 0.9427 |
| in_degree | depth_to_sink | 0.0616 |
| out_degree | pagerank | 0.8480 |
| out_degree | k_core | 0.9539 |
| out_degree | depth_to_sink | 0.0614 |
| pagerank | k_core | 0.8357 |
| pagerank | depth_to_sink | 0.0614 |
| k_core | depth_to_sink | 0.0616 |

## D18-D frozen curvature sensitivity

| curvature | angular AUC | geodesic AUC | median tangent norm | apex radius (median) | conformal factor (median) | share near ball boundary |
| --- | --- | --- | --- | --- | --- | --- |
| 0.25 | 0.8381 | 0.8415 | 2.1368 | 4.2736 | 5.295 | 0.0000 |
| 0.5 | 0.8381 | 0.8430 | 2.1368 | 4.2736 | 11.29 | 0.0000 |
| 1.0 | 0.8381 | 0.8440 | 2.1368 | 4.2736 | 36.9 | 0.0000 |
| 2.0 | 0.8381 | 0.8420 | 2.1368 | 4.2736 | 211.7 | 0.0000 |
| 4.0 | 0.8381 | 0.8367 | 2.1368 | 4.2736 | 2577 | 0.9049 |

in_degree, out_degree and PageRank are popularity-flavoured; k_core and depth_to_sink are structural. They are reported side by side and the cross-correlation block shows how far the two families agree, so radial position is never read as popularity alone.

## Conclusions

1. **Behaviour neighbours do not form an angular cluster.** At the encoder the partners that define a source's own direction sit a median 28.6 deg from it, and two partners of the same source sit 34.4 deg apart. The cone axis is no closer to its own defining points than the points are to each other, so there is no angular cluster for a cone to capture.
2. **A held-out successor is not atypical among its own partners, it is simply farther than an unrelated item.** AUC positive-vs-in-sample-partners is 0.531 (indistinguishable), while AUC positive-vs-popularity-matched is 0.161, i.e. real successors are farther from the behaviour direction than popularity-matched negatives are. The persistent-pair failure is therefore not an outlier problem inside a cluster; it is the absence of the cluster.
3. **Entailment cones over the behaviour direction have no discriminative power.** In the opens-away-from-root reading, a 20-degree cone covers 0.037 of positives but 0.329 of popularity-matched negatives, a lift below 1; the opens-toward-root reading has zero coverage at every aperture. Metric apertures, which curvature makes tiny at these radii, cover nothing. Cones therefore cannot be the next lever on this representation.
4. **Quantization barely moves any of this.** Encoder, L1-L1, L1-L2 and L1-L3 agree to within a few degrees on every angular statistic, and the AUC ordering survives all three levels: the behaviour structure is already absent before the codes are assigned.
5. **Radial position tracks the popularity family mildly and the structural family not at all.** in_degree rho = 0.211, out_degree 0.266, pagerank 0.117, k_core 0.286, depth_to_sink 0.041. The cross-correlation block shows why the two families must be kept apart: degree, PageRank and k-core agree at rho > 0.83, while depth-to-sink is uncorrelated with all of them (rho ~ 0.06), so a depth reading is not a popularity reading in disguise here.
6. **Curvature changes the metric read only modestly on these frozen embeddings.** The angular AUC is 0.8381 at every curvature because it is measured directly in the frozen tangent coordinates; the geodesic AUC moves between 0.8367 and 0.8440, a range of 0.0072. The median tangent norm is 2.14; this is not a ball radius. The exp map places every vector inside its curvature-specific Poincare ball, with the reported near-boundary share measured on the mapped points. The radius is 2||z|| for exp_0(z), independent of curvature. This is a frozen-embedding geometry comparison, not a retrained-model result.

