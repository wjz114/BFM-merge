# Shared-semantics MI transfer diagnostic

All six domains were selected by matching prototype sets, before observing transfer. Rows are source experts; columns are target domains. Entries are BAcc changes versus the original base in percentage points. No test inference or adaptation was performed.

## calibration

| Expert / target | OpenBMI | ShanghaiU | HighGamma | Cho2017 | Shin2017A | PhysioNet |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenBMI | +16.942 | +0.635 | -38.477 | -2.148 | -1.817 | -3.628 |
| ShanghaiU | -22.650 | +40.311 | -9.180 | -15.625 | -9.398 | -14.559 |
| HighGamma | -44.557 | -5.171 | +33.398 | -22.656 | -20.793 | -32.860 |
| Cho2017 | -10.563 | -0.523 | -33.203 | +28.125 | +1.923 | -17.150 |
| Shin2017A | -19.922 | -0.818 | -19.727 | -7.617 | +32.976 | -20.491 |
| PhysioNet | -9.835 | -7.797 | -24.023 | -11.719 | -9.430 | +24.274 |

Off-diagonal mean -14.459 pp; 2/30 positive pairs.

## validation

| Expert / target | OpenBMI | ShanghaiU | HighGamma | Cho2017 | Shin2017A | PhysioNet |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenBMI | +7.870 | +0.096 | -35.328 | -3.713 | +5.314 | -3.891 |
| ShanghaiU | -19.863 | +13.326 | -9.331 | -11.386 | -8.605 | -12.268 |
| HighGamma | -42.034 | -2.514 | +24.162 | -18.564 | -19.750 | -28.329 |
| Cho2017 | -6.147 | +0.447 | -28.298 | +12.252 | +11.611 | -15.993 |
| Shin2017A | -16.717 | -1.785 | -15.189 | -6.559 | +19.371 | -17.177 |
| PhysioNet | -7.527 | +0.521 | -22.033 | -8.106 | -11.464 | +10.683 |

Off-diagonal mean -11.819 pp; 5/30 positive pairs.

## Interpretation

Most cross-domain expert changes are harmful relative to base, while some shared-semantic pairs transfer beneficially. Prediction change alone cannot distinguish these cases. This is descriptive evidence: it does not establish that RI v2 loses the beneficial pairs, that eliminating harmful pairs improves a merged model, or that readout equality makes the adaptation targets mathematically contradictory. A controlled adaptation and merge comparison is required.

Full logits and help/harm counts are retained in `seed0/shared_mi_transfer`. The source validation set is development-exposed, and domains/pairs are not independent statistical replicates.
