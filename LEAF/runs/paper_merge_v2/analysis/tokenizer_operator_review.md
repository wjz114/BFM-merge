# Tokenizer operator-coordinate pilot

All16 original expert functions pass real-data preservation checks. The base, architecture, calibration IDs, optimizer budget, and average window are fixed.

| Coordinates | Selected validation BAcc | Fixed-average validation BAcc |
| --- | ---: | ---: |
| original | 63.8608% | 63.8344% |
| canonical | 63.7385% | 63.5942% |

Canonical-minus-original: selected -0.1223 pp; fixed average -0.2402 pp.

Exact expert function preservation changes the merged search space and its optimization geometry. Any effect cannot be assigned solely to factor cross-terms without further controls. Operator/gauge-invariant merging is existing prior art, including SLOA and GeoMerge. One exposed validation seed is not independent confirmation.

Full per-task deltas, export checks, and costs are in `tokenizer_operator_summary.json`. No official test was performed.

Precision-audit amendment: the original coefficient fit above used BF16 while the canonical fit used FP32. These earlier differences cannot isolate the coordinate intervention. The completed original FP32 control has fixed-average validation BAcc63.9126%, versus canonical63.5942% (−0.3184pp). The matched comparison is retained separately in `tokenizer_operator_summary_fp32_control.json`. Expert function-preservation checks are unaffected.
