# Fixed-average factorial review

All6 ordinary-CE exports and official evaluations completed. Every model was frozen before new test inference. Existing test sets remain development-exposed.

| Layout | CE | Checkpoint | Mean official BAcc (%) | Sample SD (pp) |
| --- | --- | --- | ---: | ---: |
| block | ordinary | selected source | 57.071250 | 0.262449 |
| block | ordinary | fixed average | 56.858958 | 0.343043 |
| block | balanced | selected source | 57.111875 | 0.488015 |
| block | balanced | fixed average | 57.185625 | 0.425515 |
| tensor | ordinary | selected source | 57.204167 | 0.271409 |
| tensor | ordinary | fixed average | 57.039167 | 0.311253 |
| tensor | balanced | selected source | 57.314375 | 0.309695 |
| tensor | balanced | fixed average | 57.509375 | 0.348189 |

Tensor paired contrasts, percentage points (mean ± sample SD):

- average_balanced_ce: +0.195000 ± 0.225350; positive in 2/3 seeds.
- average_ordinary_ce: -0.165000 ± 0.068946; positive in 0/3 seeds.
- balance_with_average: +0.470208 ± 0.058344; positive in 3/3 seeds.
- balance_without_average: +0.110208 ± 0.110260; positive in 3/3 seeds.
- balance_x_average: +0.360000 ± 0.162015; positive in 3/3 seeds.

The class-weighting benefit after averaging is concentrated in Workload:
- Workload mean gain: +6.756667 pp; share of net macro gain: 89.81%.
- Mean over the other15 tasks: +0.051111 ± 0.229211 pp; paired values [-0.18666666666666498, 0.0693333333333342, 0.27066666666666833].
- Seed0:11 of16 tasks decline in the balanced-vs-ordinary averaged comparison. The primary16-task macro metric is unchanged; these are descriptive sensitivity checks.

Interpretation: averaging is not a uniformly beneficial independent component. The combined result is a small, task-concentrated baseline improvement. It does not establish broad interference removal, novel methodology, or an independent confirmation. The best prior candidate remains57.509375%, Joint60.143958%, with2.634583 pp remaining gap.

Numerical source: `average_factorial.json`; concentration: `average_task_concentration.json`.
