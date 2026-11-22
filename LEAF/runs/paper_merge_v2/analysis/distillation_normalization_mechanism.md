# Fixed-norm versus angular feature supervision

This is a posthoc explanation and fixed follow-up design, not a demonstrated accuracy mechanism.

Let the raw student output be $z=ru$, with $\|u\|=1$, the teacher unit feature be $t$, and the frozen initial student norm be $n_0$. The complete-feature auxiliary currently tested is proportional to

$$\|z/n_0-t\|^2=(r/n_0-1)^2+(r/n_0)\|u-t\|^2.$$

At initialization $r/n_0=1$, its value equals the unit-feature loss, but its gradient also has a radial component. For a misaligned teacher, the derivative with respect to $r/n_0$ is positive, so local output-space gradient descent initially tends to reduce the norm. The native cosine classifier has zero directional derivative along this radial direction. That does not make the component necessarily harmful: norm changes alter future gradient scales, and shared network Jacobians and Adam's parameter preconditioning do not preserve this output-space decomposition.

The follow-up uses $\|u-t\|^2$ or $\|A_t(u-t)\|^2$, with derivatives through normalization. Writing $P=A_t^\top A_t$ (or $I$), its raw-output gradient is proportional to

$$\frac{1}{r}(I-uu^\top)P(u-t).$$

It is tangent to the unit sphere at every nonzero raw output. This is an exact readout identity, not a claim that the trained network's feature norms remain constant.

For centered prototype rows $C$, the own-contrast projector satisfies $P=C^\top(CC^\top)^+C$. Thus projected unit-feature matching equals

$$d^\top(CC^\top)^+d,\qquad d=C(u-t).$$

This is a whitened centered-logit quadratic. It uses the same teacher decision information as the native logits. For binary tasks the contrast space has rank1, so whitening only rescales a single margin. The measured prototype contrast condition numbers span1–1.859, which does not support a severe within-task ill-conditioning claim.

`full_distillation_radial_and_label_geometry.json` uses the exact204800-entry fitting multiset, full-pool class weights and frozen initial outputs. Its mean per-task radial energy fractions are47.764973% for full-feature and30.125088% for own-contrast supervision. Initial teacher correctness is96.840881%; local CE conflict fractions are only2.543997%–3.499127%. No validation labels enter these statistics.

`angular_distillation_scales.json` verifies analytic gradients at initial and1.7x raw outputs, scale invariance, vanishing radial gradients, exact zero-auxiliary CE, and the centered-logit quadratic identity. Maximum analytic relative error2.819e-16; maximum radial energy5.909e-31; maximum logit equivalence error1.981e-15. The initial RMS rule raises auxiliary scalars by1.102–1.741x (full) and1.101–1.388x (contrast). Therefore the experiment compares complete fixed recipes; it is not an isolated normalization intervention at identical scalar coefficient or identical parameter-gradient strength.

The frozen design is `../inputs/angular_distillation_design.json`. Broad norm/direction separation and spherical distillation are established prior art; see [SKD](https://arxiv.org/html/2010.07485v1) and the literature ledger. Accuracy, task stability and a distinct contribution still require evidence.
