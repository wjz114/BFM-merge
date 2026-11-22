# Paired instance structure versus class-conditioned target information

Frozen design: `../inputs/prehead_distillation_design.json`.
This is a conventional feature-distillation control, not a new method claim.
It follows the completed preprojection audit and close LELP/relational-KD
prior, and uses existing frozen experts and ordinary LEAF export.

Let h_i be the student's2048-dimensional final Linear input and t_i the
own-expert feature. All four arms use the same full-pool balanced CE stream,
initial merger,1600 Adam updates, learning rate1e-5 and fixed six-state mean.
The three auxiliary arms use the same per-task alpha, fixed from initial
paired-feature versus CE gradient RMS at this interface. The targets are:

- paired: t_i;
- mean: the fitting-multiset mean of t for the true class of i;
- shuffled: t_pi(i), with pi a fixed mapping within true class and fitting
  exposure multiplicity. Every repeated input keeps one target. Singleton
  strata remain unchanged, with their exposure counts reported.

The shuffle preserves the exact weighted target multiset, including its
class means and second moments. It changes the input-target correspondence.
With w_i the fixed class weight and d=2048, the auxiliary-loss difference is

    L_paired - L_shuffled
      = -2/(N d) sum_i w_i <h_i, t_i - t_pi(i)>.

The student squared norm cancels pointwise; the teacher squared norm cancels
after summation because the target multiset and class weights are preserved.
This elementary identity explains what the intervention removes; it is not
a new theorem or a guarantee of better generalization. Initial gradient
norms can still differ because input-target correlations differ. They will
be reported, and alpha will not be rematched after observing results.

Writing t_i = mu_y + r_i similarly decomposes paired-feature MSE into the
class-mean matching term, a fixed residual-energy term, and a student/teacher
residual cross term. The mean control removes the sample-dependent target
residual. Means are computed inFP64 and stored inFP32 for native training;
the intervention uses the stored values, with no learned class centers.

The primary question is whether paired features improve BAcc over matched
ordinary CE. Paired-versus-mean and paired-versus-shuffle then test whether
sample correspondence is necessary for that benefit. Beating shuffle while
failing to improve CE would only show that scrambled targets can hurt; it
would not establish useful knowledge transfer. Matching mean performance
would not support a necessity claim for instance variation at this recipe.

All16 task effects and deletion sensitivity will be retained. This is one
expert bank and exposed validation. A material result would still require
replication, an intervention-linked mechanism and differentiation from
existing feature/subclass distillation before satisfying the paper goal.
No rank, layer, temperature, loss-scale or learning-rate sweep is registered.
