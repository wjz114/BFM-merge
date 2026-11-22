# Shared readout-null diagnostic

The exact full-pool fitting stream gives mean weighted raw-output energy5.674234% in the complement of the22-dimensional union of task prototype contrasts. Ordinary CE raw-output gradient energy in that complement averages4.978872%, ranging1.770431% on SEED4 to22.676276% on ADHD. These are output-space quantities, not parameter-space update fractions or bounds on generalization effects.

Removing this global-null output component preserves all training argmax predictions in FP64 and changes centered cosine logits only through a positive, sample-dependent scalar. The mean per-task weighted scalar is1.033293; the largest task averages are1.103092 on Workload and1.102998 on SSVEP. Weighted fitting CE changes0.784781 to0.780071 without changing these decisions. The audit checks the raw-gradient derivative, contrast preservation and logit scaling algebra; it performs no new inference, fitting or validation analysis.

This does not establish a dominant average gradient-energy explanation for the residual Joint gap. Given the completed scale/projection controls and the pending initialization comparison, another global-null training intervention is not prioritized on this diagnostic alone. It is neither a negative performance experiment nor proof that such a method cannot help. Source:`readout_global_null_diagnostic.json`; runtime8.20s.
