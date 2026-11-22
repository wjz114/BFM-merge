# Binary anti-alignment does not explain the observed residual errors

The specific hypothesis was that many wrong shared-model predictions sit
almost opposite the correct prototype contrast, where a normalized readout
has an almost-zero angular derivative. The cache evidence does not support
this as a broad explanation or justify an escape-oriented training method.

For raw feature h, unit u=h/||h||, prototype difference d=p1-p0, and binary
label y, the native binary CE feature gradient is

    grad_h CE = (sigmoid(u^T d) - y) (I - uu^T) d / ||h||.

Let a=(2y-1)u^T d/||d|| be the alignment with the correct contrast.
The readout's angular factor is sqrt(1-a²). It becomes zero at both a=1
and a=-1; the latter is a wrong extreme. This elementary derivative is not
a new mechanism theorem. `diagnose_binary_antialignment.py` checks it
against FP64 autograd on32 cached raw features per binary task; the maximum
absolute gradient error is5.96e-19.

The diagnostic is cache-only and completed in2.96s. It keeps all eight binary
tasks, the existing full fitting exposures, and the dependent original and
replacement128-row caches for initial, CE and CE+KL models. Original class
weights are retained. No validation inference, test read or optimization.

In the full initial-merger fitting stream, mean weighted error is17.7273%.
36.7096% of that macro error mass has a<-0.9, but **none** has a<-0.99 or
a<-0.999. The expert-correct/shared-wrong subset gives34.5210%,0%,0%
respectively. None of the wrong predictions in either128-row subset of
initial, CE or CE+KL models has a<-0.99 either. All cutoffs are reported;
none becomes a selected loss gate.

For reference, a=-0.9 still has angular factor0.4359 and a=-0.99 has0.1411.
Across the eight tasks, the10th-percentile factor among full-stream wrong
predictions ranges from0.2183 to0.9499. Raw-feature norm and actual analytic
feature-gradient quantiles are also retained in the JSON. Absolute norm
scales cannot alone diagnose a parameter-optimization bottleneck: the upstream
parameter Jacobian and Adam updates matter, and positive rescaling of a
normalized projection changes the parameterization without changing logits.

KL reduces the share of binary subset errors below-0.9 from31.78% to12.50%
on the original subset and34.90% to13.79% on the replacement subset, relative
to CE. Their error rates do not improve consistently: original CE/KL errors
are10.4492%/10.9375%, replacement14.5508%/14.1602%. These are dependent
training subsets, not generalization estimates or a causal demonstration.

[Draganov et al., The Hidden Pitfalls of the Cosine Similarity Loss,
HiLD2024](https://arxiv.org/pdf/2406.16468), Section3, already studies inverse
norm and opposite-direction gradient effects. Its direct embedding updates
are not the full shared LEAF/Adam trajectory. No radial escape, norm-scaling
or cutoff sweep is launched from this diagnostic. The near-antipode
explanation is unsupported in the measured scope; broader optimization
effects remain separate questions.
