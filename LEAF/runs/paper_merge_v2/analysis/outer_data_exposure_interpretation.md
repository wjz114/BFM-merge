# Held out from instruction tuning is not necessarily fresh for this study

The official LEAF dataset pages identify Weibo2014 and Dreyer2023 as outside
instruction tuning. Both processed files are available locally. This is
useful provenance, but does not by itself establish a fresh confirmation set
for the current research process or absence from self-supervised pretraining.

Weibo2014 already appears in historical POC result CSVs for all10subjects and
1580 trials, including multiple models and all three original POC16 seeds.
Those measured outcomes predate the current method search. It can remain a
development-exposed transfer benchmark; it must not be relabeled as fresh
independent confirmation. Its repository builder supplies the event-code
mapping1/2 to Left/Right labels0/1. The builder's raw-data path is currently
absent, so no raw-to-processed reproduction was performed in this audit.

Dreyer2023 has the documented nested train/val/test structure. Its builder
and evaluator explicitly state that the Left/Right assignment is assumed,
because the delivered file has no label legend. The public held-out role
does not resolve that local mapping. No model inference or performance-based
label-order choice is justified by this metadata check.

`outer_data_exposure_audit.json` records all scanned result files, their
hashes and matching metric rows, plus HDF5 structures/attributes. No EEG
arrays or new model predictions were evaluated. Earlier SSVEP/Speech
provenance notes are preserved rather than replaced. The current four-arm
prehead experiment continues on its already frozen development protocol.

Primary documentation read:
https://leaf-bci.github.io/docs/datasets/ and
https://leaf-bci.github.io/docs/datasets/held-out.html.
