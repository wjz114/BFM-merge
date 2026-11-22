#!/usr/bin/env bash
# With no argument, download the official snapshot. For offline use pass its local directory.
set -eu
cd "$(dirname "$0")"
rev=e8c3b32edf5434bc2275fc9bab85f82640a19130
model_dir=${1:-"checkpoints/all-mpnet-base-v2/$rev"}
prototypes=runs/continual_v12/stage_a/mpnet_descriptions
if [ "$#" -eq 0 ]; then
    python - "$rev" "$model_dir" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("sentence-transformers/all-mpnet-base-v2", revision=sys.argv[1], local_dir=sys.argv[2],
                  allow_patterns=["model.safetensors", "config.json", "tokenizer*.json", "vocab.txt",
                                  "special_tokens_map.json", "modules.json", "1_Pooling/config.json",
                                  "sentence_bert_config.json", "config_sentence_transformers.json"])
PY
fi
python -B continual_prototypes.py --mpnet "$model_dir" --outdir "$prototypes"
python -B continual_stage_a.py fit --prototype-dir "$prototypes"
