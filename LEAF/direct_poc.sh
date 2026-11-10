#!/usr/bin/env bash
# Official test of one POC run: every checkpoint (Base, experts, Joint, merges) is
# evaluated by c_direct_inference.py as-is (frozen model, instruction levels 0/1/2,
# instruction choice seed 42), split across two GPUs, then collected into
# $RUN/direct_results.csv by eval_poc.py --collect-direct.
#   RUN=runs/poc16/seed0 DATASETS=<sources> [DIRECT_DS=<subset>] bash direct_poc.sh
# DIRECT_DS empty (default) = all datasets in configs/tasks.yaml.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

BASE=${BASE:-checkpoints/leaf-v1.0-instruct-mpnet-base.ckpt}
: "${RUN:?set RUN}" "${DATASETS:?set DATASETS (source datasets)}"
DIRECT_DS=${DIRECT_DS:-}
OUT=$RUN/direct_inference
mkdir -p "$OUT" "$RUN/logs"

JOBS=("base:$BASE" "joint:$RUN/joint/best.ckpt" "uniform:$RUN/merge_uniform/merged.ckpt"
      "global:$RUN/merge_global/merged.ckpt" "block:$RUN/merge_block/merged.ckpt")
for DS in ${DATASETS//,/ }; do JOBS+=("expert_$DS:$RUN/experts/$DS/best.ckpt"); done
DSARG=()
if [[ -n "$DIRECT_DS" ]]; then DSARG=(--ds "$DIRECT_DS"); fi

run_list() {
  local gpu=$1; shift
  for pair in "$@"; do
    local name=${pair%%:*} ckpt=${pair#*:}
    python c_direct_inference.py --ckpt "$ckpt" "${DSARG[@]}" --gpu "$gpu" --bs 256 --level 0,1,2 \
      --output "$OUT/$name.csv" > "$OUT/$name.log" 2>&1
  done
}
GPU0=(); GPU1=()
for i in "${!JOBS[@]}"; do
  if (( i % 2 == 0 )); then GPU0+=("${JOBS[$i]}"); else GPU1+=("${JOBS[$i]}"); fi
done
run_list 0 "${GPU0[@]}" & P0=$!
run_list 1 "${GPU1[@]}" & P1=$!
wait "$P0"; wait "$P1"

python eval_poc.py --collect-direct "$OUT" --datasets "$DATASETS" \
  --output "$RUN/direct_results.csv" > "$RUN/logs/direct_collect.log" 2>&1
