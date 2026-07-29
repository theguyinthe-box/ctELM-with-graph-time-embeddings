#!/bin/bash
# Submit the raw-text baseline sweep (finetune vs. zeroshot) as job arrays.
# 2 conditions (raw_finetune, raw_zeroshot) x 8 chain experiments (chain2-5,
# generate/reconstruct), on the base pipeline (no labels/numbers variant) so
# both conditions compare on identical graph data. The ctELM/embedding
# condition is not part of this sweep -- its flat embeddings/embeddings.npy
# was never built (only per-labels/numbers-variant embeddings exist).
# Assumes graph data is already built on the cluster.
# Usage: bash scripts/submit_all_raw.sh

set -euo pipefail

echo "Submitting prepare array (8 experiments, shared across both conditions)..."
PREP_JID=$(sbatch --parsable scripts/prepare_array_raw.sh)
echo "  prepare array job $PREP_JID"

# train is 16-wide (2 conditions x 8 experiments) but prepare is only 8-wide
# (condition-agnostic dataset), so the array cardinalities don't match --
# aftercorr (index-for-index) isn't valid here; wait for the whole prepare
# array to finish instead.
echo "Submitting train array (16 experiments, afterok:prepare)..."
TRAIN_JID=$(sbatch --parsable --dependency=afterok:$PREP_JID scripts/train_array_raw.sh)
echo "  train array job $TRAIN_JID"

# train and eval are both 16-wide with identical (condition, experiment)
# indexing, so aftercorr (index-for-index) is valid and lets eval start on
# each index as soon as its own train task finishes, instead of waiting for
# the slowest one.
echo "Submitting eval array (16 experiments, aftercorr:train)..."
EVAL_JID=$(sbatch --parsable --dependency=aftercorr:$TRAIN_JID scripts/eval_array_raw.sh)
echo "  eval array job $EVAL_JID"

echo ""
echo "Pipeline: prepare[$PREP_JID] → train[$TRAIN_JID] → eval[$EVAL_JID]"
echo "Each job will email david.kaauwai@yale.edu on END or FAIL."
