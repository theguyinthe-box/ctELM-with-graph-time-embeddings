#!/bin/bash
# Usage:
#   bash scripts/submit_pipeline.sh configs/experiments/<name>.yaml
#   bash scripts/submit_pipeline.sh configs/experiments/<name>.yaml --variant configs/pipeline_<variant>.yaml
#   bash scripts/submit_pipeline.sh configs/experiments/<name>.yaml --variant configs/pipeline_<variant>.yaml --with-embed
#   bash scripts/submit_pipeline.sh configs/experiments/<name>.yaml --variant configs/pipeline_<variant>.yaml --after-embed <JID>
#
# --variant <yaml>:        variant pipeline config (labels/numbers combination, or raw-text condition)
# --with-build:            submit build_graph + embed jobs and chain this experiment to them.
# --after-embed <JID>      chain this experiment's prepare step to an already-submitted embed job.
# --skip-train:            skip the train job (e.g. train.finetune: false in the variant) and point
#                          evaluate directly at prepare_dataset -- for zero-shot raw-text baselines.
# Omit --with-build/--after-embed if graph data and embeddings already exist on the cluster.

set -euo pipefail

EXPERIMENT="${1:-}"
VARIANT=""
WITH_BUILD=0
EMBED_JID=""
SKIP_TRAIN=0

i=2
while [ $i -le $# ]; do
    arg="${!i}"
    if [ "$arg" = "--variant" ]; then
        i=$((i + 1))
        VARIANT="${!i}"
    elif [ "$arg" = "--with-build" ]; then
        WITH_BUILD=1
    elif [ "$arg" = "--after-embed" ]; then
        i=$((i + 1))
        EMBED_JID="${!i}"
    elif [ "$arg" = "--skip-train" ]; then
        SKIP_TRAIN=1
    fi
    i=$((i + 1))
done

if [ -z "$EXPERIMENT" ]; then
    echo "Usage: bash scripts/submit_pipeline.sh <experiment.yaml> [--variant <variant.yaml>] [--with-build | --after-embed <JID>] [--skip-train]"
    exit 1
fi

EXP_NAME=$(basename "$EXPERIMENT" .yaml)
VARIANT_NAME=$([ -n "$VARIANT" ] && basename "$VARIANT" .yaml || echo "default")
echo "Submitting pipeline for experiment: $EXP_NAME (variant: $VARIANT_NAME)"

if [ "$WITH_BUILD" -eq 1 ]; then
    BUILD_JID=$(sbatch --parsable scripts/build_graph.sh "$VARIANT")
    echo "  build_graph     job $BUILD_JID"
    EMBED_JID=$(sbatch --parsable --dependency=afterok:$BUILD_JID scripts/embed.sh "$VARIANT")
    echo "  embed           job $EMBED_JID"
fi

DEP_PREP=""
[ -n "$EMBED_JID" ] && DEP_PREP="--dependency=afterok:$EMBED_JID"

JID_PREP=$(sbatch --parsable $DEP_PREP scripts/prepare_dataset.sh "$EXPERIMENT" "$VARIANT")
echo "  prepare_dataset job $JID_PREP"

if [ "$SKIP_TRAIN" -eq 1 ]; then
    JID_EVAL=$(sbatch --parsable --dependency=afterok:$JID_PREP scripts/evaluate.sh "$EXPERIMENT" "$VARIANT")
    echo "  evaluate        job $JID_EVAL (train skipped)"

    echo ""
    echo "Chain: ${JID_PREP} → ${JID_EVAL}"
else
    JID_TRAIN=$(sbatch --parsable --dependency=afterok:$JID_PREP scripts/train.sh "$EXPERIMENT" "$VARIANT")
    echo "  train           job $JID_TRAIN"

    JID_EVAL=$(sbatch --parsable --dependency=afterok:$JID_TRAIN scripts/evaluate.sh "$EXPERIMENT" "$VARIANT")
    echo "  evaluate        job $JID_EVAL"

    echo ""
    echo "Chain: ${JID_PREP} → ${JID_TRAIN} → ${JID_EVAL}"
fi
echo "Each job will email david.kaauwai@yale.edu on END or FAIL."
