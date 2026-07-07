#!/bin/bash
#SBATCH --job-name=ctELM_prepare
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/prepare_%j.out
#SBATCH --error=logs/prepare_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=david.kaauwai@yale.edu

module load miniconda
conda activate ctELM_proj
export HF_HOME="${SLURM_SUBMIT_DIR}/.hf_cache"
export HF_HUB_DISABLE_PROGRESS_BARS=1

EXPERIMENT=${1:-}
VARIANT=${2:-}

ARGS=""
[ -n "$VARIANT" ]    && ARGS="$ARGS --variant $VARIANT"
[ -n "$EXPERIMENT" ] && ARGS="$ARGS --experiment $EXPERIMENT"

python prepare_graph_dataset.py --config configs/pipeline.yaml $ARGS
