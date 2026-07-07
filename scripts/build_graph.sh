#!/bin/bash
#SBATCH --job-name=ctELM_build
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/build_%j.out
#SBATCH --error=logs/build_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=david.kaauwai@yale.edu

module load miniconda
conda activate ctELM_proj
export HF_HOME="${SLURM_SUBMIT_DIR}/.hf_cache"
export HF_HUB_DISABLE_PROGRESS_BARS=1

VARIANT=${1:-}

if [ -n "$VARIANT" ]; then
    python -m openelm.graph --config configs/pipeline.yaml --variant "$VARIANT"
else
    python -m openelm.graph --config configs/pipeline.yaml
fi
