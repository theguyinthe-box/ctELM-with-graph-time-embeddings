#!/bin/bash
#SBATCH --job-name=ctELM_train_raw
#SBATCH --array=0-15
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_raw_%A_%a.out
#SBATCH --error=logs/train_raw_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=david.kaauwai@yale.edu

module load miniconda
conda activate ctELM_proj
export HF_HOME="${SLURM_SUBMIT_DIR}/.hf_cache"
export HF_HUB_DISABLE_PROGRESS_BARS=1
source "${SLURM_SUBMIT_DIR}/secrets.sh"

# Condition axis: raw_finetune = raw-text baseline B, raw_zeroshot =
# raw-text baseline C (this job still launches for zeroshot indices, but
# train.py's finetune=false guard makes it a fast no-op -- evaluate_model.py loads
# the base model directly for those). ctELM/embedding condition dropped from
# this sweep -- its flat embeddings/embeddings.npy was never built (only the
# per-labels/numbers-variant embeddings under embed_array.sh's subdirs exist).
CONDITIONS=(
    configs/pipeline_raw_finetune.yaml
    configs/pipeline_raw_zeroshot.yaml
)

EXPERIMENTS=(
    configs/experiments/chain2_reconstruct.yaml
    configs/experiments/chain2_generate.yaml
    configs/experiments/chain3_reconstruct.yaml
    configs/experiments/chain3_generate.yaml
    configs/experiments/chain4_reconstruct.yaml
    configs/experiments/chain4_generate.yaml
    configs/experiments/chain5_reconstruct.yaml
    configs/experiments/chain5_generate.yaml
)

COND_IDX=$((SLURM_ARRAY_TASK_ID / 8))
EXP_IDX=$((SLURM_ARRAY_TASK_ID % 8))
VARIANT=${CONDITIONS[$COND_IDX]}
EXPERIMENT=${EXPERIMENTS[$EXP_IDX]}
echo "Task $SLURM_ARRAY_TASK_ID: variant='$VARIANT' + $EXPERIMENT"

# Unique rendezvous port per array task -- avoids EADDRINUSE when Slurm
# co-schedules multiple array tasks on the same node (default torchrun port
# 29500 is shared, so concurrent tasks on one node collide).
MASTER_PORT=$((23000 + SLURM_ARRAY_TASK_ID))
torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE --master_port=$MASTER_PORT train.py --config configs/pipeline.yaml --variant "$VARIANT" --experiment "$EXPERIMENT"
