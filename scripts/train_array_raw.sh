#!/bin/bash
#SBATCH --job-name=ctELM_train_raw
#SBATCH --array=0-23
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

# Condition axis: "" = ctELM (embedding, compressed, fine-tuned -- default,
# no --variant needed), raw_finetune = raw-text baseline B, raw_zeroshot =
# raw-text baseline C (this job still launches for zeroshot indices, but
# train.py's finetune=false guard makes it a fast no-op -- evaluate.py loads
# the base model directly for those).
CONDITIONS=(
    ""
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
echo "Task $SLURM_ARRAY_TASK_ID: variant='${VARIANT:-embedding (default)}' + $EXPERIMENT"

if [ -n "$VARIANT" ]; then
    torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE train.py --config configs/pipeline.yaml --variant "$VARIANT" --experiment "$EXPERIMENT"
else
    torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE train.py --config configs/pipeline.yaml --experiment "$EXPERIMENT"
fi
