#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Change these!
#SBATCH --partition=devlab,learnlab,learnfair,scavenge
#SBATCH --job-name=sentencesim
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=10
#SBATCH --constraint=volta32gb
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --open-mode=append
#SBATCH --requeue

# Example usage:
# sbatch slurm_train.sh
# Run using conda and make sure to have the conda env activated when running sbatch.
source /private/home/${USER}/.bashrc
source activate datacomp

module load openmpi
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`
echo go $COUNT_NODE
echo $HOSTNAMES
echo "$PWD" 

DATA_DIR=$1

RESULT_DIR=$2

echo "Data folder is: ${DATA_DIR}"
echo "Result folder is ${RESULT_DIR}"

if [ -z "$2" ]; then
    echo "Result file not set"
    python sentence_similarity_inference.py \
        --per_device_batch_size 128 \
        --parquet_dir $DATA_DIR 
else
    echo "Result file is set"
    python sentence_similarity_inference.py \
        --per_device_batch_size 128 \
        --parquet_dir $DATA_DIR \
        --output_dir $RESULT_DIR
fi

echo "finished inference"


