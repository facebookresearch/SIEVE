#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Change these!
#SBATCH --partition=devlab,learnlab,learnfair,scavenge
#SBATCH --job-name=eval
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=10
#SBATCH --constraint=volta32gb
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --open-mode=append
#SBATCH --requeue

# Example usage:
# Run using conda and make sure to have the conda env activated when running sbatch.
source /private/home/${USER}/.bashrc
source activate sieve

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

# Get options
track='filtering'
while getopts f:t:e: arg
do
    case "$arg" in
    f)  infofolder="$OPTARG";;
    t)  track="$OPTARG";;
    esac
done

# Run evaluation
infofolder_fullpath="training_output/${infofolder}"
results="eval_results/${infofolder}"
mkdir "${results}"

echo "Running with the following arguments:
evaluate.py:
infofolder_fullpath=$infofolder_fullpath
results=$results
"

python evaluate.py \
    --train "$infofolder_fullpath" \
    --output "$results" \
    --batch_size 2048 
    

echo "finished evaluation"


