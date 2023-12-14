#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Change these!
#SBATCH --partition=devlab,learnlab,learnfair,scavenge
#SBATCH --job-name=train_and_eval
#SBATCH --time=32:00:00
#SBATCH --nodes=16
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --constraint=volta32gb
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --mem-per-gpu=100G

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


# Get command line arguments
while :; do
    case $1 in
    -h|-\?|--help)
        show_help    # Display a usage synopsis.
        exit
        ;;
    # train.py parameters
    -d|--data_dir)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            DATA_PATH=$2
            shift
        else
            die 'ERROR: "--data_dir" requires a non-empty option argument.'
        fi
        ;;
    -t|--subset_file)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            TRIE_FILE=$2
            shift
        else
            die 'ERROR: "Path of Trie file required'
        fi
        ;;
    -s|--scale)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            SCALE=$2
            shift
        else
            die 'ERROR: "--data_dir" requires a non-empty option argument.'
        fi
        ;;
    -r|--seed)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            SEED=$2
            shift
        else
            die 'Enter seed for training'
        fi
        ;;
    -?*)
        printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
        ;;
    *)               # Default case: No more options, so break out of the loop.
        break
    esac

    shift
done

OUTPUT_DIR="training_output/"
NUM_CHECKPOINTS=8
PRECISION="amp"  # You can also use amp_bfloat16 if supported by your hardware.

# Extract parent directory without the full path
parent_dir=$(basename "$(dirname "$TRIE_FILE")")
echo "Parent directory: $parent_dir"
# Extract filename without extension
filename_without_extension="${TRIE_FILE##*/}"
filename_without_extension="${filename_without_extension%.*}"
echo "Filename without extension: $filename_without_extension"
# experiment identifier
EXP_NAME="${parent_dir}_${filename_without_extension}"
echo "Experiment name: $EXP_NAME"

EXP_NAME="${EXP_NAME}_seed${SEED}_sc${scale}"

echo "Running with the following arguments:
train.py:
    --scale ${SCALE} 
    --data_dir ${DATA_PATH} 
    --output_dir ${OUTPUT_DIR} 
    --exp_name ${EXP_NAME} 
    --precision ${PRECISION} 
    --num_checkpoints ${NUM_CHECKPOINTS} 
    --seed ${SEED} 
    --subset_file ${TRIE_FILE}
"


# Change comment as needed
srun  python train.py \
    --scale ${SCALE} \
    --data_dir ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --exp_name ${EXP_NAME} \
    --precision ${PRECISION} \
    --num_checkpoints ${NUM_CHECKPOINTS} \
    --seed ${SEED} \
    --dataset_resampled \
    --accum_freq 1 \
    --subset_file ${TRIE_FILE}


