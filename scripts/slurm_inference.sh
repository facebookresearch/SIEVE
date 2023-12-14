#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Change these!
#SBATCH --partition=learnlab,learnfair,scavenge
#SBATCH --time=1-0:00:00
#SBATCH --job-name=inference
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=40
#SBATCH --constraint=volta32gb
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --mem=400G


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


nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

# Run untar
#DATA_DIR="/checkpoint/nasmahmoud/datacomp_data/small/downloaded_data/shards/{00000000..00001287}.tar"
#DATA_DIR="/checkpoint/nasmahmoud/datacomp_data/small/downloaded_data/shards/{00000000..00000001}.tar"
#DATA_DIR="/datasets01/datacomp_medium/shards/{00000000..00012895}.tar"
PER_DEVICE_BATCH_SIZE=256
MODEL_SIZE='base' # ('base', 'large')

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
            DATA_DIR=$2
            shift
        else
            die 'ERROR: "--data_dir" requires a non-empty option argument.'
        fi
        ;;
    -ms|--model_size)
        if [ "$2" ]; then
                MODEL_SIZE=$2
                shift
            else
                die 'ERROR: "--data_dir" requires a non-empty option argument.'
            fi
            ;;
    -s|--scale)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            SCALE=$2
            shift
        else
            die 'ERROR: "--scale" requires a non-empty option argument.'
        fi
        ;;
    -b|--per_device_batch_size)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            PER_DEVICE_BATCH_SIZE=$2
            shift
        else
            die 'ERROR: "--scale" requires a non-empty option argument.'
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

# Change these as needed!
#MODE="--clipcap"
MODE="--captioning"
NUM_WORKERS=10

#TIMESTAMP=$(date +"%m%d%H%M%S")
OUTPUT_DIR="inference_results/${MODE:2}_${SLURM_JOB_NAME}${SLURM_JOB_ID}"
mkdir "${OUTPUT_DIR}"

echo "Running with the following arguments:
webdataset_inference.py:
MODE=$MODE
MODEL_SIZE=$MODEL_SIZE
PER_DEVICE_BATCH_SIZE=$PER_DEVICE_BATCH_SIZE
NUM_WORKERS=$NUM_WORKERS
DATA_DIR=$DATA_DIR
OUTPUT_DIR=$OUTPUT_DIR
Numder_of_nodes=$SLURM_JOB_NUM_NODES
GPUS_Requested=$SLURM_GPUS_PER_TASK
"

srun torchrun \
    --nnodes $SLURM_JOB_NUM_NODES \
    --nproc_per_node $SLURM_GPUS_PER_TASK  \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:29500 \
    webdataset_inference.py $MODE  --data_dir $DATA_DIR \
    --model_size $MODEL_SIZE \
    --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --output_dir $OUTPUT_DIR \
    --sentence_language_model unilingual \
    --save_all_captions \
    --model_url "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth"

# copy log files
cp "slurm_logs/${SLURM_JOB_NAME}_${SLURM_JOBID}.out" "${OUTPUT_DIR}/"
cp "slurm_logs/${SLURM_JOB_NAME}_${SLURM_JOBID}.err" "${OUTPUT_DIR}/"