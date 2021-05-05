#!/bin/bash
#SBATCH -J GLUE
#SBATCH -o out-GLUE_%J.txt
#SBATCH -e err-GLUE_%J.txt
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH -t 10:00:00
#SBATCH --mem-per-cpu=64000
#SBATCH --gres=gpu:v100:1
#SBATCH --account=project_2000309
# run command

module purge
module load pytorch/1.6
export TASK=${1}
export DATASET=${2}
export POS=${3}

srun python run_corrupt_glue.py \
    --model_name_or_path roberta-base \
    --do_eval \
    --do_train \
    --pos $POS \
    --corrupt_dataset $DATASET \
    --task_name $TASK \
    --save_total_limit 2 \
    --overwrite_cache True \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --output_dir output/$TASK/$DATASET/$POS/