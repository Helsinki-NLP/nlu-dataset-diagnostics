#!/bin/bash

export TASK=${1}
export DATASET=${2}
export POS=${3}

if [ "$TASK_NAME" = "mrpc" ]
then
    export EPOCHS=5
else
    export EPOCHS=3
fi

python3 run_corrupt_glue.py \
    --model_name_or_path roberta-base \
    --do_eval \
    --do_train \
    --pos $POS \
    --corrupt_dataset $DATASET \
    --task_name $TASK\
    --save_total_limit 2 \
    --overwrite_cache True \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs $EPOCHS \
    --output_dir output/$TASK/$DATASET/$POS/