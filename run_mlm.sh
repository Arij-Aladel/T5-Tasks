#!/bin/bash

# these commands are for the model training, you might want to change them

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_LAUNCH_BLOCKING=1
cmd="horovodrun --gloo -np 8 python3 run_mlm_hvd.py \
        --batch_size=20 \
        --gradient_accumulation_steps=1 \
        --experiment_path=MLM/128/    \
        --tokenizer_path=tokenizer/  \
        --train_file_path=Data/wiki_103/128train_data.pt  \
        --valid_file_path=Data/wiki_103/128valid_data.pt  \
        --save_interval=1000 \
        --working_dir=. \
        --max_seq_length=128 \
        --config_dir=configs \
        --model_name=transformers:T5ForConditionalGeneration \
        --lr=5e-03 \
        --epochs=10 \
        --optimizer=Adafactor"



echo $cmd
$cmd
