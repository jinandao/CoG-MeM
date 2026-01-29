#!/bin/bash

# 设置参数
MODEL_DIR="/root/Models/Qwen2.5-7B-Instruct"
TRAIN_JSON_PATH="./Datasets/memory_query/train/memory_query_data_train.json"
TEST_JSON_PATH="./Datasets/memory_query/test/memory_query_data_test.json"
OUTPUT_DIR="./Output/Memory_Query"
PER_DEVICE_TRAIN_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=1
LOGGING_STEPS=10
NUM_TRAIN_EPOCHS=5
LEARNING_RATE=2e-5

# 运行训练脚本
python memory_query_sft.py \
    --model_dir "$MODEL_DIR" \
    --train_json_path "$TRAIN_JSON_PATH" \
    --test_json_path "$TEST_JSON_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --logging_steps $LOGGING_STEPS \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --gradient_checkpointing