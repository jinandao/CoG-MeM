#!/bin/bash

# ==========================================
# 课程学习：压缩对话训练脚本
# 1. 先运行SFT训练
# 2. 然后运行DPO训练（使用SFT训练的模型）
# ==========================================

# 设置通用参数
MODEL_DIR="Qwen/Qwen2.5-7B-Instruct"

# SFT训练参数
SFT_TRAIN_JSON_PATH="./Datasets/compress_conversation/compress_conversation_sft.json"
SFT_OUTPUT_DIR="./Output/Compress_Conversation/Compress_Conversation_SFT"
SFT_PER_DEVICE_TRAIN_BATCH_SIZE=2
SFT_GRADIENT_ACCUMULATION_STEPS=1
SFT_NUM_TRAIN_EPOCHS=4
SFT_LEARNING_RATE=2e-5

# DPO训练参数
DPO_TRAIN_JSON_PATH="./Datasets/compress_conversation/compress_conversation_dpo.json"
DPO_OUTPUT_DIR="./Output/Compress_Conversation/Compress_Conversation_DPO"
DPO_PER_DEVICE_TRAIN_BATCH_SIZE=2
DPO_GRADIENT_ACCUMULATION_STEPS=2
DPO_LEARNING_RATE=2e-6
DPO_BETA=0.2
DPO_MAX_LENGTH=4096
DPO_MAX_PROMPT_LENGTH=2560
DPO_MAX_GRAD_NORM=0.1
DPO_NUM_TRAIN_EPOCHS=5
DPO_SEED=100

# ==========================================
# 步骤1: 运行SFT训练
# ==========================================
echo "开始SFT训练..."
echo "SFT输出目录: $SFT_OUTPUT_DIR"

python memory_compress_sft.py \
    --train_json_path "$SFT_TRAIN_JSON_PATH" \
    --model_dir "$MODEL_DIR" \
    --output_dir "$SFT_OUTPUT_DIR" \
    --per_device_train_batch_size $SFT_PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $SFT_GRADIENT_ACCUMULATION_STEPS \
    --num_train_epochs $SFT_NUM_TRAIN_EPOCHS \
    --learning_rate $SFT_LEARNING_RATE \
    --gradient_checkpointing

SFT_EXIT_CODE=$?
if [ $SFT_EXIT_CODE -ne 0 ]; then
    echo "SFT训练失败，退出码: $SFT_EXIT_CODE"
    exit 1
fi

echo "SFT训练完成！"
# 方案A: 如果按照方案1修改了代码，模型会保存在final_model子目录
SFT_MODEL_PATH=""
FINAL_MODEL_DIR="$SFT_OUTPUT_DIR/final_model"
if [ -d "$FINAL_MODEL_DIR" ]; then
    SFT_MODEL_PATH="$FINAL_MODEL_DIR"
    echo "找到最终模型: $SFT_MODEL_PATH"
else
    # 如果没有final_model目录，则使用SFT_OUTPUT_DIR
    SFT_MODEL_PATH="$SFT_OUTPUT_DIR"
    echo "未找到final_model目录，使用SFT输出目录: $SFT_MODEL_PATH"
fi
# ==========================================
# 步骤2: 运行DPO训练
# 使用SFT训练的模型作为起点
# ==========================================
echo ""
echo "开始DPO训练..."
echo "使用SFT模型: $SFT_OUTPUT_DIR"
echo "DPO输出目录: $DPO_OUTPUT_DIR"

python memory_compress_dpo.py \
    --model_dir "$MODEL_DIR" \
    --train_json_path "$DPO_TRAIN_JSON_PATH" \
    --sft_lora_path "$SFT_MODEL_PATH" \
    --per_device_train_batch_size $DPO_PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $DPO_GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $DPO_LEARNING_RATE \
    --output_dir "$DPO_OUTPUT_DIR" \
    --beta $DPO_BETA \
    --max_length $DPO_MAX_LENGTH \
    --max_prompt_length $DPO_MAX_PROMPT_LENGTH \
    --max_grad_norm $DPO_MAX_GRAD_NORM \
    --num_train_epochs $DPO_NUM_TRAIN_EPOCHS \
    --seed $DPO_SEED \
    --gradient_checkpointing \
    --fp16 \
    --continue_train

DPO_EXIT_CODE=$?
if [ $DPO_EXIT_CODE -ne 0 ]; then
    echo "DPO训练失败，退出码: $DPO_EXIT_CODE"
    exit 1
fi

# echo "DPO训练完成！"
# echo "DPO模型保存在: $DPO_OUTPUT_DIR"
# echo ""
# echo "课程学习完成！"