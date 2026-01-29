#!/bin/bash
# Azy-Memory 演示脚本 (参数硬编码版)
# 请直接修改下方路径后运行: ./run_demo.sh

# ==== 请在此处修改您的路径 ====
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
COMPRESS_PATH="jinandao/cog_mem_compress_conversation_dpo_lora"
QUERY_PATH="jinandao/cog_mem_memory_query_lora"
CONVERSATION_MODEL_PATH="jinandao/cog_mem_conversation_use_memory_lora"
CONVERSATION_FILE="./Configs/demo2/conversation.json"
TEACH_FILE="./Configs/demo2/conversation_teach.json"
MEMORIES_FILE="./Configs/demo2/memories.json"
# ===============================

echo "启动 Azy-Memory 演示..."
python3 run_demo.py \
    --model_path "$MODEL_PATH" \
    --compress_model_path "$COMPRESS_PATH" \
    --query_model_path "$QUERY_PATH" \
    --conversation_model_path "$CONVERSATION_MODEL_PATH" \
    --conversation_path "$CONVERSATION_FILE" \
    --conversation_teach_path "$TEACH_FILE" \
    --memories_path "$MEMORIES_FILE"