import json
from functools import partial

import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
import os
os.environ["WANDB_DISABLED"] = "true"
import argparse

def process_func(example, tokenizer):
    """
    将数据集进行预处理
    """
    # print(example)
    memories = example['memories']
    input_str = f"<|im_start|>system\n你是一名AI助手，擅长根据query查找memories里的相关记忆，你需要在<related_memories></related_memories>里指出高相关记忆，在<low_related_memories></low_related_memories>里指出低相关记忆"
    input_str_ids = tokenizer(input_str, add_special_tokens=False)
    input_ids = []
    input_ids.extend(input_str_ids["input_ids"])
    attention_mask = []
    attention_mask.extend(input_str_ids["attention_mask"])
    labels = []
    labels.extend([-100] * len(input_str_ids["input_ids"]))
    for i in range(len(memories)):
        memory_item = memories[i]
        memory_time = memory_item["time"]
        memory_id = memory_item["mem_id"]
        memory_content = memory_item["memory"]
        cur_input_str = "<|im_start|>memory\nid:" + str(memory_id) + "\ntime:" + str(memory_time) + "\ncontent:" + str(memory_content) + "<|im_end|>"
        cur_input_ids = tokenizer(cur_input_str, add_special_tokens=False)
        input_ids.extend(cur_input_ids['input_ids'])
        attention_mask.extend(cur_input_ids['attention_mask'])
        labels.extend([-100] * len(cur_input_ids['input_ids']))
        input_str += cur_input_str

    query = example['query']
    query_str = "<|im_start|>query\n" + query + "<|im_end|>"
    query_ids = tokenizer(query_str, add_special_tokens=False)
    input_ids.extend(query_ids['input_ids'])
    attention_mask.extend(query_ids['attention_mask'])
    labels.extend([-100] * len(query_ids['input_ids']))
    input_str += query_str

    related_memories = example['related_memories']
    low_related_memories = example['low_related_memories']
    final_memories_str = "<related_memories>" + str(related_memories) + "</related_memories><low_related_memories>" + str(low_related_memories) + "</low_related_memories><|im_end|>"
    final_memories_ids = tokenizer(final_memories_str, add_special_tokens=False)
    input_ids.extend(final_memories_ids['input_ids'])
    attention_mask.extend(final_memories_ids['attention_mask'])
    labels.extend(final_memories_ids['input_ids'])
    input_str += final_memories_str

    input_ids = (input_ids)
    attention_mask = attention_mask
    labels = (labels)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "length": len(input_ids)}

def filter_by_length(example):
    """过滤掉长度大于4096的样本"""
    return example["length"] <= 4096

def predict(example, model, tokenizer):
    memories = example['memories']
    whole_str = f"<|im_start|>system\n你是一名AI助手，擅长根据query查找memories里的相关记忆 ，你需要在<related_memories></related_memories>里指出高相关记忆，在<low_related_memories></low_related_memories>里指出低相关记忆"
    for i in range(len(memories)):
        memory_item = memories[i]
        memory_time = memory_item["time"]
        memory_id = memory_item["mem_id"]
        memory_content = memory_item["memory"]
        cur_input_str = "<|im_start|>memory\nid:" + str(memory_id) + "\ntime:" + str(memory_time) + "\ncontent:" + str(
            memory_content) + "<|im_end|>"
        whole_str += cur_input_str

    query = example['query']
    query_str = "<|im_start|>query\n" + query + "<|im_end|>"
    whole_str += query_str
    inputs = tokenizer(whole_str, add_special_tokens=False, return_tensors='pt').to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1024,
                                    do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id,)
    response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    print("记忆和query：", whole_str)
    print("模型输出:", response)
    related_memories = example['related_memories']
    low_related_memories = example['low_related_memories']
    print("正确答案 related_memories：", related_memories, "low_related_memories:", low_related_memories)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Memory Query SFT Training Script")

    # 模型路径
    parser.add_argument("--model_dir", type=str, required=True,
                        help="预训练模型路径")
    # 数据集路径
    parser.add_argument("--train_json_path", type=str, required=True,
                        help="训练集JSON文件路径")
    parser.add_argument("--test_json_path", type=str, required=True,
                        help="测试集JSON文件路径")

    # 输出目录
    parser.add_argument("--output_dir", type=str, default="./Output/Memory_Query",
                        help="模型输出目录")

    # 训练参数
    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                        help="每个设备的训练批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="梯度累积步数")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="日志记录步数")
    parser.add_argument("--num_train_epochs", type=int, default=5,
                        help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="学习率")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="是否使用梯度检查点")

    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    # 使用解析的参数
    model_dir = args.model_dir
    train_json_path = args.train_json_path
    test_json_path = args.test_json_path
    output_dir = args.output_dir
    per_device_train_batch_size = args.per_device_train_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    logging_steps = args.logging_steps
    num_train_epochs = args.num_train_epochs
    learning_rate = args.learning_rate
    gradient_checkpointing = args.gradient_checkpointing

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, use_fast=False, trust_remote_code=True
    )
    train_df = pd.read_json(train_json_path)
    train_dataset = Dataset.from_pandas(train_df)
    train_dataset.shuffle()
    process_func = partial(process_func, tokenizer=tokenizer)
    train_dataset = train_dataset.map(process_func, num_proc=1)

    test_df = pd.read_json(test_json_path)
    test_dataset = Dataset.from_pandas(test_df)

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,  # 启用8bit量化
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.float16,  # 计算时使用float16
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation='sdpa'
    )
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        inference_mode=False,  # 训练模式
        r=8,  # Lora 秩
        lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1,  # Dropout 比例
    )
    model = get_peft_model(model, config)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        gradient_checkpointing=gradient_checkpointing,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    print("begin train")
    trainer.train()

    # 从测试集取前20条数据
    model.eval()
    print("begin test")
    # test_samples = min(20, len(test_dataset))
    for i in range(len(test_dataset)):
        example = test_dataset[i]
        predict(example, model, tokenizer)
        # predict_no_COT(example, model, tokenizer)
    print("test end")