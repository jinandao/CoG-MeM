import json
from functools import partial

import pandas as pd
import torch
from datasets import Dataset
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
    messages = example['conversations']
    input_str = f"<|im_start|>system\n你是一个AI助手，现在在和用户聊天，当用户说起之前的事情时，你需要通过调用memory_query_call函数并传入查询query来查找记忆，然后根据memory_query角色返回的记忆，在<think></think>块内思考记忆片段的信息，再根据整理的信息做正确生成。如果用户没有提及以前的事情，就根据当前语境做正确生成"
    input_str_ids = tokenizer(input_str, add_special_tokens=False)
    input_ids = []
    input_ids.extend(input_str_ids["input_ids"])
    attention_mask = []
    attention_mask.extend(input_str_ids["attention_mask"])
    labels = []
    labels.extend([-100] * len(input_str_ids["input_ids"]))
    for i in range(len(messages)):
        # pass
        if messages[i]['role'] == 'user':
            cur_input_str = "<|im_end|>\n<|im_start|>user\n" + messages[i]['content'] + "<|im_end|>\n<|im_start|>assistant\n"
            cur_input_ids = tokenizer(cur_input_str, add_special_tokens=False)
            input_ids.extend(cur_input_ids['input_ids'])
            attention_mask.extend(cur_input_ids['attention_mask'])
            labels.extend([-100] * len(cur_input_ids['input_ids']))
        elif messages[i]['role'] == 'memory_query':
            cur_input_str = "<|im_end|>\n<|im_start|>memory_query\n" + messages[i]['content'] + "<|im_end|>\n<|im_start|>assistant\n"
            cur_input_ids = tokenizer(cur_input_str, add_special_tokens=False)
            input_ids.extend(cur_input_ids['input_ids'])
            attention_mask.extend(cur_input_ids['attention_mask'])
            labels.extend([-100] * len(cur_input_ids['input_ids']))
        else:
            cur_input_str = ""
            if 'think' in messages[i] and messages[i]['think'] is not None:
                cur_input_str += "<think>" + messages[i]['think'] + "</think>"
            cur_input_str += messages[i]['content'] + tokenizer.eos_token
            cur_input_ids = tokenizer(cur_input_str, add_special_tokens=False)
            input_ids.extend(cur_input_ids['input_ids'])
            attention_mask.extend(cur_input_ids['attention_mask'])
            labels.extend(cur_input_ids['input_ids'])
        input_str += cur_input_str
    input_ids = (input_ids)
    attention_mask = attention_mask
    labels = (labels)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "length": len(input_ids)}

def filter_by_length(example):
    """过滤掉长度大于1024的样本"""
    return example["length"] <= 4096

def predict(example, model, tokenizer):
    messages = example['conversations']
    whole_str = f"<|im_start|>system\n你是一个AI助手，现在在和用户聊天，当用户说起之前的事情时，你需要通过调用memory_query_call函数并传入查询query来查找记忆，然后根据memory_query角色返回的记忆，在<think></think>块内思考记忆片段的信息，再根据整理的信息做正确生成。如果用户没有提及以前的事情，就根据当前语境做正确生成"
    for i in range(len(messages)):
        if messages[i]['role'] == 'user':
            cur_input_str = "<|im_end|>\n<|im_start|>user\n" + messages[i]['content'] + "<|im_end|>\n<|im_start|>assistant\n"
            whole_str += cur_input_str
            inputs = tokenizer(whole_str, add_special_tokens=False, return_tensors='pt').to(model.device)
            outputs = model.generate(**inputs,
                                    max_new_tokens=384,
                                    temperature=0.1,
                                    do_sample=True,
                                    pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id,)
            response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
            whole_str += response
        elif messages[i]['role'] == 'memory_query':
            cur_input_str = "<|im_end|>\n<|im_start|>memory_query\n" + messages[i]['content'] + "<|im_end|>\n<|im_start|>assistant\n"
            whole_str += cur_input_str
            inputs = tokenizer(whole_str, add_special_tokens=False, return_tensors='pt').to(model.device)
            outputs = model.generate(**inputs,
                                     max_new_tokens=384,
                                     temperature=0.1,
                                     do_sample=True,
                                     pad_token_id=tokenizer.pad_token_id,
                                     eos_token_id=tokenizer.eos_token_id, )
            response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
            whole_str += response
    print("whole_str:", whole_str)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Memory Conversation SFT Training Script")

    # 数据集路径
    parser.add_argument("--train_json_path", type=str, required=True,
                        help="训练集JSON文件路径")

    # 模型路径
    parser.add_argument("--model_dir", type=str, required=True,
                        help="预训练模型路径")

    # 训练参数
    parser.add_argument("--seed", type=int, default=12345,
                        help="随机种子")
    parser.add_argument("--output_dir", type=str, default="./Output/Conversation_Use_Memory",
                        help="模型输出目录")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="每个设备的训练批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="梯度累积步数")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="日志记录步数")
    parser.add_argument("--num_train_epochs", type=int, default=20,
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
    train_json_path = args.train_json_path
    model_dir = args.model_dir
    seed = args.seed
    output_dir = args.output_dir
    per_device_train_batch_size = args.per_device_train_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    logging_steps = args.logging_steps
    num_train_epochs = args.num_train_epochs
    learning_rate = args.learning_rate
    gradient_checkpointing = args.gradient_checkpointing

    train_df = pd.read_json(train_json_path)
    dataset = Dataset.from_pandas(train_df)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=seed, shuffle=True)
    train_dataset, test_dataset = split_dataset['train'], split_dataset['test']

    # Transformers加载模型权重
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, use_fast=False, trust_remote_code=True
    )
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

    process_func = partial(process_func, tokenizer=tokenizer)
    train_dataset = train_dataset.map(process_func, remove_columns=train_dataset.column_names, num_proc=4)
    train_dataset = train_dataset.filter(filter_by_length, num_proc=4)  # 过滤掉长度大于1024的样本
    print(train_dataset)  # 查看数据集
    print(test_dataset)

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
    print("load OK")

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

    model.eval()
    print("begin test")
    test_samples = min(50, len(test_dataset))
    for i in range(test_samples):
        example = test_dataset[i]
        predict(example, model, tokenizer)
    print("test end")