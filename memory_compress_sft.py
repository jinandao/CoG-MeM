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
    conversations = example['conversation']
    input_str = f"<|im_start|>system\n你是一名AI助手，擅长总结压缩对话，你需要在<think></think>块里提取对话关键点并梳理，然后在<memory></memory>里做最后总结"
    input_str_ids = tokenizer(input_str, add_special_tokens=False)
    input_ids = []
    input_ids.extend(input_str_ids["input_ids"])
    attention_mask = []
    attention_mask.extend(input_str_ids["attention_mask"])
    labels = []
    labels.extend([-100] * len(input_str_ids["input_ids"]))
    for i in range(len(conversations)):
        if conversations[i]['role'] == 'user':
            cur_input_str = "<|im_end|>\n<|im_start|>user\n" + conversations[i]['content'] + "<|im_end|>\n<|im_start|>assistant\n"
        else:
            cur_input_str = conversations[i]['content']
        cur_input_ids = tokenizer(cur_input_str, add_special_tokens=False)
        input_ids.extend(cur_input_ids['input_ids'])
        attention_mask.extend(cur_input_ids['attention_mask'])
        labels.extend([-100] * len(cur_input_ids['input_ids']))
        input_str += cur_input_str
    think_str = example['think']
    memory_str = example['memory']
    label_str = "总结：<think>" + think_str + "</think><memory>" + memory_str + "</memory>" + tokenizer.eos_token

    input_str += label_str
    cur_input_ids = tokenizer(label_str, add_special_tokens=False)
    input_ids.extend(cur_input_ids['input_ids'])
    attention_mask.extend(cur_input_ids['attention_mask'])
    labels.extend(cur_input_ids['input_ids'])

    input_ids = (input_ids)
    attention_mask = attention_mask
    labels = (labels)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "length": len(input_ids)}

def filter_by_length(example):
    """过滤掉长度大于4096的样本"""
    return example["length"] <= 4096

def predict(example, model, tokenizer):
    # assert False
    conversations = example['conversation']
    whole_str = f"<|im_start|>system\n你是一名AI助手，擅长总结压缩对话，你需要在<think></think>块里提取对话关键点并梳理，然后在<memory></memory>里做最后总结"
    for i in range(len(conversations)):
        if conversations[i]['role'] == 'user':
            cur_input_str = "<|im_end|>\n<|im_start|>user\n" + conversations[i]['content'] + "<|im_end|>\n<|im_start|>assistant\n"
        else:
            cur_input_str = conversations[i]['content']
        whole_str += cur_input_str
    inputs = tokenizer(whole_str, add_special_tokens=False, return_tensors='pt').to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1024,
                                    temperature=0.1,
                                    do_sample=True,
                                    pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id,)
    response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    whole_str += response
    print("whole_str:", whole_str)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Memory Compression SFT Training Script")

    # 数据集路径
    parser.add_argument("--train_json_path", type=str, required=True,
                        help="训练集JSON文件路径")

    # 模型路径
    parser.add_argument("--model_dir", type=str, required=True,
                        help="预训练模型路径")

    # 输出目录
    parser.add_argument("--output_dir", type=str, required=True,
                        help="SFT模型输出目录")

    # 训练参数
    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                        help="每个设备的训练批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="梯度累积步数")
    parser.add_argument("--num_train_epochs", type=int, default=4,
                        help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="学习率")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="是否使用梯度检查点")
    parser.add_argument("--seed", type=int, default=145,
                        help="随机种子")

    return parser.parse_args()

if __name__ == "__main__":
    # train_json_path = r"./Datasets/compress_conversation/compress_conversation_sft.json"
    # model_dir = r"F:\Others\models\Qwen2.5-7B-Instruct"
    # output_dir = r"./Output/Compress_Conversation/Conversation_Compress_SFT"
    # per_device_train_batch_size = 2
    # gradient_accumulation_steps = 1
    # num_train_epochs = 4
    # learning_rate = 2e-5
    # gradient_checkpointing = True

    args = parse_args()

    # 使用解析的参数
    train_json_path = args.train_json_path
    model_dir = args.model_dir
    output_dir = args.output_dir
    per_device_train_batch_size = args.per_device_train_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    num_train_epochs = args.num_train_epochs
    learning_rate = args.learning_rate
    gradient_checkpointing = args.gradient_checkpointing
    seed = args.seed

    train_df = pd.read_json(train_json_path)
    dataset = Dataset.from_pandas(train_df)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=seed, shuffle=True)
    train_dataset, test_dataset = split_dataset['train'], split_dataset['test']

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, use_fast=False, trust_remote_code=True
    )
    process_func = partial(process_func, tokenizer=tokenizer)
    train_dataset = train_dataset.map(process_func, num_proc=1)
    # train_dataset = train_dataset.map(process_func_no_COT, num_proc=1)

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
    print("load OK")

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=10,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        save_on_each_node=True,
        gradient_checkpointing=gradient_checkpointing,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    print("begin train")
    trainer.train()

    final_model_dir = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_model_dir)

    # 从测试集取前20条数据
    model.eval()
    print("begin test")
    test_samples = min(10, len(test_dataset))
    for i in range(test_samples):
        example = test_dataset[i]
        predict(example, model, tokenizer)
    print("test end")