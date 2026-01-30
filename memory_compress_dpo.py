import pandas as pd
from datasets import Dataset
from functools import partial
import os
os.environ["WANDB_DISABLED"] = "true"
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from trl import DPOTrainer, DPOConfig
import torch
import random
import argparse

def preprocess_dpo_data(example, tokenizer):

    prompt_input_str = f"<|im_start|>system\n你是一名AI助手，擅长总结压缩对话，你需要在<think></think>块里提取对话关键点并梳理，然后在<memory></memory>里做最后总结"
    conversations = example['conversation']
    for i in range(len(conversations)):
        if conversations[i]['role'] == 'user':
            cur_input_str = "<|im_start|>user\n" + conversations[i][
                'content'] + "<|im_end|>\n"
        else:
            cur_input_str = "<|im_start|>assistant\n" + conversations[i]['content'] + "<|im_end|>\n"
        prompt_input_str += cur_input_str
    prompt_input_str_ids = tokenizer(prompt_input_str, add_special_tokens=False)
    chosen_part = example['chosen']
    chosen_think_str = chosen_part['think']
    chosen_memory_str = chosen_part['memory']
    chosen_label_str = prompt_input_str + "总结：<think>" + chosen_think_str + "</think><memory>" + chosen_memory_str + "</memory>" + tokenizer.eos_token
    ret_chosen_label_str = "总结：<think>" + chosen_think_str + "</think><memory>" + chosen_memory_str + "</memory>" + tokenizer.eos_token
    chosen_tokens = tokenizer(chosen_label_str, add_special_tokens=False)
    rejected_part = example['rejected']
    rejected_think_str = rejected_part['think']
    rejected_memory_str = rejected_part['memory']
    rejected_label_str = prompt_input_str + "总结：<think>" + rejected_think_str + "</think><memory>" + rejected_memory_str + "</memory>" + tokenizer.eos_token
    ret_rejected_label_str = "总结：<think>" + rejected_think_str + "</think><memory>" + rejected_memory_str + "</memory>" + tokenizer.eos_token
    rejected_tokens = tokenizer(rejected_label_str, add_special_tokens=False)
    return {
        'prompt': prompt_input_str,
        'chosen': ret_chosen_label_str,
        'rejected': ret_rejected_label_str,
    }

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
    parser = argparse.ArgumentParser(description="Memory Compression DPO Training Script")

    # 模型路径
    parser.add_argument("--model_dir", type=str, required=True,
                        help="预训练模型路径")

    # 数据集路径
    parser.add_argument("--train_json_path", type=str, required=True,
                        help="训练集JSON文件路径")

    # SFT模型路径
    parser.add_argument("--sft_lora_path", type=str, required=True,
                        help="SFT训练的LoRA模型路径")

    # 训练参数
    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                        help="每个设备的训练批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-6,
                        help="学习率")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="DPO模型输出目录")
    parser.add_argument("--beta", type=float, default=0.2,
                        help="DPO beta参数")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="最大序列长度")
    parser.add_argument("--max_prompt_length", type=int, default=2560,
                        help="最大prompt长度")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="是否使用梯度检查点")
    parser.add_argument("--fp16", action="store_true",
                        help="是否使用FP16混合精度训练")
    parser.add_argument("--max_grad_norm", type=float, default=0.1,
                        help="最大梯度范数")
    parser.add_argument("--remove_unused_columns", action="store_true",
                        help="是否移除未使用的列")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--seed", type=int, default=10,
                        help="随机种子")
    parser.add_argument("--continue_train", action="store_true",
                        help="是否从SFT模型继续训练")

    return parser.parse_args()

if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()

    # 使用解析的参数
    model_dir = args.model_dir
    train_json_path = args.train_json_path
    sft_lora_path = args.sft_lora_path
    per_device_train_batch_size = args.per_device_train_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    learning_rate = args.learning_rate
    output_dir = args.output_dir
    beta = args.beta
    max_length = args.max_length
    max_prompt_length = args.max_prompt_length
    gradient_checkpointing = args.gradient_checkpointing
    fp16 = args.fp16
    max_grad_norm = args.max_grad_norm
    remove_unused_columns = args.remove_unused_columns
    num_train_epochs = args.num_train_epochs
    seed = args.seed
    continue_train = args.continue_train

    train_df = pd.read_json(train_json_path)
    dataset = Dataset.from_pandas(train_df)

    split_dataset = dataset.train_test_split(test_size=0.1, seed=seed, shuffle=True)
    train_dataset, test_dataset = split_dataset['train'], split_dataset['test']

    # Transformers加载模型权重
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, use_fast=False, trust_remote_code=True
    )

    # 可选：使用量化加载模型以节省显存
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

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation='sdpa'
    )

    if continue_train:
        lora_model_path = sft_lora_path
        model = PeftModel.from_pretrained(model, lora_model_path, is_trainable=True)
        ref_model = PeftModel.from_pretrained(ref_model, lora_model_path, is_trainable=False)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
    if not continue_train:
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

    preprocess_dpo_data = partial(preprocess_dpo_data, tokenizer=tokenizer)
    train_dataset = train_dataset.map(preprocess_dpo_data, remove_columns=dataset.column_names)
    print(train_dataset)

    args = DPOConfig(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,  # 梯度累积步数
        learning_rate=learning_rate,
        optim='rmsprop',
        report_to='none',
        output_dir=output_dir,
        beta=beta,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        gradient_checkpointing=gradient_checkpointing,
        fp16=fp16,  # 使用混合精度
        max_grad_norm=max_grad_norm,  # 梯度裁剪
        remove_unused_columns=remove_unused_columns,  # 重要：不要自动移除列
        num_train_epochs=num_train_epochs,
    )

    # 然后继续创建DPOTrainer，其他部分不变
    dpo_trainer = DPOTrainer(model,
                         ref_model,
                         args=args,
                         train_dataset=train_dataset,
                         processing_class=tokenizer)

    print("开始DPO训练...")
    dpo_trainer.train()

    # 从测试集取前20条数据
    model.eval()
    print("begin test")
    for i in range(len(test_dataset)):
        example = test_dataset[i]
        predict(example, model, tokenizer)
    print("test end")