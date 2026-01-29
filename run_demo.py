import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import transformers
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import PeftModel
import json
from datetime import datetime
import re
import argparse


def load_models(model_path, compress_model_path, query_model_path, conversation_model_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True
    )
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,  # 启用8bit量化
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.float16,  # 计算时使用float16
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation='sdpa'
    )
    model = PeftModel.from_pretrained(base_model, compress_model_path, adapter_name="compress")
    model.load_adapter(query_model_path, adapter_name="query")
    model.load_adapter(conversation_model_path, adapter_name="conversation")
    return model, tokenizer

def load_data(conversation_path, conversation_teach_path, memories_path):
    # 加载 conversation.json
    with open(conversation_path, 'r', encoding='utf-8') as f:
        conversation_data = json.load(f)
    # 加载 conversation_teach.json
    with open(conversation_teach_path, 'r', encoding='utf-8') as f:
        conversation_teach_data = json.load(f)
    # 加载 memories.json
    with open(memories_path, 'r', encoding='utf-8') as f:
        memories_data = json.load(f)
    return conversation_data, conversation_teach_data, memories_data


def compress_data(compress_model, conversation):
    model.set_adapter("compress")
    conversations = conversation['conversation']
    whole_str = f"<|im_start|>system\n你是一名AI助手，擅长总结压缩对话，你需要在<think></think>块里提取对话关键点并梳理，然后在<memory></memory>里做最后总结"
    for i in range(len(conversations)):
        if conversations[i]['role'] == 'user':
            cur_input_str = "<|im_end|>\n<|im_start|>user\n" + conversations[i][
                'content'] + "<|im_end|>\n<|im_start|>assistant\n"
        else:
            cur_input_str = conversations[i]['content']
        whole_str += cur_input_str
    inputs = tokenizer(whole_str, add_special_tokens=False, return_tensors='pt').to(compress_model.device)
    outputs = compress_model.generate(**inputs, max_new_tokens=1024,
                             do_sample=False,
                             pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=tokenizer.eos_token_id, )
    response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    whole_str += response
    return response

def get_azy_timestamp():
    """生成符合 Azy-Memory 系统标准的格式化时间戳"""
    return datetime.now().strftime("%Y-%m-%d-%H:%M")

def parse_memory_id(text):
    tags = ['related_memories', 'low_related_memories']
    results = {}
    for tag in tags:
        # 1. 提取标签中间的内容
        match = re.search(f'<{tag}>(.*?)</{tag}>', text, re.S)
        if match:
            content = match.group(1).strip()
            # 2. 判定是否为“None”字符串
            if content.lower() == 'none' or not content:
                results[tag] = []
                continue
            # 3. 尝试解析为数组
            try:
                # 使用 json.loads 将 "[5]" 转换成 [5]
                data = json.loads(content)
                if isinstance(data, list):
                    # 确保只取数字（过滤掉非数字项）
                    results[tag] = [int(x) for x in data if isinstance(x, (int, float))]
                else:
                    results[tag] = []
            except (json.JSONDecodeError, ValueError):
                # 如果不是标准 JSON 格式，尝试用正则暴力抓取数字（容错处理）
                nums = re.findall(r'\d+', content)
                results[tag] = [int(n) for n in nums]
        else:
            results[tag] = []
    return results

def query_data(model, memories, query):
    model.set_adapter("query")
    whole_str = f"<|im_start|>system\n你是一名AI助手，擅长根据query查找memories里的相关记忆 ，你需要在<think></think>块里梳理每一条记忆和query是否有关，然后在<related_memories></related_memories>里指出高相关记忆，在<low_related_memories></low_related_memories>里指出低相关记忆"
    for i in range(len(memories)):
        memory_item = memories[i]
        memory_time = memory_item["time"]
        memory_id = memory_item["mem_id"]
        memory_content = memory_item["memory"]
        cur_input_str = "<|im_start|>memory\nid:" + str(memory_id) + "\ntime:" + str(memory_time) + "\ncontent:" + str(
            memory_content) + "<|im_end|>"
        whole_str += cur_input_str

    query_str = "<|im_start|>query\n" + query + "<|im_end|>"
    whole_str += query_str
    inputs = tokenizer(whole_str, add_special_tokens=False, return_tensors='pt').to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1024,
                             do_sample=False,
                             pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=tokenizer.eos_token_id, )
    response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    results = parse_memory_id(response)
    memory_str = "相关记忆片段："
    if len(results['related_memories']) > 0:
        related_memories_ids = results['related_memories']
        for id in related_memories_ids:
            memory = memories[id - 1]
            mem_item_str = "[mem - id: " + str(id) + "]时间：" + memory["time"] + "，内容：" + memory["memory"]
            memory_str += mem_item_str
    if len(results['low_related_memories']) > 0:
        memory_str = "低相关记忆片段："
        low_related_memories_ids = results['low_related_memories']
        for id in low_related_memories_ids:
            memory = memories[id - 1]
            low_mem_item_str = "[mem - id: " + str(id) + "]时间：" + memory["time"] + "，内容：" + memory["memory"]
            memory_str += low_mem_item_str
    # print("memory_str:", memory_str)
    return memory_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='运行记忆增强LLM Demo')
    parser.add_argument('--model_path', type=str, required=True, help='基础模型路径')
    parser.add_argument('--compress_model_path', type=str, required=True, help='对话压缩模型路径')
    parser.add_argument('--query_model_path', type=str, required=True, help='记忆查询模型路径')
    parser.add_argument('--conversation_model_path', type=str, required=True, help='对话生成模型路径')
    parser.add_argument('--conversation_path', type=str, required=True, help='对话文件路径')
    parser.add_argument('--conversation_teach_path', type=str, required=True, help='教学对话文件路径')
    parser.add_argument('--memories_path', type=str, required=True, help='记忆库文件路径')

    args = parser.parse_args()

    model, tokenizer = load_models(args.model_path, args.compress_model_path, args.query_model_path, args.conversation_model_path)
    conversation_data, conversation_teach_data, memories_data = load_data(args.conversation_path, args.conversation_teach_path, args.memories_path)
    memories = memories_data["memories"]

    # 压缩对话提取记忆
    add_memory = compress_data(model, conversation_teach_data)
    print("压缩记忆：", add_memory)
    print("------------------------------")
    match = re.search(r'<memory>(.*?)</memory>', add_memory, re.S)
    memory_content = ""
    if match:
        memory_content = match.group(1).strip()

    # 将记忆加入记忆库
    time_str = get_azy_timestamp()
    new_memory = {"mem_id": len(memories) + 1, "time": time_str, "memory": memory_content}
    memories.append(new_memory)
    print("所有记忆数据：")
    for memory in memories:
        print(memory)
    print("------------------------------")

    messages = conversation_data['conversations']
    whole_str = f"<|im_start|>system\n你是一个AI助手，现在在和用户聊天，当用户说起之前的事情时，你需要通过调用memory_query_call函数并传入查询query来查找记忆，然后根据memory_query角色返回的记忆，在<think></think>块内思考记忆片段的信息，再根据整理的信息做正确生成。如果用户没有提及以前的事情，就根据当前语境做正确生成"
    for i in range(len(messages)):
        if messages[i]['role'] == 'user':
            cur_input_str = "<|im_end|>\n<|im_start|>user\n" + messages[i][
                'content'] + "<|im_end|>\n<|im_start|>assistant\n"
            whole_str += cur_input_str
        elif messages[i]['role'] == 'memory_query':
            cur_input_str = "<|im_end|>\n<|im_start|>memory_query\n" + messages[i][
                'content'] + "<|im_end|>\n<|im_start|>assistant\n"
            whole_str += cur_input_str
        else:
            cur_input_str = ""
            if 'think' in messages[i]:
                cur_input_str += "<think>" + messages[i]['think'] + "</think>"
            cur_input_str += messages[i]['content']
            whole_str += cur_input_str
    print("原始完整对话:", whole_str)
    print("------------------------------")
    print("生成对话中")
    # 生成对话
    messages = conversation_data['conversations']
    whole_str = f"<|im_start|>system\n你是一个AI助手，现在在和用户聊天，当用户说起之前的事情时，你需要通过调用memory_query_call函数并传入查询query来查找记忆，然后根据memory_query角色返回的记忆，在<think></think>块内思考记忆片段的信息，再根据整理的信息做正确生成。如果用户没有提及以前的事情，就根据当前语境做正确生成"
    for i in range(len(messages)):
        if messages[i]['role'] == 'user':
            cur_input_str = "<|im_end|>\n<|im_start|>user\n" + messages[i][
                'content'] + "<|im_end|>\n<|im_start|>assistant\n"
            whole_str += cur_input_str
            inputs = tokenizer(whole_str, add_special_tokens=False, return_tensors='pt').to(model.device)
            model.set_adapter("conversation")
            outputs = model.generate(**inputs,
                                     max_new_tokens=384,
                                     do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id,
                                     eos_token_id=tokenizer.eos_token_id, )
            response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
            whole_str += response

            # 1. 提取 function 的类型 (例如: memory_query_call)
            func_match = re.search(r'<function>(.*?)</function>', response)
            # 2. 提取 content 的内容 (例如: 艾泽拉斯世界的速度定律)
            cont_match = re.search(r'<content>(.*?)</content>', response)
            if func_match and cont_match:
                func_type = func_match.group(1).strip()
                query_content = cont_match.group(1).strip()

                print(f"执行操作: {func_type}")
                print(f"搜索内容: {query_content}")
                if func_type == "memory_query_call":
                    related_memories = query_data(model, memories, query_content)
                    print("检索记忆：", related_memories, " 相关query:", query_content)
                    print("------------------------------")
                    cur_input_str = "<|im_end|>\n<|im_start|>memory_query\n" + related_memories + "<|im_end|>\n<|im_start|>assistant\n"
                    whole_str += cur_input_str
                    inputs = tokenizer(whole_str, add_special_tokens=False, return_tensors='pt').to(model.device)
                    model.set_adapter("conversation")
                    outputs = model.generate(**inputs,
                                             max_new_tokens=384,
                                             do_sample=False,
                                             pad_token_id=tokenizer.pad_token_id,
                                             eos_token_id=tokenizer.eos_token_id, )
                    response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
                    whole_str += response
    print("完整生成对话:", whole_str)

