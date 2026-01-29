# CoG-Mem: Context-to-Memory Compression and Reasoning

本项目致力于大语言模型（LLM）的记忆增强研究，涵盖了从**上下文记忆压缩**、**记忆筛选检索**到**基于记忆的对话生成**的全流程。项目特别引入了**课程学习 (Curriculum Learning)** 训练策略，并支持针对**非参数化知识推理**的快速演示。

---

## 📂 项目架构 (Project Structure)

```plaintext
CoG-Mem/
├── Configs/                # 实验配置文件夹，包含 run_demo 所需的配置文件
├── Datasets/               # 数据预处理脚本及样本数据集
├── Models/                 # 存放预训练/开源权重（用于直接运行 run_demo.sh）
├── Output/                 # 训练脚本生成的模型输出路径
├── memory_compress.sh      # 核心脚本：执行记忆压缩的课程学习流程（SFT -> DPO）
├── memory_compress_sft.py  # 记忆压缩的监督微调实现
├── memory_compress_dpo.py  # 记忆压缩的直接偏好优化实现
├── memory_query_sft.sh     # 脚本：复现端到端记忆筛选实验
├── memory_query_sft.py     # 记忆筛选与检索的微调实现
├── memory_conversation_sft.sh # 脚本：复现基于记忆的对话生成实验
├── memory_conversation_sft.py # 记忆对话生成的微调实现
├── run_demo.sh             # 脚本：非参数化学习效果演示
└── run_demo.py             # Demo 运行入口
```

## ⚙️ 环境准备与安装

### 创建 Python 3.10 的 Conda 环境
conda create -n cogmem python=3.10 -y
### 激活环境
conda activate cogmem
### 安装 PyTorch (GPU 版本)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
### 安装核心依赖
pip install -r requirements.txt


## 🚀训练流程与实验复现
本项目所有的训练流程均已封装在 .sh 脚本中，训练完成后会自动打印测试集输出，便于直接评估模型效果。

**1. 记忆压缩实验 (Memory Compression)**,
该实验采用课程学习 (Curriculum Learning) 策略：先通过 SFT 进行基础能力对齐，再通过 DPO 优化记忆压缩的质量。
```plaintext
   bash memory_compress.sh
```
预期效果：实验结束后将展示压缩对话后提取的记忆片段。

**2. 记忆筛选实验 (Memory Query)**，复现端到端的记忆筛选逻辑，验证模型在海量记忆中提取关键信息的能力。
```plaintext
   bash memory_query_sft.sh
```
预期效果：自动打印测试集中的LLM筛选结果与正确结果。

**3. 基于记忆的对话生成 (Memory-based Conversation)**，验证模型利用提取到的记忆进行下游对话生成的完整性。
```plaintext
   bash memory_conversation_sft.sh
```
预期效果：打印测试集中AI针对用户输入的对话生成结果。

## 🧪Demo 演示：非参数化学习 (Non-parametric Learning)
我们提供了一个快速演示脚本，用于查看LLM在面对**新知识（如修改后的公式）**时的表现。
```plaintext
   bash run_demo.sh
```
配置说明：内部配置了 demo1 和 demo2 路径，包含修改了公式逻辑的 JSON 数据。

演示功能：您可以查看 LLM 是否能理解并应用这些新定义的公式进行输出。

注意：用户可以微调公式数据，但由于目前 Demo 数据规模较小，微调后的泛化准确性不作绝对保证。

## 📬 联系方式 (Contact)
邮箱:u201312560@alumni.hust.edu.cn

**⚠️ 关于数据集 ID 的说明**：
本项目中的部分数据经过了手动标注与微调。由于数据处理过程中重点关注内容质量，部分样本的 `ID` 字段存在编号不连续或逻辑混乱的情况。
**请注意**：由于本项目的训练和评估逻辑（包括记忆压缩与检索）并不依赖 `ID` 字段进行索引，因此该问题不会对数据集的有效性、模型训练或实验结果的复现产生任何影响。