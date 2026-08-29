# -*- coding: utf-8 -*-
"""
LLM 微调：SFT / LoRA / RLHF —— 典型代码示例
=============================================
覆盖知识点：
  1. SFT 的"指令-回答"数据准备与基本流程
  2. LoRA：用 peft 库把大模型冻结 + 只训低秩适配器
  3. training args：学习率 / warmup / batch 等关键配置
  4. 对比全量微调 vs LoRA 的可训练参数量级
  5. 简单说明 RLHF 概念（此处不实际训练，避免依赖）

依赖：pip install torch transformers peft datasets
"""

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

# =====================================================================
# 一、模拟少量"指令-回答"训练数据（真实项目替换为你自己的 datasets）
# =====================================================================
tiny_data = [
    {"instruction": "中国的首都是哪个城市？", "output": "北京。"},
    {"instruction": "1 加 1 等于几？", "output": "2。"},
    {"instruction": "机器学习三大类任务是什么？", "output": "分类、回归、聚类。"},
]

# 用模板拼接成适合因果 LM 训练的完整文本
prompt_template = "### 指令:{instruction}\n### 回答:{output}"
texts = [prompt_template.format(**d) for d in tiny_data]
print("[SFT数据] 拼接后的训练样本示例:\n", texts[0], "\n")

# =====================================================================
# 二、准备 tokenizer 与模型（此处用很小的白盒模型演示结构）
# =====================================================================
def load_tiny_model():
    """装载一个小型因果模型。真实场景接入 LLaMA/Qwen/GPT 等。"""
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.for_model("gpt2", n_layer=2, n_head=2,
                                  n_embd=64, vocab_size=1000)
    return AutoModelForCausalLM.from_config(config)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_tiny_model().to(device)
print("[模型] 已装载; 全量参数 =", 
      sum(p.numel() for p in model.parameters()), "个")

# =====================================================================
# 三、LoRA：冻结原权重，只训练低秩适配器
# =====================================================================
lora_config = LoraConfig(
    r=8,                    # 低秩维度
    lora_alpha=16,          # 缩放系数（控制适配器强度）
    target_modules=["c_attn"],   # 作用于哪些模块（一般作用于注意力 QKV）
    lora_dropout=0.05,
)

lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()
# 会输出类似 :trainable params: X || 全量参数 Y || trainable% 约 0.xx%

print("\n[LoRA] 冻结了大部分参数，仅更新低秩 A/B 矩阵 → 显存与算力大幅下降")

# =====================================================================
# 四、训练配置（SFT 的标准参数）
# =====================================================================
training_args = TrainingArguments(
    output_dir="./lora_out",       # 输出目录
    num_train_epochs=1,            # 训练轮数
    per_device_train_batch_size=2, # 每设备批大小
    learning_rate=2e-4,            # LoRA 常用学习率（比全量微调高）
    warmup_ratio=0.1,              # 预热比例
    logging_steps=1,
    save_steps=50,
    remove_unused_columns=False,
    no_cuda=not torch.cuda.is_available(),
)
print("\n[训练参数] 学习率=2e-4, warmup=10%, batch=2 → LoRA 微调就绪")

# 说明：实际 start `Trainer.fit` 需要真实 dataset/tokenizer，
# 此处为了不依赖下载模型而省略真实训练循环。要点是：
#   数据用模板拼 -> tokenizer(vectorize) -> data collator 做随机掩码。

# =====================================================================
#    五、关于 RLHF（概念速览，不实际训练）
# =====================================================================
print("""
[RLHF 三阶段概览]
 ① SFT           : 有监督地教模型输出风格/指令跟随
 ② 奖励模型 RM   : 从人类偏好排序中训练打分器（谁能回答得更好）
 ③ 策略优化      : PPO/DPO 让模型朝"高奖励"优化，同时 KL 惩罚防止跑偏
[DPO vs PPO]
 PPO 需额外采样+训练奖励模型；DPO 直接把偏好目标写成监督目标，更轻量，
 是近年微调对齐的主流方案。
""")


# =====================================================================
# 六、把适配器保存与加载（LoRA 使用闭环）
# =====================================================================
# 保存时只存低秩适配器（很小），不存整份大模型
# lora_model.save_pretrained("./lora_adapter")
# 加载时：base 模型 + adapter 合并即可上线
from peft import PeftModel
def load_adapted(base_model, adapter_dir):
    """把训练好的 LoRA 适配器挂到基座模型上推理。"""
    return PeftModel.from_pretrained(base_model, adapter_dir)

print("\n[闭环] LoRA 适配器可独立分发/加载；推理时与基座合并，零额外性能开销")
