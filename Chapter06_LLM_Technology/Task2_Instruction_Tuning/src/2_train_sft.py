import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, Subset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

# --- 1. 数据集定义 ---
class MessagesDataset(Dataset):
    """读取标准化消息格式的数据集 [cite: 226, 230]"""
    def __init__(self, path: str):
        self.items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.items.append(json.loads(line))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]["messages"]

# --- 2. 核心：Assistant-Only Loss 数据整理器 ---
@dataclass
class DataCollatorAssistantOnly:
    """
    实现损失掩码机制：仅对 assistant 回复部分计算交叉熵损失 [cite: 213, 215, 247]
    """
    tokenizer: Any
    max_length: int = 1024

    def __call__(self, batch_messages: List[List[Dict[str, str]]]):
        input_ids_list = []
        labels_list = []
        attn_list = []

        for messages in batch_messages:
            # 使用聊天模板渲染完整对话 [cite: 204, 206]
            full_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            # 渲染不含答案的前缀部分，用于定位答案起始位置 [cite: 252, 253]
            msgs_prefix = messages[:-1] 
            prefix_text = self.tokenizer.apply_chat_template(
                msgs_prefix, tokenize=False, add_generation_prompt=True
            )

            full = self.tokenizer(full_text, truncation=True, max_length=self.max_length, padding=False)
            prefix = self.tokenizer(prefix_text, truncation=True, max_length=self.max_length, padding=False)

            input_ids = full["input_ids"]
            attention_mask = full["attention_mask"]
            prefix_len = len(prefix["input_ids"])

            # 构造标签：将前缀部分（System/User）设为 -100 以忽略损失计算 [cite: 247, 255]
            labels = list(input_ids)
            for i in range(min(prefix_len, len(labels))):
                labels[i] = -100
            
            # 检查是否有有效的标签（至少有一个非 -100 的标签）
            if all(l == -100 for l in labels):
                # 如果所有标签都被掩码，至少保留最后一个 token 用于计算损失
                if len(labels) > 0:
                    labels[-1] = input_ids[-1]

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attn_list.append(attention_mask)

        # Padding 处理 [cite: 254]
        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(x) for x in input_ids_list)
        
        def pad(seq, pad_value):
            return seq + [pad_value] * (max_len - len(seq))

        return {
            "input_ids": torch.tensor([pad(x, pad_id) for x in input_ids_list], dtype=torch.long),
            "attention_mask": torch.tensor([pad(x, 0) for x in attn_list], dtype=torch.long),
            "labels": torch.tensor([pad(x, -100) for x in labels_list], dtype=torch.long),
        }

# --- 3. 训练主流程 ---
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--data", default="data/train_messages.jsonl")
    p.add_argument("--out", default="outputs/sft_lora")
    p.add_argument("--epochs", type=int, default=1)
    args = p.parse_args()

    # 加载分词器 [cite: 304]
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型并应用显存优化技术 [cite: 197, 269]
    print(f">>> 正在加载基座模型...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    
    # 优化2：开启梯度检查点以节省显存峰值 [cite: 269]
    model.gradient_checkpointing_enable()

    # 配置 LoRA 参数高效微调 [cite: 271, 308, 311]
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # 准备数据集并切分验证集用于评测 
    full_dataset = MessagesDataset(args.data)
    num_val = min(int(len(full_dataset) * 0.05), 50)  # 最多取50条作为验证集，节省显存
    train_ds = Subset(full_dataset, range(num_val, len(full_dataset)))
    eval_ds = Subset(full_dataset, range(num_val)) if num_val > 0 else None
    
    collator = DataCollatorAssistantOnly(tokenizer=tokenizer, max_length=512)  # 减小最大长度节省显存

    # 训练超参数配置 [cite: 265, 268]
    train_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,  # 评估时使用更小的 batch size
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_steps=100,
        bf16=torch.cuda.is_available(),
        report_to="none",
        dataloader_pin_memory=False,  # 禁用 pin_memory 节省显存
        max_grad_norm=1.0,  # 梯度裁剪，防止梯度爆炸
        warmup_steps=50,  # 学习率预热，提高训练稳定性
    )

    # 实例化 Trainer [cite: 234, 256]
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # 执行训练 [cite: 218]
    print(">>> 开始监督微调（SFT）...")
    trainer.train()

    # 性能评估：计算验证集困惑度 (PPL) [cite: 273]
    if eval_ds is not None:
        print(">>> 正在执行最终评估...")
        # 清理显存
        torch.cuda.empty_cache()
        try:
            metrics = trainer.evaluate()
            eval_loss = metrics.get("eval_loss", float("nan"))
            
            # 检查损失是否为有效值
            if math.isnan(eval_loss) or math.isinf(eval_loss):
                print("⚠️  评估损失为 NaN/Inf，可能是训练过程中出现了数值不稳定")
                print("   建议：检查训练日志，降低学习率，或检查数据格式")
            else:
                try:
                    ppl = math.exp(eval_loss)
                    print(f"📊 训练完成！验证集 Loss: {eval_loss:.4f}, 困惑度 (PPL): {ppl:.2f}")
                except OverflowError:
                    print(f"📊 训练完成！验证集 Loss: {eval_loss:.4f}, 困惑度 (PPL): 溢出")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("⚠️  评估时显存不足，跳过最终评估")
            else:
                raise
    else:
        print("📊 训练完成！")

    # 保存权重与分词器 [cite: 318, 321]
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"✅ 模型与适配器已保存至: {args.out}")

if __name__ == "__main__":
    main()