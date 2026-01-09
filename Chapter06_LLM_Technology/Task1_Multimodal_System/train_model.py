import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

# 导入自定义模块
from dataset_loader import Flickr8kDataset
from model_core import load_clip_model, ClipCaptionModel

# --- 1. 相对路径配置 ---
# 建议：在项目根目录下运行终端，这样 "./data" 才能被找到
IMG_DIR = "./data/Flicker8k_Dataset"
TOKEN_FILE = "./data/Flickr8k_text/Flickr8k.token.txt"
OUTPUT_DIR = "./checkpoints" 

# 训练超参数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 5              
BATCH_SIZE = 32         
LR = 2e-5               

# 确保保存目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

def train():
    print(f">>> [训练启动] 使用设备: {DEVICE}")
    
    # 检查路径是否存在，避免后续报错
    if not os.path.exists(IMG_DIR) or not os.path.exists(TOKEN_FILE):
        raise FileNotFoundError(f"❌ 找不到数据！请检查当前目录下是否有 data 文件夹。\n当前工作目录: {os.getcwd()}")

    # --- 2. 准备数据 ---
    dataset = Flickr8kDataset(
        img_dir=IMG_DIR,
        token_file=TOKEN_FILE
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # --- 3. 准备模型 ---
    print(">>> 正在加载 CLIP (Fixed)...")
    clip_model = load_clip_model(DEVICE)
    
    print(">>> 正在初始化生成模型 (Trainable)...")
    model = ClipCaptionModel().to(DEVICE)
    model.train() 
    
    # --- 4. 优化器设置 ---
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=EPOCHS * len(dataloader)
    )
    
    # 忽略 padding token 的 Loss
    ignore_idx = dataset.processor.tokenizer.pad_token_id
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_idx)

    print(">>> 开始训练循环...")
    
    # --- 5. 训练循环 ---
    for epoch in range(EPOCHS):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        total_loss = 0
        
        for step, batch in enumerate(progress_bar):
            model.zero_grad()
            
            pixel_values = batch['pixel_values'].to(DEVICE) 
            input_ids = batch['input_ids'].to(DEVICE)       
            
            # A. 提取视觉特征 (冻结参数)
            with torch.no_grad():
                img_embeds = clip_model.get_image_features(pixel_values)
                img_embeds = img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)
                img_embeds = img_embeds.float()

            # B. 前向传播
            # model 返回 CausalLMOutput; 取 logits
            logits = model(img_embeds, input_ids).logits
            # 去掉前缀部分，仅保留文本 token 的 logits，避免维度不对齐
            logits = logits[:, model.prefix_length:, :]
            
            # C. 计算损失 (错位预测)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # --- 6. 保存权重 ---
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} 完成 | 平均 Loss: {avg_loss:.4f}")
        
        # 保存到 checkpoints 文件夹
        save_path = os.path.join(OUTPUT_DIR, f"caption_model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"模型已保存至: {save_path}")

if __name__ == "__main__":
    train()