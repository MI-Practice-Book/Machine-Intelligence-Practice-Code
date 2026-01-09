import torch
import torch.nn as nn
from transformers import CLIPModel, GPT2LMHeadModel

# 1. 检索模型 (直接使用预训练 CLIP)
def load_clip_model(device):
    print("正在加载 CLIP 模型...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model.eval() # 检索通常只需推理模式
    return model

# 2. 生成模型 (CLIP + Mapping + GPT-2)
class ClipCaptionModel(nn.Module):
    def __init__(self, prefix_length=10):
        super().__init__()
        self.prefix_length = prefix_length
        
        # 加载 GPT-2
        print("正在加载 GPT-2 解码器...")
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        gpt_dim = self.gpt.config.n_embd # 768维
        clip_dim = 512                   # CLIP Base 输出维度
        
        # 核心组件：映射网络 (Mapping Network)
        # 将 CLIP 的视觉特征翻译成 GPT-2 的词嵌入空间
        self.clip_project = nn.Sequential(
            nn.Linear(clip_dim, gpt_dim * prefix_length),
            nn.Tanh()
        )
        
    def forward(self, img_embeds, token_ids=None):
        # img_embeds: [batch, 512] (来自 CLIP)
        
        # 1. 映射生成视觉前缀
        # [batch, 10, 768]
        prefixs = self.clip_project(img_embeds).view(-1, self.prefix_length, self.gpt.config.n_embd)
        
        # 2. 如果提供了文本 (训练阶段)，则拼接
        if token_ids is not None:
            text_embeds = self.gpt.transformer.wte(token_ids)
            embedding_cat = torch.cat((prefixs, text_embeds), dim=1)
            # 通过 GPT-2 计算输出
            out = self.gpt(inputs_embeds=embedding_cat)
            return out
        else:
            # 推理阶段只返回前缀
            return prefixs