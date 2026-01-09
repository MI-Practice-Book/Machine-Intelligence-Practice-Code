import os
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor

class Flickr8kDataset(Dataset):
    def __init__(self, img_dir, token_file, model_name="openai/clip-vit-base-patch32"):
        self.img_dir = img_dir
        # 初始化 CLIP 处理器 (自动处理 Resize, Normalize, Tokenize)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.data = []

        # 解析标注文件
        # 原始格式示例: 1000268201_693b08cb0e.jpg#0	A child in a pink dress...
        print(f"正在加载标注文件: {token_file} ...")
        with open(token_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_id_full, caption = parts[0], parts[1]
                    # 去掉文件名后的 #0, #1 等标记
                    img_filename = img_id_full.split('#')[0]
                    self.data.append({"img_filename": img_filename, "caption": caption})
        
        print(f"✅ 数据加载完毕，共找到 {len(self.data)} 个图文对。")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.img_dir, item['img_filename'])
        text = item['caption']

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # 容错处理：若图片损坏，递归读取下一张
            return self.__getitem__((idx + 1) % len(self))

        # 使用 Processor 同时处理图文
        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77
        )

        return {
            "pixel_values": inputs.pixel_values.squeeze(), # [3, 224, 224]
            "input_ids": inputs.input_ids.squeeze(),       # [77]
            "caption_raw": text,                           # 原始文本(用于展示)
            "img_path": img_path                           # 图片路径(用于展示)
        }

# --- 单元测试代码 (仅在直接运行此文件时执行) ---
if __name__ == "__main__":
    # 测试加载效果
    ds = Flickr8kDataset(
        # 本地文件夹名是 Flicker8k_Dataset（缺少一个 r），与上方默认参数不同
        # 这里用实际存在的目录，避免 FileNotFoundError
        img_dir="./data/Flicker8k_Dataset",
        token_file="./data/Flickr8k_text/Flickr8k.token.txt"
    )
    sample = ds[0]
    print(f"\n[测试输出] 样本 0:")
    print(f"文本: {sample['caption_raw']}")
    print(f"图像张量形状: {sample['pixel_values'].shape}")