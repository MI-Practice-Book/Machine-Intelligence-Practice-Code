import os
from typing import Tuple, List

import torch
from PIL import Image
from torch.utils.data import DataLoader

from dataset_loader import Flickr8kDataset
from model_core import ClipCaptionModel, load_clip_model

# --- 相对路径配置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_DIR = "./data/Flicker8k_Dataset"
TOKEN_FILE = "./data/Flickr8k_text/Flickr8k.token.txt"
# 这里指向刚才训练保存的权重 (如果训练了5轮，就选epoch_5)
MODEL_PATH = "./checkpoints/caption_model_epoch_5.pt" 


def build_image_index(
    clip_model, loader: DataLoader, device: str, max_batches: int = 3
) -> Tuple[torch.Tensor, List[str]]:
    """
    预提取一小部分图像特征，作为检索用的“向量数据库”。
    为了演示速度，默认只取前 max_batches 个 batch。
    由于 Flickr8k 每张图片有多条标注，这里按图片路径去重，以避免同图重复出现在 TopK。
    """
    img_feats, img_paths = [], []
    seen = set()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            pixel_values = batch["pixel_values"].to(device)
            feats = clip_model.get_image_features(pixel_values)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)

            # 逐条按图片路径去重
            for feat, path in zip(feats, batch["img_path"]):
                if path in seen:
                    continue
                seen.add(path)
                img_feats.append(feat.unsqueeze(0))
                img_paths.append(path)

            if i + 1 >= max_batches:
                break

    if not img_feats:
        raise RuntimeError("未能构建图像向量索引，请检查数据加载。")

    return torch.cat(img_feats, dim=0), img_paths

def main():
    print(f"\n>>> [系统启动] 正在加载数据与模型 (Device: {DEVICE})...")

    # 1. 基础检查
    if not os.path.exists(IMG_DIR):
        print(f"❌ 错误：找不到路径 {IMG_DIR}，请检查当前工作目录。")
        return

    dataset = Flickr8kDataset(IMG_DIR, TOKEN_FILE)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 2. 加载模型
    clip_model = load_clip_model(DEVICE)  # 检索用
    caption_model = ClipCaptionModel().to(DEVICE)  # 生成用

    # [关键] 加载训练好的权重
    if os.path.exists(MODEL_PATH):
        print(f">>> 正在加载训练权重: {MODEL_PATH}")
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        caption_model.load_state_dict(state_dict)
        print(">>> 权重加载成功！")
    else:
        print(f"⚠️ 警告：找不到权重文件 {MODEL_PATH}")
        print("⚠️ 生成的描述将是乱码！请先运行 train_model.py。")

    caption_model.eval()

    # 3. 预构建小型图像索引，用于跨模态检索
    print("\n>>> [索引构建] 正在对图像提取特征...")
    img_features, img_paths = build_image_index(clip_model, loader, DEVICE, max_batches=3)
    print(f">>> 已构建 {len(img_paths)} 张图片的向量索引，可用于文本检索。")

    # 4. 循环交互式选择任务
    while True:
        print("\n" + "=" * 40)
        print("请选择任务：")
        print("1) 文本 -> 找图 (跨模态检索)")
        print("2) 图片 -> 生成描述 (看图说话)")
        print("0) 退出")
        print("=" * 40)
        task = input("请输入 0 / 1 / 2：").strip()

        if task == "0":
            print("已退出。")
            break

        if task == "1":
            # --- 功能A：以文搜图 ---
            query = input("请输入检索文本：").strip()
            if not query:
                print("❌ 未输入文本，请重试。")
                continue

            text_inputs = dataset.processor(text=[query], return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                text_features = clip_model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

            similarity = (text_features @ img_features.T).softmax(dim=-1)
            k = min(3, img_features.shape[0])
            values, indices = similarity[0].topk(k)

            print("\n✅ 检索结果（Top 3）：")
            for rank, (score, idx) in enumerate(zip(values, indices), start=1):
                print(f"{rank}. 相似度: {score.item():.4f} | 图片: {img_paths[idx]}")

        elif task == "2":
            # --- 功能B：看图说话 ---
            img_path = input("请输入图片路径：").strip()
            if not os.path.exists(img_path):
                print(f"❌ 找不到图片文件: {img_path}")
                continue

            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as exc:
                print(f"❌ 无法打开图片: {exc}")
                continue

            # 获取目标图片特征
            with torch.no_grad():
                img_inputs = dataset.processor(images=image, return_tensors="pt").to(DEVICE)
                target_feat = clip_model.get_image_features(**img_inputs)
                target_feat = target_feat / target_feat.norm(p=2, dim=-1, keepdim=True)

                prefix = caption_model(target_feat.float())
                generated_ids = caption_model.gpt.generate(
                    inputs_embeds=prefix,
                    max_length=30,
                    num_beams=5,
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                )

            output_text = dataset.processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print("\n🤖 AI描述:", output_text)
        else:
            print("❌ 未知任务选项，请重新输入。")

    print("=" * 40)

if __name__ == "__main__":
    main()