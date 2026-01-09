import argparse
import json
from pathlib import Path

from datasets import load_dataset

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/train_messages.jsonl")
    p.add_argument("--max_samples", type=int, default=3000)
    p.add_argument("--system", default="你是一个严谨、简洁的AI助教。")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("databricks/databricks-dolly-15k", split="train")  # 15k rows
    if args.max_samples and args.max_samples < len(ds):
        ds = ds.select(range(args.max_samples))

    with out_path.open("w", encoding="utf-8") as f:
        for ex in ds:
            instruction = (ex.get("instruction") or "").strip()
            context = (ex.get("context") or "").strip()
            response = (ex.get("response") or "").strip()

            user = instruction if not context else f"{instruction}\n\n上下文：{context}"
            messages = [
                {"role": "system", "content": args.system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": response},
            ]
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {len(ds)} samples to {out_path}")

if __name__ == "__main__":
    main()
