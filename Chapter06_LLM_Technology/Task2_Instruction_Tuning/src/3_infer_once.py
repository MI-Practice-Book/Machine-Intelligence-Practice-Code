import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="outputs/sft_lora")
    p.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    # 如果你保存的是 LoRA adapter 到同一目录，transformers 会自动加载（某些保存方式需 peft）
    # 这里为了“最少依赖”，直接用 base_model + tokenizer 跑通也可以；
    # 若要严格加载 LoRA，请按你当前保存方式引入 peft 的 PeftModel（见后面的 chat_cli）。
    base.eval()

    messages = [
        {"role": "system", "content": "你是一个严谨、简洁的AI助教。"},
        {"role": "user", "content": "用一句话解释什么是监督微调（SFT）。"},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(base.device)

    out = base.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.9)

    # 只取新增 token（这是“像对话系统”的关键）
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    print(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

if __name__ == "__main__":
    main()
