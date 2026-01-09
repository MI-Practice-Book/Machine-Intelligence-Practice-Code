import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def manage_context(messages, max_history_len=5):
    """
    上下文管理：保留 System Prompt 的同时，应用滑动窗口截断旧对话
    """
    if len(messages) <= max_history_len + 1:
        return messages
    
    # 始终提取并保留第一条 System 消息作为模型的人格基石
    system_msg = messages[0] if messages[0]['role'] == 'system' else None
    
    # 滑动窗口：仅截取最近的 N 条对话记录
    recent_msgs = messages[-(max_history_len):]
    
    # 重新组合：System Message + 最近对话
    return [system_msg] + recent_msgs if system_msg else recent_msgs

def generate_response(model, tokenizer, messages):
    # 将对话格式化为模型特定的模板
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512, 
        do_sample=True, 
        temperature=0.7,
        top_k=50
    )
    
    # 仅解码新生成的文本部分
    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return response

def main():
    base_model = "Qwen/Qwen2.5-0.5B-Instruct"
    lora_path = "outputs/sft_lora"
    
    # tokenizer 从 adapter 目录加载（因为训练时保存了 tokenizer）
    tokenizer = AutoTokenizer.from_pretrained(lora_path, use_fast=True)
    
    # 加载模型时集成显存优化：自动设备映射
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    
    # 载入微调后的 LoRA 权重
    model = PeftModel.from_pretrained(base, lora_path)
    model.eval()
    
    # 开启梯度检查点优化（推理阶段可进一步节省峰值显存）
    model.gradient_checkpointing_enable()

    # 初始化对话历史，包含 System 指令
    messages = [{"role": "system", "content": "你是一个严谨、简洁的AI助教。"}]
    
    print("✅ 智能助教系统已就绪（输入 exit 退出，自动维护 5 轮上下文记忆）\n")
    
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in {"exit", "quit"}: break
        
        # 1. 记录用户输入
        messages.append({"role": "user", "content": user_input})
        
        # 2. 上下文截断管理（防止超出模型最大长度）
        optimized_messages = manage_context(messages, max_history_len=6)
        
        # 3. 生成回复
        reply = generate_response(model, tokenizer, optimized_messages)
        print(f"Assistant: {reply}\n")
        
        # 4. 记录助手回复
        messages.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    main()