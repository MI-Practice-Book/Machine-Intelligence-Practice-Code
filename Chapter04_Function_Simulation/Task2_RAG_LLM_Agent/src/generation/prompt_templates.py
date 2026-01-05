"""
提示词模板
"""

# 基础提示模板
PROMPT_TEMPLATE_V1 = """你是一个精通中国名著《西游记》的专家学者。你的唯一职责是根据提问，直接提供精简并准确的解答，避免与问题无关的内容。
为了更好地回答问题，我提供了一些可能有帮助的《西游记》文章片段用于参考，注意，这些文章片段不一定包含问题答案，仅供参考。

【用户问题】
{question}

【相关原文片段】
{context}

【回答要求】
1. 答案必须基于已知事实或上述原文片段,不得编造不存在的信息
2. 如果原文片段不足以回答问题,请明确说明"根据提供的原文,无法完全回答此问题"
3. 回答简洁明了,200字以内为宜
4. 不能谈论任何与《西游记》无关的话题
5. 只使用简体中文

【答案】
"""

# 带章回信息的提示模板
PROMPT_TEMPLATE_V2 = """你是一个精通中国名著《西游记》的专家学者。你的唯一职责是根据提问，直接提供精简并准确的解答，避免与问题无关的内容。
为了更好地回答问题，我提供了一些可能有帮助的《西游记》文章片段用于参考，注意，这些文章片段不一定包含问题答案，仅供参考。

【用户问题】
{question}

【相关原文片段】
{context}

【回答要求】
1. 答案必须基于已知事实或上述原文片段,不得编造不存在的信息
2. 如果原文片段不足以回答问题,请明确说明"根据提供的原文,无法完全回答此问题"
3. 回答简洁明了,200字以内为宜
4. 尽量在答案中引用具体章回,如"根据第X回..."
5. 不能谈论任何与《西游记》无关的话题
6. 只使用简体中文

【答案】
"""


def build_prompt(question, contexts, metadata=None, template_version='v2'):
    """
    构造提示词
    
    Args:
        question: 用户问题
        contexts: List[str], 检索到的文本片段
        metadata: List[Dict], 元数据（可选）
        template_version: 模板版本 ('v1' 或 'v2')
        
    Returns:
        prompt: 完整的提示词
    """
    # 选择模板
    if template_version == 'v1':
        template = PROMPT_TEMPLATE_V1
    else:
        template = PROMPT_TEMPLATE_V2
    
    # 组织上下文
    context_text = ""
    if metadata:
        for i, (ctx, meta) in enumerate(zip(contexts, metadata), 1):
            chapter = meta.get('chapter_num', '?')
            chapter_title = meta.get('chapter_title', '')
            if chapter_title:
                context_text += f"[片段{i} - 第{chapter}回: {chapter_title}]\n{ctx}\n\n"
            else:
                context_text += f"[片段{i} - 第{chapter}回]\n{ctx}\n\n"
    else:
        for i, ctx in enumerate(contexts, 1):
            context_text += f"[片段{i}]\n{ctx}\n\n"
    
    # 填充模板
    prompt = template.format(
        context=context_text.strip(),
        question=question
    )
    
    return prompt