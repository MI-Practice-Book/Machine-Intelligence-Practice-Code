"""
问答生成器
"""
import torch
import logging
import re
from .prompt_templates import build_prompt

logger = logging.getLogger(__name__)


class QAGenerator:
    """问答生成器"""
    
    def __init__(self, model, tokenizer, config):
        """
        Args:
            model: LLM模型
            tokenizer: 分词器
            config: 生成配置
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = config.get('max_new_tokens', 512)
        self.temperature = config.get('temperature', 0.3)
        # self.top_p = config.get('top_p', 0.9)
        self.top_p = config.get('top_p', 0.85)  # 降低top_p减少随机性
        self.repetition_penalty = config.get('repetition_penalty', 1.2)  # 添加重复惩罚
    
    
    def generate(self, prompt):
        """
        生成答案
        
        Args:
            prompt: 完整的提示词
            
        Returns:
            answer: 生成的答案文本
        """
        # 编码输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # 生成（添加更多约束）
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=10,  # 最少生成10个token
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                repetition_penalty=self.repetition_penalty,  # 重复惩罚
                no_repeat_ngram_size=3,  # 禁止3-gram重复
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True  # 遇到结束符提前停止
            )
        
        # 解码输出
        full_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        # 提取答案部分
        answer = self._extract_answer(full_text, prompt)

        # 额外的停止处理
        answer = self._apply_stop_conditions(answer)
        
        return answer

    def _extract_answer(self, full_text, prompt):
        """从完整输出中提取答案（改进版）"""
        # 方法1：移除prompt部分
        if full_text.startswith(prompt):
            answer = full_text[len(prompt):].strip()
        else:
            # 方法2：查找答案标记
            markers = ["【答案】", "答案：", "答案:", "Answer:"]
            for marker in markers:
                if marker in full_text:
                    parts = full_text.split(marker)
                    if len(parts) > 1:
                        answer = parts[-1].strip()
                        break
            else:
                # 方法3：取最后一段
                answer = full_text.strip()
        
        # 清理答案
        answer = self._clean_answer(answer)
        
        return answer
    
    def _clean_answer(self, answer):
        """清理答案中的问题"""
        # 1. 移除开头的重复问题
        # 匹配"问题："或"【问题】"等模式
        answer = re.sub(r'^(问题[:：】]|【问题】).+?[\n。]', '', answer)
        
        # 2. 移除"根据提供的原文"等无意义开头（如果后面有实际内容）
        prefixes_to_remove = [
            r'^根据(提供的)?原文[，,]',
            r'^根据(第\d+回)?[，,]',
            r'^原文(中|提到)[，,]'
        ]
        for pattern in prefixes_to_remove:
            answer = re.sub(pattern, '', answer)
        
        # 3. 检测并移除重复的句子
        sentences = re.split(r'[。！？]', answer)
        seen = set()
        unique_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if sent and sent not in seen:
                seen.add(sent)
                unique_sentences.append(sent)
        
        # 如果去重后只剩一句，用原答案
        if len(unique_sentences) > 1:
            answer = '。'.join(unique_sentences) + '。'
        
        # 4. 去除首尾空白和多余标点
        answer = answer.strip()
        answer = re.sub(r'。+', '。', answer)  # 多个句号变一个
        
        return answer
    
    def answer_question(self, question, contexts, metadata=None):
        """
        端到端问答
        
        Args:
            question: 用户问题
            contexts: 检索到的上下文
            metadata: 元数据（可选）
            
        Returns:
            answer: 最终答案
        """
        # 构造提示词
        prompt = build_prompt(question, contexts, metadata)
        
        # 生成答案
        answer = self.generate(prompt)
        
        # 后处理
        answer = self._postprocess(answer)
        
        return answer
    
    def _apply_stop_conditions(self, answer):
        """
        应用停止条件，截断不该有的内容
        """
        # 停止标记列表（按优先级）
        stop_markers = [
            "\n\nHuman:",
            "\n\nAssistant:",
            "\n\n【用户问题】",
            "\n\n【问题】",
            "\n\n问题：",
            "\n【用户问题】",
            "\n【问题】",
            "Human:",
            "Assistant:",
            "【用户问题】",
            "<|im_end|>",
            "<|im_start|>",
            "【原文片段】",
            "\n\n【",  # 任何新的标题块
        ]
        
        # 找到最早出现的停止标记
        min_pos = len(answer)
        for marker in stop_markers:
            pos = answer.find(marker)
            if pos != -1 and pos < min_pos:
                min_pos = pos
        
        # 截断
        if min_pos < len(answer):
            answer = answer[:min_pos].strip()
            logger.info(f"在位置 {min_pos} 截断答案")
        
        # 额外检查：如果出现连续3个以上换行，也截断
        triple_newline_pos = answer.find('\n\n\n')
        if triple_newline_pos != -1:
            answer = answer[:triple_newline_pos].strip()
            logger.info("检测到连续换行，已截断")
        
        return answer

    def _postprocess(self, answer):
        """答案后处理"""
        # 去除首尾空白
        answer = answer.strip()
        
        # 如果答案过长，在句末截断
        max_length = 300
        if len(answer) > max_length:
            truncate_pos = answer[:max_length].rfind('。')
            if truncate_pos > 0:
                answer = answer[:truncate_pos+1]
        
        return answer