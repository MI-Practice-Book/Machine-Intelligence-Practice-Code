"""
文本预处理模块
"""
import re
import logging

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """文本预处理器：清洗、标准化、章节提取"""
    
    def preprocess(self, text):
        """
        文本预处理
        
        Args:
            text: 原始文本
            
        Returns:
            text: 预处理后的文本
        """
        logger.info("开始文本预处理...")
        
        # 去除多余的空行
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # 去除每行首尾空白
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # 统一章回标题格式
        text = self._normalize_chapter_titles(text)
        
        # 去除页码
        text = re.sub(r'-\s*\d+\s*-', '', text)
        
        # 去除网址
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # 标准化标点符号
        text = self._normalize_punctuation(text)
        
        logger.info("文本预处理完成")
        return text
    
    def _normalize_chapter_titles(self, text):
        """统一章回标题格式"""
        def normalize(match):
            chapter_num = match.group(1)
            if chapter_num.isdigit():
                return f"第{int(chapter_num)}回"
            return match.group(0)
        
        text = re.sub(r'第([0-9零一二三四五六七八九十百]+)回', normalize, text)
        return text
    
    def _normalize_punctuation(self, text):
        """标准化标点符号"""
        punctuation_map = {
            ',': '，',
            '.': '。',
            '!': '！',
            '?': '？',
            ';': '；',
            ':': '：',
        }
        for eng, chn in punctuation_map.items():
            text = text.replace(eng, chn)
        return text
    
    def extract_chapters(self, text):
        """
        提取章回信息

        Args:
            text: 预处理后的文本

        Returns:
            chapters: List[Dict], 每个元素包含章回信息
        """
        chapter_pattern = r'第([0-9零一二三四五六七八九十百千两]+)回\s+(.+?)(?=\n)'
        chapters = []
        matches = list(re.finditer(chapter_pattern, text))

        def chinese2digits(chinese_num):
            # 简单的中文数字转阿拉伯数字（仅适用于西游记常见格式）
            cn_num = {'零':0, '一':1, '二':2, '两':2, '三':3, '四':4, '五':5, '六':6, '七':7, '八':8, '九':9}
            cn_unit = {'十':10, '百':100, '千':1000}
            result = 0
            unit = 1
            num = 0
            for c in reversed(chinese_num):
                if c in cn_unit:
                    unit = cn_unit[c]
                    if num == 0:
                        num = 1
                    result += num * unit
                    num = 0
                    unit = 1
                elif c in cn_num:
                    num = cn_num[c]
                    result += num * unit
                    num = 0
            if result == 0:
                return int(chinese_num) if chinese_num.isdigit() else 0
            return result

        for i, match in enumerate(matches):
            chapter_num_raw = match.group(1)
            # 支持中文和数字
            if chapter_num_raw.isdigit():
                chapter_num = int(chapter_num_raw)
            else:
                chapter_num = chinese2digits(chapter_num_raw)
            title = match.group(2).strip()
            start_pos = match.end()

            # 确定章回正文的结束位置
            if i < len(matches) - 1:
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(text)

            content = text[start_pos:end_pos].strip()

            chapters.append({
                'chapter_num': chapter_num,
                'title': title,
                'content': content,
                'start_pos': start_pos,
                'end_pos': end_pos
            })

        logger.info(f"识别出 {len(chapters)} 个章回")
        return chapters

