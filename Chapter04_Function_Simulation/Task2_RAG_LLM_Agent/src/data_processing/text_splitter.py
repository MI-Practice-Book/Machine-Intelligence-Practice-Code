"""
文本切分模块
"""
import logging

logger = logging.getLogger(__name__)


class TextSplitter:
    """文本切分器：混合切分策略"""
    
    def __init__(self, chunk_size=500, chunk_overlap=100):
        """
        Args:
            chunk_size: 目标块大小（字符数）
            chunk_overlap: 重叠大小（字符数）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split(self, text, chapters=None):
        """
        混合切分策略：章节+段落+重叠
        
        Args:
            text: 输入文本
            chapters: 章节信息（可选）
            
        Returns:
            chunks: List[str], 文本块列表
            metadata: List[Dict], 每个块的元数据
        """
        logger.info(f"开始文本切分 (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})")

        if len(chapters) == 0:
            # 如果没有提供章节信息，使用简单的固定长度切分
            return self._split_fixed_length(text)
        
        chunks = []
        metadata = []
        
        for chapter_info in chapters:
            content = chapter_info['content']
            chapter_num = chapter_info['chapter_num']
            chapter_title = chapter_info['title']
            
            # 短章节直接作为一个块
            if len(content) <= self.chunk_size:
                chunks.append(content)
                metadata.append({
                    'chapter_num': chapter_num,
                    'chapter_title': chapter_title,
                    'chunk_type': 'full_chapter'
                })
            else:
                # 长章节按段落分组
                chapter_chunks, chapter_metadata = self._split_by_paragraphs(
                    content, chapter_num, chapter_title
                )
                chunks.extend(chapter_chunks)
                metadata.extend(chapter_metadata)
        
        logger.info(f"文本切分完成，共 {len(chunks)} 个文本块")
        logger.info(f"平均块大小: {sum(len(c) for c in chunks) / len(chunks):.1f} 字符")
        
        return chunks, metadata
    
    def _split_by_paragraphs(self, content, chapter_num, chapter_title):
        """按段落分组切分"""
        chunks = []
        metadata = []
        
        paragraphs = content.split('\n')
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # 如果加入该段落后不超过chunk_size，继续累积
            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                current_chunk += para + "\n"
            else:
                # 保存当前块
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    metadata.append({
                        'chapter_num': chapter_num,
                        'chapter_title': chapter_title,
                        'chunk_type': 'paragraph_group'
                    })
                current_chunk = para + "\n"
        
        # 保存最后一个块
        if current_chunk:
            chunks.append(current_chunk.strip())
            metadata.append({
                'chapter_num': chapter_num,
                'chapter_title': chapter_title,
                'chunk_type': 'paragraph_group'
            })
        
        return chunks, metadata
    
    def _split_fixed_length(self, text):
        """固定长度切分（带重叠）"""
        chunks = []
        metadata = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            metadata.append({'chunk_type': 'fixed_length'})
            
            # 下一个块的起始位置
            start += (self.chunk_size - self.chunk_overlap)
        
        return chunks, metadata