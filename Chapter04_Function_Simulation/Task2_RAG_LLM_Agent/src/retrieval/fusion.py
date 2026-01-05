"""
检索结果融合算法
"""


def reciprocal_rank_fusion(ranked_lists, k=60):
    """
    Reciprocal Rank Fusion (RRF) 算法
    
    Args:
        ranked_lists: List[List[int]], 多个检索器返回的文档ID列表
        k: 平滑常数，默认60
        
    Returns:
        fused_ranking: List[int], 融合后的排序结果
    """
    fusion_scores = {}
    
    # 遍历每个检索器的结果
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            if doc_id not in fusion_scores:
                fusion_scores[doc_id] = 0.0
            # 累加RRF分数: 1 / (k + rank)
            fusion_scores[doc_id] += 1.0 / (k + rank)
    
    # 按分数降序排序
    sorted_docs = sorted(
        fusion_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # 返回排序后的文档ID列表
    fused_ranking = [doc_id for doc_id, score in sorted_docs]
    
    return fused_ranking