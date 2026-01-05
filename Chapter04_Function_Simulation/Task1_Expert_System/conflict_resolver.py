# conflict_resolver.py - 冲突消解器

import time


class ConflictResolver:
    """冲突消解器 - 使用优先级策略"""
    
    def __init__(self):
        self.conflict_history = []  # 冲突历史记录
        self.strategy = "priority"  # 默认策略：优先级
    
    def resolve(self, conflicting_rules):
        """
        冲突消解：从多个可触发的规则中选择一个
        
        Args:
            conflicting_rules: 可触发的规则列表
            
        Returns:
            选中的规则
        """
        if not conflicting_rules:
            return None
        
        if len(conflicting_rules) == 1:
            return conflicting_rules[0]
        
        # 记录冲突
        conflict_info = {
            'timestamp': time.time(),
            'conflict_count': len(conflicting_rules),
            'rules': [r.name for r in conflicting_rules]
        }
        
        # 按优先级排序（salience值高的优先）
        sorted_rules = sorted(
            conflicting_rules,
            key=lambda r: (r.salience, self._calculate_specificity(r)),
            reverse=True
        )
        
        selected_rule = sorted_rules[0]
        conflict_info['selected_rule'] = selected_rule.name
        conflict_info['selected_salience'] = selected_rule.salience
        
        self.conflict_history.append(conflict_info)
        
        return selected_rule
    
    def _calculate_specificity(self, rule):
        """
        计算规则的专一性（条件越多越专一）
        
        Args:
            rule: 规则对象
            
        Returns:
            专一性分数
        """
        return len(rule.conditions)
    
    def analyze(self):
        """分析冲突消解情况"""
        print("\n" + "="*60)
        print("冲突消解分析")
        print("="*60)
        
        if not self.conflict_history:
            print("本次推理未发生规则冲突")
            return
        
        print(f"\n发生冲突次数: {len(self.conflict_history)}")
        
        for i, conflict in enumerate(self.conflict_history, 1):
            print(f"\n冲突 #{i}:")
            print(f"  冲突规则数: {conflict['conflict_count']}")
            print(f"  冲突规则: {', '.join(conflict['rules'])}")
            print(f"  消解策略: 优先级策略 (salience值高的规则优先)")
            print(f"  选择的规则: {conflict['selected_rule']}")
            print(f"  规则优先级: {conflict['selected_salience']}")
        
        print("\n" + "="*60)
    
    def get_statistics(self):
        """获取冲突消解统计信息"""
        if not self.conflict_history:
            return {
                "total_conflicts": 0,
                "avg_conflict_size": 0
            }
        
        total_conflicts = len(self.conflict_history)
        avg_conflict_size = sum(c['conflict_count'] for c in self.conflict_history) / total_conflicts
        
        return {
            "total_conflicts": total_conflicts,
            "avg_conflict_size": avg_conflict_size,
            "first_conflict_time": self.conflict_history[0]['timestamp'],
            "last_conflict_time": self.conflict_history[-1]['timestamp']
        }
    
    def reset(self):
        """重置冲突历史"""
        self.conflict_history = []
