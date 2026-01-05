# rule.py - 规则定义模块

class Rule:
    """规则类"""
    def __init__(self, name, conditions, actions, salience=0, description=""):
        self.name = name
        self.conditions = conditions  # 前提条件列表
        self.actions = actions  # 结论动作列表
        self.salience = salience  # 优先级
        self.description = description
        self.fired_count = 0  # 触发次数
    
    def check_conditions(self, facts):
        """检查规则的所有条件是否满足"""
        for condition in self.conditions:
            if not self._check_single_condition(condition, facts):
                return False
        return True
    
    def _check_single_condition(self, condition, facts):
        """检查单个条件"""
        condition_type = condition.get('type')
        
        if condition_type == 'HasFeature':
            feature = condition.get('feature')
            return any(fact.get('feature') == feature 
                      for fact in facts if isinstance(fact, HasFeature))
        
        elif condition_type == 'IsClassification':
            category = condition.get('category')
            return any(fact.get('category') == category 
                      for fact in facts if isinstance(fact, IsClassification))
        
        elif condition_type == 'IsAnimal':
            species = condition.get('species')
            return any(fact.get('species') == species 
                      for fact in facts if isinstance(fact, IsAnimal))
        
        return False
    
    def execute(self, facts):
        """执行规则的动作"""
        self.fired_count += 1
        new_facts = []
        
        for action in self.actions:
            action_type = action.get('type')
            
            if action_type == 'declare_classification':
                category = action.get('category')
                new_fact = IsClassification(category)
                if new_fact not in facts:
                    new_facts.append(new_fact)
            
            elif action_type == 'declare_animal':
                species = action.get('species')
                new_fact = IsAnimal(species)
                if new_fact not in facts:
                    new_facts.append(new_fact)
        
        return new_facts
    
    def __repr__(self):
        return f"Rule({self.name}, salience={self.salience})"


# 导入事实类
from facts import HasFeature, IsClassification, IsAnimal


def create_rule(name, conditions, actions, salience=0, description=""):
    """工厂函数创建规则"""
    return Rule(name, conditions, actions, salience, description)
