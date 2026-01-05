# facts.py - 事实定义模块

class Fact:
    """基础事实类"""
    def __init__(self, **kwargs):
        self.attributes = kwargs
    
    def __repr__(self):
        attrs = ', '.join([f"{k}={v}" for k, v in self.attributes.items()])
        return f"{self.__class__.__name__}({attrs})"
    
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.attributes == other.attributes
    
    def __hash__(self):
        return hash((self.__class__.__name__, tuple(sorted(self.attributes.items()))))
    
    def get(self, key, default=None):
        return self.attributes.get(key, default)
    
    def matches(self, **conditions):
        """检查事实是否匹配给定条件"""
        for key, value in conditions.items():
            if self.attributes.get(key) != value:
                return False
        return True


class HasFeature(Fact):
    """拥有某个特征"""
    def __init__(self, feature):
        super().__init__(feature=feature)


class IsClassification(Fact):
    """属于某个分类"""
    def __init__(self, category):
        super().__init__(category=category)


class IsAnimal(Fact):
    """是某种动物"""
    def __init__(self, species):
        super().__init__(species=species)


class RuleFired(Fact):
    """规则已触发"""
    def __init__(self, rule_name):
        super().__init__(rule_name=rule_name)
