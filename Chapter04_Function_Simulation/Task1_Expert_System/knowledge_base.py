# knowledge_base.py - 知识库定义

from rule import Rule
from facts import HasFeature, IsClassification, IsAnimal


class KnowledgeBase:
    """知识库 - 存储所有规则"""
    
    def __init__(self):
        self.rules = []
        self._build_rules()
    
    def _build_rules(self):
        """
        构建规则库
        
        优先级策略（Salience）设计：
        - 100-199: 基础分类规则（特征 → 大类）
        - 200-299: 中间分类规则（大类 → 子类）
        - 300-399: 具体识别规则（子类+特征 → 物种）
        
        在同一层级内，条件越多的规则优先级越高（更具体）
        """
        
        # ===== 第1层：基础分类规则 (salience=100-150) =====
        # 从基础特征推导出动物大类（哺乳动物、鸟类等）
        
        # 哺乳动物识别规则
        self.add_rule(
            name="rule_has_hair",
            conditions=[
                {'type': 'HasFeature', 'feature': '有毛发'}
            ],
            actions=[
                {'type': 'declare_classification', 'category': '哺乳动物'}
            ],
            salience=100,  # 单条件，基础优先级
            description="如果有毛发，那么是哺乳动物"
        )
        
        self.add_rule(
            name="rule_produces_milk",
            conditions=[
                {'type': 'HasFeature', 'feature': '产奶'}
            ],
            actions=[
                {'type': 'declare_classification', 'category': '哺乳动物'}
            ],
            salience=100,  # 单条件，基础优先级
            description="如果产奶，那么是哺乳动物"
        )
        
        # 鸟类识别规则
        self.add_rule(
            name="rule_has_feathers",
            conditions=[
                {'type': 'HasFeature', 'feature': '有羽毛'}
            ],
            actions=[
                {'type': 'declare_classification', 'category': '鸟类'}
            ],
            salience=100,  # 单条件，基础优先级
            description="如果有羽毛，那么是鸟类"
        )
        
        self.add_rule(
            name="rule_flies_and_lays_eggs",
            conditions=[
                {'type': 'HasFeature', 'feature': '会飞'},
                {'type': 'HasFeature', 'feature': '生蛋'}
            ],
            actions=[
                {'type': 'declare_classification', 'category': '鸟类'}
            ],
            salience=110,  # 两个条件，更具体，优先级稍高
            description="如果会飞且生蛋，那么是鸟类"
        )
        
        # ===== 第2层：中间分类规则 (salience=200-250) =====
        # 从动物大类推导出更具体的子类（食肉动物、有蹄类等）
        
        # 食肉动物
        self.add_rule(
            name="rule_carnivore",
            conditions=[
                {'type': 'IsClassification', 'category': '哺乳动物'},
                {'type': 'HasFeature', 'feature': '吃肉'}
            ],
            actions=[
                {'type': 'declare_classification', 'category': '食肉动物'}
            ],
            salience=200,  # 中间分类，标准优先级
            description="如果是哺乳动物且吃肉，那么是食肉动物"
        )
        
        # 有蹄类动物
        self.add_rule(
            name="rule_ungulate",
            conditions=[
                {'type': 'IsClassification', 'category': '哺乳动物'},
                {'type': 'HasFeature', 'feature': '有蹄'}
            ],
            actions=[
                {'type': 'declare_classification', 'category': '有蹄类动物'}
            ],
            salience=200,  # 中间分类，标准优先级
            description="如果是哺乳动物且有蹄，那么是有蹄类动物"
        )
        
        # ===== 第3层：具体动物识别规则 (salience=300-399) =====
        # 从子类和特征推导出具体物种
        # 条件越多，越具体，优先级越高
        
        # 豹（3个条件）
        self.add_rule(
            name="rule_leopard",
            conditions=[
                {'type': 'IsClassification', 'category': '食肉动物'},
                {'type': 'HasFeature', 'feature': '黄褐色'},
                {'type': 'HasFeature', 'feature': '暗斑点'}
            ],
            actions=[
                {'type': 'declare_animal', 'species': '豹'}
            ],
            salience=330,  # 3个条件：中等具体性
            description="如果是食肉动物且黄褐色且有暗斑点，那么是豹"
        )
        
        # 虎（3个条件）
        self.add_rule(
            name="rule_tiger",
            conditions=[
                {'type': 'IsClassification', 'category': '食肉动物'},
                {'type': 'HasFeature', 'feature': '黄褐色'},
                {'type': 'HasFeature', 'feature': '黑色条纹'}
            ],
            actions=[
                {'type': 'declare_animal', 'species': '虎'}
            ],
            salience=330,  # 3个条件：中等具体性
            description="如果是食肉动物且黄褐色且有黑色条纹，那么是虎"
        )
        
        # 长颈鹿（5个条件）
        self.add_rule(
            name="rule_giraffe",
            conditions=[
                {'type': 'IsClassification', 'category': '有蹄类动物'},
                {'type': 'HasFeature', 'feature': '长腿'},
                {'type': 'HasFeature', 'feature': '长脖子'},
                {'type': 'HasFeature', 'feature': '黄褐色'},
                {'type': 'HasFeature', 'feature': '暗斑点'}
            ],
            actions=[
                {'type': 'declare_animal', 'species': '长颈鹿'}
            ],
            salience=350,  # 5个条件：高具体性，最高优先级
            description="如果是有蹄类动物且长腿且长脖子且黄褐色且有暗斑点，那么是长颈鹿"
        )
        
        # 斑马（2个条件）
        self.add_rule(
            name="rule_zebra",
            conditions=[
                {'type': 'IsClassification', 'category': '有蹄类动物'},
                {'type': 'HasFeature', 'feature': '黑白条纹'}
            ],
            actions=[
                {'type': 'declare_animal', 'species': '斑马'}
            ],
            salience=320,  # 2个条件：较低具体性
            description="如果是有蹄类动物且有黑白条纹，那么是斑马"
        )
        
        # 鸵鸟（4个条件）
        self.add_rule(
            name="rule_ostrich",
            conditions=[
                {'type': 'IsClassification', 'category': '鸟类'},
                {'type': 'HasFeature', 'feature': '长腿'},
                {'type': 'HasFeature', 'feature': '长脖子'},
                {'type': 'HasFeature', 'feature': '不会飞'}
            ],
            actions=[
                {'type': 'declare_animal', 'species': '鸵鸟'}
            ],
            salience=340,  # 4个条件：较高具体性
            description="如果是鸟类且长腿且长脖子且不会飞，那么是鸵鸟"
        )
        
        # 企鹅（4个条件）
        self.add_rule(
            name="rule_penguin",
            conditions=[
                {'type': 'IsClassification', 'category': '鸟类'},
                {'type': 'HasFeature', 'feature': '不会飞'},
                {'type': 'HasFeature', 'feature': '会游泳'},
                {'type': 'HasFeature', 'feature': '黑白两色'}
            ],
            actions=[
                {'type': 'declare_animal', 'species': '企鹅'}
            ],
            salience=340,  # 4个条件：较高具体性
            description="如果是鸟类且不会飞且会游泳且黑白两色，那么是企鹅"
        )
        
        # 信天翁（2个条件）
        self.add_rule(
            name="rule_albatross",
            conditions=[
                {'type': 'IsClassification', 'category': '鸟类'},
                {'type': 'HasFeature', 'feature': '善飞'}
            ],
            actions=[
                {'type': 'declare_animal', 'species': '信天翁'}
            ],
            salience=320,  # 2个条件：较低具体性
            description="如果是鸟类且善飞，那么是信天翁"
        )
    
    def add_rule(self, name, conditions, actions, salience=0, description=""):
        """添加规则到知识库"""
        rule = Rule(name, conditions, actions, salience, description)
        self.rules.append(rule)
    
    def get_all_rules(self):
        """获取所有规则"""
        return self.rules
    
    def get_rule_by_name(self, name):
        """根据名称获取规则"""
        for rule in self.rules:
            if rule.name == name:
                return rule
        return None
    
    def get_statistics(self):
        """获取知识库统计信息"""
        return {
            'total_rules': len(self.rules),
            'rules_by_priority': self._count_by_priority(),
            'rules_by_type': self._count_by_type()
        }
    
    def _count_by_priority(self):
        """按优先级统计规则数量"""
        priority_counts = {}
        for rule in self.rules:
            priority = rule.salience
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        return priority_counts
    
    def _count_by_type(self):
        """按类型统计规则数量"""
        type_counts = {
            'feature_rules': 0,      # 特征规则 (100-199)
            'classification_rules': 0,  # 分类规则 (200-299)
            'recognition_rules': 0   # 识别规则 (300-399)
        }
        
        for rule in self.rules:
            if 100 <= rule.salience < 200:
                type_counts['feature_rules'] += 1
            elif 200 <= rule.salience < 300:
                type_counts['classification_rules'] += 1
            elif 300 <= rule.salience < 400:
                type_counts['recognition_rules'] += 1
        
        return type_counts
