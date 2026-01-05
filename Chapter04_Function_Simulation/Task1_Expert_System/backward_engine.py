# backward_engine.py - 反向推理引擎

import time
from knowledge_base import KnowledgeBase


class BackwardInferenceEngine:
    """反向推理引擎 - 目标驱动的推理"""
    
    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.verified_facts = {}  # 已验证的事实 {fact: True/False}
        self.inference_path = []  # 推理路径
        self.question_count = 0  # 提问次数
        self.start_time = None
        self.end_time = None
        
        # 定义系统知识
        self._initialize_knowledge()
    
    def _initialize_knowledge(self):
        """初始化系统知识"""
        # 可识别的动物列表
        self.animals = ['豹', '虎', '长颈鹿', '斑马', '鸵鸟', '企鹅', '信天翁']
        
        # 基础特征列表
        self.basic_features = [
            '有毛发', '产奶', '有羽毛', '会飞', '生蛋',
            '吃肉', '有蹄', '黄褐色', '暗斑点', '黑色条纹',
            '长腿', '长脖子', '黑白条纹', '不会飞', '会游泳',
            '黑白两色', '善飞'
        ]
        
        # 中间分类
        self.classifications = ['哺乳动物', '鸟类', '食肉动物', '有蹄类动物']
        
        # 规则库映射（从知识库提取）
        self._build_rule_mappings()
    
    def _build_rule_mappings(self):
        """构建规则映射，方便反向查找"""
        self.animal_rules = {
            '豹': [
                {
                    'premises': ['食肉动物', '黄褐色', '暗斑点'],
                    'description': '如果是食肉动物且黄褐色且有暗斑点，那么是豹'
                }
            ],
            '虎': [
                {
                    'premises': ['食肉动物', '黄褐色', '黑色条纹'],
                    'description': '如果是食肉动物且黄褐色且有黑色条纹，那么是虎'
                }
            ],
            '长颈鹿': [
                {
                    'premises': ['有蹄类动物', '长腿', '长脖子', '黄褐色', '暗斑点'],
                    'description': '如果是有蹄类动物且长腿且长脖子且黄褐色且有暗斑点，那么是长颈鹿'
                }
            ],
            '斑马': [
                {
                    'premises': ['有蹄类动物', '黑白条纹'],
                    'description': '如果是有蹄类动物且有黑白条纹，那么是斑马'
                }
            ],
            '鸵鸟': [
                {
                    'premises': ['鸟类', '长腿', '长脖子', '不会飞'],
                    'description': '如果是鸟类且长腿且长脖子且不会飞，那么是鸵鸟'
                }
            ],
            '企鹅': [
                {
                    'premises': ['鸟类', '不会飞', '会游泳', '黑白两色'],
                    'description': '如果是鸟类且不会飞且会游泳且黑白两色，那么是企鹅'
                }
            ],
            '信天翁': [
                {
                    'premises': ['鸟类', '善飞'],
                    'description': '如果是鸟类且善飞，那么是信天翁'
                }
            ]
        }
        
        self.classification_rules = {
            '哺乳动物': [
                {
                    'premises': ['有毛发'],
                    'description': '如果有毛发，那么是哺乳动物'
                },
                {
                    'premises': ['产奶'],
                    'description': '如果产奶，那么是哺乳动物'
                }
            ],
            '鸟类': [
                {
                    'premises': ['有羽毛'],
                    'description': '如果有羽毛，那么是鸟类'
                },
                {
                    'premises': ['会飞', '生蛋'],
                    'description': '如果会飞且生蛋，那么是鸟类'
                }
            ],
            '食肉动物': [
                {
                    'premises': ['哺乳动物', '吃肉'],
                    'description': '如果是哺乳动物且吃肉，那么是食肉动物'
                }
            ],
            '有蹄类动物': [
                {
                    'premises': ['哺乳动物', '有蹄'],
                    'description': '如果是哺乳动物且有蹄，那么是有蹄类动物'
                }
            ]
        }
    
    def reset(self):
        """重置引擎"""
        self.verified_facts = {}
        self.inference_path = []
        self.question_count = 0
    
    def ask_user(self, feature):
        """向用户提问"""
        self.question_count += 1
        
        print(f"\n问题 {self.question_count}: 动物有 '{feature}' 这个特征吗？")
        
        while True:
            answer = input("  请回答 (y/n 或 是/否): ").strip().lower()
            
            if answer in ['y', 'yes', '是', '有']:
                print(f"  ✓ 用户确认: 动物有 '{feature}'")
                self.inference_path.append(f"用户确认特征: {feature}")
                return True
            elif answer in ['n', 'no', '否', '没有']:
                print(f"  ✗ 用户否认: 动物没有 '{feature}'")
                self.inference_path.append(f"用户否认特征: {feature}")
                return False
            else:
                print("  输入无效，请输入 y/n 或 是/否")
    
    def verify_goal(self, goal):
        """
        验证目标（递归验证）
        
        Args:
            goal: 要验证的目标（特征、分类或动物）
            
        Returns:
            True/False
        """
        # 如果已经验证过，直接返回结果
        if goal in self.verified_facts:
            return self.verified_facts[goal]
        
        print(f"\n验证目标: {goal}")
        self.inference_path.append(f"开始验证目标: {goal}")
        
        # 如果是基础特征，询问用户
        if goal in self.basic_features:
            result = self.ask_user(goal)
            self.verified_facts[goal] = result
            return result
        
        # 如果是分类，查找支持规则
        if goal in self.classifications:
            result = self._verify_classification(goal)
            self.verified_facts[goal] = result
            return result
        
        # 如果是动物，查找支持规则
        if goal in self.animals:
            result = self._verify_animal(goal)
            self.verified_facts[goal] = result
            return result
        
        # 未知目标
        print(f"  ✗ 未知目标: {goal}")
        self.verified_facts[goal] = False
        return False
    
    def _verify_classification(self, classification):
        """验证分类"""
        rules = self.classification_rules.get(classification, [])
        
        if not rules:
            return False
        
        print(f"  查找 '{classification}' 的支持规则，共 {len(rules)} 条")
        
        # 尝试每条规则（OR逻辑）
        for i, rule in enumerate(rules, 1):
            print(f"\n  尝试规则 {i}: {rule['description']}")
            self.inference_path.append(f"尝试规则: {rule['description']}")
            
            # 验证规则的所有前提（AND逻辑）
            all_premises_verified = True
            for premise in rule['premises']:
                if not self.verify_goal(premise):
                    all_premises_verified = False
                    self.inference_path.append(f"  前提 '{premise}' 验证失败")
                    break
            
            if all_premises_verified:
                print(f"  ✓ 分类验证成功: {classification}")
                self.inference_path.append(f"分类验证成功: {classification}")
                return True
        
        print(f"  ✗ 所有规则验证失败，分类 '{classification}' 不成立")
        return False
    
    def _verify_animal(self, animal):
        """验证具体动物"""
        rules = self.animal_rules.get(animal, [])
        
        if not rules:
            return False
        
        print(f"  查找 '{animal}' 的识别规则，共 {len(rules)} 条")
        
        # 尝试每条规则
        for i, rule in enumerate(rules, 1):
            print(f"\n  尝试规则 {i}: {rule['description']}")
            self.inference_path.append(f"尝试规则: {rule['description']}")
            
            # 验证规则的所有前提
            all_premises_verified = True
            for premise in rule['premises']:
                if not self.verify_goal(premise):
                    all_premises_verified = False
                    self.inference_path.append(f"  前提 '{premise}' 验证失败")
                    break
            
            if all_premises_verified:
                print(f"  ✓ 动物识别成功: {animal}")
                self.inference_path.append(f"动物识别成功: {animal}")
                return True
        
        print(f"  ✗ 所有规则验证失败，排除动物 '{animal}'")
        return False
    
    def run_backward_inference(self):
        """执行反向推理"""
        print("\n" + "="*60)
        print("反向推理引擎 - 目标驱动推理")
        print("="*60)
        
        self.start_time = time.time()
        
        # 重置引擎
        self.reset()
        
        print("\n系统将尝试识别以下动物:")
        for i, animal in enumerate(self.animals, 1):
            print(f"  {i}. {animal}")
        
        print("\n开始推理...\n")
        
        # 尝试识别每个动物
        identified_animal = None
        
        for animal in self.animals:
            print("\n" + "="*60)
            print(f"尝试识别: {animal}")
            print("="*60)
            
            if self.verify_goal(animal):
                identified_animal = animal
                break
        
        self.end_time = time.time()
        
        # 显示结果
        self._show_result(identified_animal)
        
        return identified_animal
    
    def _show_result(self, animal):
        """显示推理结果"""
        print("\n" + "="*60)
        print("推理完成")
        print("="*60)
        
        inference_time = self.end_time - self.start_time
        
        if animal:
            print(f"\n✓ 识别结果: 该动物是 【{animal}】")
        else:
            print("\n✗ 识别结果: 无法识别该动物")
        
        print(f"\n推理统计:")
        print(f"  • 推理耗时: {inference_time:.4f} 秒")
        print(f"  • 提问次数: {self.question_count}")
        print(f"  • 验证的目标数: {len(self.verified_facts)}")
        print(f"  • 推理路径长度: {len(self.inference_path)}")
        
        # 显示推理路径
        self._show_inference_path()
    
    def _show_inference_path(self):
        """显示推理路径"""
        print("\n推理路径:")
        print("-" * 60)
        
        for i, step in enumerate(self.inference_path, 1):
            # 根据步骤类型添加缩进
            if "用户确认" in step or "用户否认" in step:
                print(f"{i:3d}.     {step}")
            elif "前提" in step:
                print(f"{i:3d}.       {step}")
            else:
                print(f"{i:3d}.   {step}")
        
        print("-" * 60)
    
    def show_verified_facts(self):
        """显示已验证的事实"""
        print("\n已验证的事实:")
        print("-" * 60)
        
        confirmed = [k for k, v in self.verified_facts.items() if v]
        denied = [k for k, v in self.verified_facts.items() if not v]
        
        if confirmed:
            print("\n确认为真:")
            for fact in confirmed:
                print(f"  ✓ {fact}")
        
        if denied:
            print("\n确认为假:")
            for fact in denied:
                print(f"  ✗ {fact}")
        
        print("-" * 60)
