# forward_engine.py - 正向推理引擎

import time
from knowledge_base import KnowledgeBase
from facts import HasFeature, IsClassification, IsAnimal, RuleFired
from conflict_resolver import ConflictResolver


class ForwardInferenceEngine:
    """正向推理引擎 - 数据驱动的推理"""
    
    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.conflict_resolver = ConflictResolver()
        self.facts = []  # 事实库
        self.inference_steps = []  # 推理步骤记录
        self.fired_rules = set()  # 已触发的规则
        self.result = None
        self.start_time = None
        self.end_time = None
    
    def reset(self):
        """重置推理引擎"""
        self.facts = []
        self.inference_steps = []
        self.fired_rules = set()
        self.result = None
        self.conflict_resolver.reset()
    
    def add_feature(self, feature):
        """添加特征事实"""
        fact = HasFeature(feature)
        if fact not in self.facts:
            self.facts.append(fact)
            self.inference_steps.append(f"添加事实: {feature}")
            print(f"  添加事实: 动物有特征 '{feature}'")
    
    def add_features(self, features):
        """批量添加特征"""
        for feature in features:
            self.add_feature(feature)
    
    def find_applicable_rules(self):
        """
        查找所有可应用的规则（前提条件满足且未触发过）
        
        Returns:
            可应用的规则列表
        """
        applicable_rules = []
        
        for rule in self.knowledge_base.get_all_rules():
            # 检查规则是否已触发
            if rule.name in self.fired_rules:
                continue
            
            # 检查规则条件是否满足
            if rule.check_conditions(self.facts):
                applicable_rules.append(rule)
        
        return applicable_rules
    
    def run_inference(self, features):
        """
        执行正向推理
        
        Args:
            features: 初始特征列表
            
        Returns:
            识别结果
        """
        print("\n" + "="*60)
        print("正向推理引擎 - 数据驱动推理")
        print("="*60)
        
        self.start_time = time.time()
        
        # 1. 重置引擎
        self.reset()
        
        # 2. 添加初始事实
        print("\n【阶段1】添加初始事实:")
        self.add_features(features)
        
        # 3. 执行推理循环
        print("\n【阶段2】开始推理循环:")
        iteration = 0
        
        while True:
            iteration += 1
            print(f"\n--- 推理迭代 {iteration} ---")
            
            # 3.1 查找可应用的规则
            applicable_rules = self.find_applicable_rules()
            
            if not applicable_rules:
                print("  没有可应用的规则，推理结束")
                break
            
            print(f"  发现 {len(applicable_rules)} 条可应用的规则")
            
            # 3.2 冲突消解
            selected_rule = self.conflict_resolver.resolve(applicable_rules)
            
            if not selected_rule:
                print("  冲突消解失败，推理结束")
                break
            
            print(f"  选择规则: {selected_rule.name} (优先级={selected_rule.salience})")
            print(f"  规则描述: {selected_rule.description}")
            
            # 3.3 触发规则
            new_facts = selected_rule.execute(self.facts)
            
            # 3.4 添加新事实
            for fact in new_facts:
                if fact not in self.facts:
                    self.facts.append(fact)
                    
                    # 记录推理步骤
                    if isinstance(fact, IsClassification):
                        step = f"推导出分类: {fact.get('category')}"
                        print(f"  → {step}")
                    elif isinstance(fact, IsAnimal):
                        step = f"识别出动物: {fact.get('species')}"
                        print(f"  → {step}")
                        self.result = fact.get('species')
                    
                    self.inference_steps.append(step)
            
            # 3.5 标记规则已触发
            self.fired_rules.add(selected_rule.name)
            
            # 3.6 检查是否已识别出动物
            if self.result:
                print(f"\n  ✓ 已识别出动物: {self.result}")
                break
            
            # 防止无限循环
            if iteration > 20:
                print("\n  警告: 超过最大迭代次数，推理结束")
                break
        
        self.end_time = time.time()
        
        # 4. 输出结果
        self._show_result()
        
        return self.result
    
    def _show_result(self):
        """显示推理结果"""
        print("\n" + "="*60)
        print("推理完成")
        print("="*60)
        
        inference_time = self.end_time - self.start_time
        
        if self.result:
            print(f"\n✓ 识别结果: 该动物是 【{self.result}】")
        else:
            print("\n✗ 识别结果: 无法识别该动物")
        
        print(f"\n推理统计:")
        print(f"  • 推理耗时: {inference_time:.4f} 秒")
        print(f"  • 推理步骤数: {len(self.inference_steps)}")
        print(f"  • 触发规则数: {len(self.fired_rules)}")
        print(f"  • 最终事实数: {len(self.facts)}")
        
        # 显示推理链
        self._show_inference_chain()
    
    def _show_inference_chain(self):
        """显示推理链"""
        print("\n推理链:")
        print("-" * 60)
        
        for i, step in enumerate(self.inference_steps, 1):
            print(f"{i:2d}. {step}")
        
        print("-" * 60)
    
    def show_final_facts(self):
        """显示最终事实库"""
        print("\n最终事实库:")
        print("-" * 60)
        
        # 分类显示
        features = [f for f in self.facts if isinstance(f, HasFeature)]
        classifications = [f for f in self.facts if isinstance(f, IsClassification)]
        animals = [f for f in self.facts if isinstance(f, IsAnimal)]
        
        if features:
            print("\n特征事实:")
            for fact in features:
                print(f"  • {fact.get('feature')}")
        
        if classifications:
            print("\n分类事实:")
            for fact in classifications:
                print(f"  • {fact.get('category')}")
        
        if animals:
            print("\n动物事实:")
            for fact in animals:
                print(f"  • {fact.get('species')}")
        
        print("-" * 60)
