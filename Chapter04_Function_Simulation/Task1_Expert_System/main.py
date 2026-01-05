# main.py - 主程序

import sys
from forward_engine import ForwardInferenceEngine
from backward_engine import BackwardInferenceEngine
from knowledge_base import KnowledgeBase


class AnimalExpertSystem:
    """动物识别专家系统主程序"""
    
    def __init__(self):
        self.forward_engine = ForwardInferenceEngine()
        self.backward_engine = BackwardInferenceEngine()
        self.knowledge_base = KnowledgeBase()
    
    def print_header(self):
        """打印系统标题"""
        print("\n" + "="*70)
        print(" "*20 + "动物识别专家系统")
        print(" "*15 + "基于产生式规则的正向与反向推理")
        print("="*70)
    
    def print_menu(self):
        """打印主菜单"""
        print("\n" + "-"*70)
        print("请选择功能:")
        print("-"*70)
        print("  1. 正向推理模式（数据驱动）")
        print("  2. 反向推理模式（目标驱动）")
        print("  3. 对比两种推理模式")
        print("  4. 查看系统信息")
        print("  5. 查看知识库")
        print("  6. 冲突消解分析")
        print("  0. 退出系统")
        print("-"*70)
    
    def run_forward_inference(self):
        """运行正向推理"""
        print("\n" + "="*70)
        print("正向推理模式 - 数据驱动")
        print("="*70)
        
        print("\n可用特征列表:")
        features_list = [
            '有毛发', '产奶', '有羽毛', '会飞', '生蛋',
            '吃肉', '有蹄', '黄褐色', '暗斑点', '黑色条纹',
            '长腿', '长脖子', '黑白条纹', '不会飞', '会游泳',
            '黑白两色', '善飞'
        ]
        
        for i in range(0, len(features_list), 4):
            row = features_list[i:i+4]
            print("  " + " | ".join(f"{f:8s}" for f in row))
        
        print("\n" + "-"*70)
        print("请输入动物特征（用逗号分隔，例如: 有毛发,吃肉,黄褐色,暗斑点）")
        print("或输入 'q' 返回主菜单")
        print("-"*70)
        
        features_input = input("\n输入特征: ").strip()
        
        if features_input.lower() == 'q':
            return
        
        if not features_input:
            print("\n错误：未输入任何特征")
            return
        
        features = [f.strip() for f in features_input.split(',') if f.strip()]
        
        if not features:
            print("\n错误：未输入有效特征")
            return
        
        print(f"\n您输入的特征: {', '.join(features)}")
        
        # 执行推理
        result = self.forward_engine.run_inference(features)
        
        # 显示最终事实库
        self.forward_engine.show_final_facts()
        
        return result
    
    def run_backward_inference(self):
        """运行反向推理"""
        print("\n" + "="*70)
        print("反向推理模式 - 目标驱动")
        print("="*70)
        
        print("\n系统将通过提问的方式识别动物")
        print("请根据问题回答: y/n 或 是/否")
        print("-"*70)
        
        input("\n按回车键开始...")
        
        # 执行推理
        result = self.backward_engine.run_backward_inference()
        
        # 显示已验证的事实
        self.backward_engine.show_verified_facts()
        
        return result
    
    def compare_inference_modes(self):
        """对比两种推理模式"""
        print("\n" + "="*70)
        print("对比正向推理与反向推理")
        print("="*70)
        
        print("\n我们将用相同的测试用例对比两种推理方式")
        
        # 测试用例1：豹
        print("\n" + "="*70)
        print("测试用例1: 识别豹")
        print("="*70)
        print("\n给定特征: 有毛发, 吃肉, 黄褐色, 暗斑点")
        
        input("\n按回车键开始正向推理...")
        
        print("\n【方式1】正向推理:")
        print("-"*70)
        forward_result = self.forward_engine.run_inference(
            ['有毛发', '吃肉', '黄褐色', '暗斑点']
        )
        
        input("\n\n按回车键开始反向推理...")
        
        print("\n【方式2】反向推理:")
        print("-"*70)
        print("（系统会提问验证特征，请根据上述特征回答）")
        backward_result = self.backward_engine.run_backward_inference()
        
        # 对比总结
        print("\n" + "="*70)
        print("对比总结")
        print("="*70)
        
        print(f"\n正向推理结果: {forward_result if forward_result else '无法识别'}")
        print(f"反向推理结果: {backward_result if backward_result else '无法识别'}")
        
        print("\n推理方式对比:")
        print("-"*70)
        print("正向推理（数据驱动）:")
        print("  • 从已知事实出发")
        print("  • 自底向上推导")
        print("  • 一次性给出所有特征")
        print("  • 适合数据完整的场景")
        
        print("\n反向推理（目标驱动）:")
        print("  • 从目标假设出发")
        print("  • 自顶向下验证")
        print("  • 按需询问必要信息")
        print("  • 适合交互式诊断场景")
        print("-"*70)
    
    def show_system_info(self):
        """显示系统信息"""
        print("\n" + "="*70)
        print("系统信息")
        print("="*70)
        
        # 知识库统计
        stats = self.knowledge_base.get_statistics()
        
        print("\n知识库统计:")
        print("-"*70)
        print(f"  规则总数: {stats['total_rules']} 条")
        
        print(f"\n  按类型统计:")
        type_counts = stats['rules_by_type']
        print(f"    • 特征规则 (优先级 100-199): {type_counts['feature_rules']} 条")
        print(f"    • 分类规则 (优先级 200-299): {type_counts['classification_rules']} 条")
        print(f"    • 识别规则 (优先级 300-399): {type_counts['recognition_rules']} 条")
        
        print(f"\n  按优先级详细统计:")
        for priority, count in sorted(stats['rules_by_priority'].items(), reverse=True):
            print(f"    • 优先级 {priority}: {count} 条规则")
        
        print("\n系统能力:")
        print("-"*70)
        print("  • 可识别动物: 7 种 (豹, 虎, 长颈鹿, 斑马, 鸵鸟, 企鹅, 信天翁)")
        print("  • 特征维度: 17 种")
        print("  • 中间分类: 4 种 (哺乳动物, 鸟类, 食肉动物, 有蹄类动物)")
        
        print("\n推理引擎:")
        print("-"*70)
        print("  • 正向推理引擎: 数据驱动，从事实推导结论")
        print("  • 反向推理引擎: 目标驱动，从假设验证事实")
        
        print("\n优先级策略 (Salience):")
        print("-"*70)
        print("  • 第1层 (100-199): 基础分类规则 - 从特征推导大类")
        print("    - 单条件规则: 100")
        print("    - 多条件规则: 110")
        print("  • 第2层 (200-299): 中间分类规则 - 从大类推导子类")
        print("    - 标准优先级: 200")
        print("  • 第3层 (300-399): 识别规则 - 从子类推导物种")
        print("    - 2条件规则: 320 (低具体性)")
        print("    - 3条件规则: 330 (中具体性)")
        print("    - 4条件规则: 340 (较高具体性)")
        print("    - 5条件规则: 350 (高具体性)")
        
        print("\n冲突消解策略:")
        print("-"*70)
        print("  • 主要策略: 按规则 salience 值排序（高优先）")
        print("  • 次要策略: 条件多的规则优先（更具体）")
        print("  • 防重复: 已触发的规则不再触发")
        
        print("\n设计原则:")
        print("-"*70)
        print("  • 分层推理: 从一般到具体的三层架构")
        print("  • 具体性优先: 条件越多优先级越高")
        print("  • 清晰分离: 不同层次使用不同数量级")
        print("  • 易于扩展: 每层留有足够的优先级空间")
        
        print("\n技术特点:")
        print("-"*70)
        print("  • 纯 Python 实现，无需外部框架")
        print("  • 模块化设计，易于扩展")
        print("  • 完整的推理过程记录")
        print("  • 智能冲突消解机制")
        print("-"*70)
    
    def show_knowledge_base(self):
        """显示知识库"""
        print("\n" + "="*70)
        print("知识库规则列表")
        print("="*70)
        
        rules = self.knowledge_base.get_all_rules()
        
        # 按优先级分组
        rules_by_priority = {}
        for rule in rules:
            priority = rule.salience
            if priority not in rules_by_priority:
                rules_by_priority[priority] = []
            rules_by_priority[priority].append(rule)
        
        # 显示规则
        for priority in sorted(rules_by_priority.keys(), reverse=True):
            priority_rules = rules_by_priority[priority]
            
            if priority == 1:
                category = "特征规则"
            elif priority == 5:
                category = "分类规则"
            else:
                category = "识别规则"
            
            print(f"\n【{category}】(优先级 = {priority})")
            print("-"*70)
            
            for i, rule in enumerate(priority_rules, 1):
                print(f"\n{i}. {rule.name}")
                print(f"   描述: {rule.description}")
                print(f"   条件数: {len(rule.conditions)}")
                print(f"   动作数: {len(rule.actions)}")
        
        print("\n" + "="*70)
    
    def show_conflict_analysis(self):
        """显示冲突消解分析"""
        print("\n" + "="*70)
        print("冲突消解分析")
        print("="*70)
        
        print("\n请先运行一次正向推理以收集冲突消解数据")
        print("或查看上次推理的冲突消解情况")
        
        self.forward_engine.conflict_resolver.analyze()
        
        stats = self.forward_engine.conflict_resolver.get_statistics()
        
        if stats['total_conflicts'] > 0:
            print(f"\n统计信息:")
            print(f"  • 总冲突次数: {stats['total_conflicts']}")
            print(f"  • 平均冲突规模: {stats['avg_conflict_size']:.2f}")
    
    def run(self):
        """运行主程序"""
        self.print_header()
        
        while True:
            try:
                self.print_menu()
                choice = input("\n请选择 (0-6): ").strip()
                
                if choice == '1':
                    self.run_forward_inference()
                elif choice == '2':
                    self.run_backward_inference()
                elif choice == '3':
                    self.compare_inference_modes()
                elif choice == '4':
                    self.show_system_info()
                elif choice == '5':
                    self.show_knowledge_base()
                elif choice == '6':
                    self.show_conflict_analysis()
                elif choice == '0':
                    print("\n" + "="*70)
                    print("感谢使用动物识别专家系统！")
                    print("="*70)
                    sys.exit(0)
                else:
                    print("\n错误：无效选择，请输入 0-6")
                
                input("\n按回车键继续...")
                
            except KeyboardInterrupt:
                print("\n\n检测到中断信号，退出系统...")
                sys.exit(0)
            except Exception as e:
                print(f"\n错误：{e}")
                input("\n按回车键继续...")


def main():
    """主函数"""
    system = AnimalExpertSystem()
    system.run()


if __name__ == "__main__":
    main()
