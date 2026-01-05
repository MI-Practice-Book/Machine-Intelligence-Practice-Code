# test_forward.py - 测试正向推理

from forward_engine import ForwardInferenceEngine

def test_leopard():
    """测试识别豹"""
    print("="*70)
    print("测试案例：识别豹")
    print("="*70)
    
    engine = ForwardInferenceEngine()
    result = engine.run_inference(['有毛发', '吃肉', '黄褐色', '暗斑点'])
    
    print(f"\n预期结果: 豹")
    print(f"实际结果: {result}")
    print(f"测试{'通过' if result == '豹' else '失败'} ✓" if result == '豹' else "测试失败 ✗")

def test_giraffe():
    """测试识别长颈鹿"""
    print("\n\n" + "="*70)
    print("测试案例：识别长颈鹿")
    print("="*70)
    
    engine = ForwardInferenceEngine()
    result = engine.run_inference(['有毛发', '有蹄', '长腿', '长脖子', '黄褐色', '暗斑点'])
    
    print(f"\n预期结果: 长颈鹿")
    print(f"实际结果: {result}")
    print(f"测试{'通过' if result == '长颈鹿' else '失败'} ✓" if result == '长颈鹿' else "测试失败 ✗")

def test_penguin():
    """测试识别企鹅"""
    print("\n\n" + "="*70)
    print("测试案例：识别企鹅")
    print("="*70)
    
    engine = ForwardInferenceEngine()
    result = engine.run_inference(['有羽毛', '不会飞', '会游泳', '黑白两色'])
    
    print(f"\n预期结果: 企鹅")
    print(f"实际结果: {result}")
    print(f"测试{'通过' if result == '企鹅' else '失败'} ✓" if result == '企鹅' else "测试失败 ✗")

if __name__ == "__main__":
    test_leopard()
    test_giraffe()
    test_penguin()
    
    print("\n\n" + "="*70)
    print("所有测试完成！")
    print("="*70)
