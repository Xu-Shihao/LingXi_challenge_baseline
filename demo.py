#!/usr/bin/env python3
"""
精神疾病辅助诊断结果预测 - 演示脚本

展示如何使用训练好的baseline模型进行预测
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.tfidf_baseline import TFIDFBaseline

def demo_tfidf_prediction():
    """演示TF-IDF模型预测"""
    
    print("="*60)
    print("精神疾病辅助诊断结果预测 - TF-IDF模型演示")
    print("="*60)
    
    # 加载最佳TF-IDF模型（逻辑回归）
    model = TFIDFBaseline(model_type='logistic')
    
    try:
        model.load_model('models/saved')
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("请先运行 python models/tfidf_baseline.py 训练模型")
        return
    
    # 示例医患对话
    sample_dialogues = [
        {
            "description": "双相情感障碍案例",
            "text": """医生: 你好，最近感觉怎么样？
患者: 最近情绪波动很大，有时候特别兴奋，精力充沛，觉得自己什么都能做到。但过几天又会突然变得很沮丧，什么都不想做。
医生: 这种情况持续多久了？
患者: 大概有三个月了，兴奋的时候会连续几天不睡觉，也不觉得累。沮丧的时候就整天躺在床上，觉得生活没有意义。
医生: 兴奋的时候有没有做过一些冲动的事情？
患者: 有，上次兴奋的时候一晚上花了好几万块钱买东西，现在想起来觉得很后悔。"""
        },
        {
            "description": "抑郁症案例", 
            "text": """医生: 你这次来是因为什么问题？
患者: 我最近几个月一直感觉很沮丧，对什么都提不起兴趣，以前喜欢的事情现在都觉得没意思。
医生: 睡眠怎么样？
患者: 很糟糕，经常失眠，即使睡着了也很容易醒，白天没精神。食欲也不好，体重减轻了很多。
医生: 有没有觉得绝望或者想过伤害自己？
患者: 有时候会觉得活着很累，觉得自己一无是处，有时候确实会有这样的想法。"""
        },
        {
            "description": "焦虑症案例",
            "text": """医生: 你最近有什么困扰？
患者: 我总是担心，担心工作出错，担心家人的安全，担心未来会发生不好的事情。
医生: 这种担心影响到你的日常生活了吗？
患者: 是的，我经常心跳加速，手心出汗，坐立不安。晚上也睡不好，脑子里总是想各种可能出现的问题。
医生: 有什么特别让你焦虑的事情吗？
患者: 没有具体的事情，就是总觉得不安，好像随时会有坏事发生。集中注意力也变得很困难。"""
        },
        {
            "description": "多动症案例",
            "text": """医生: 你在工作或学习中遇到什么困难？
患者: 我很难专心，总是走神，开会的时候听着听着就想别的事情去了。
医生: 这种情况从什么时候开始的？
患者: 从小就这样，上学的时候老师经常说我注意力不集中。现在工作了还是这样，任务总是拖延，很难按时完成。
医生: 平时坐着的时候感觉怎么样？
患者: 很难安静地坐着，总想动来动去。开会的时候我会不停地转笔或者抖腿，自己也控制不住。"""
        }
    ]
    
    print(f"\n正在对 {len(sample_dialogues)} 个示例进行预测...\n")
    
    for i, dialogue in enumerate(sample_dialogues, 1):
        print(f"示例 {i}: {dialogue['description']}")
        print(f"对话内容: {dialogue['text'][:100]}...")
        
        # 进行预测
        results = model.predict(dialogue['text'])
        result = results[0]
        
        print(f"预测结果: {result['predicted_class']}")
        print("置信度分布:")
        for class_name, prob in result['probabilities'].items():
            print(f"  {class_name}: {prob:.3f}")
        print("-" * 50)
    
    # 交互式预测
    print("\n" + "="*60)
    print("交互式预测 (输入 'quit' 退出)")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n请输入医患对话内容: ")
            if user_input.lower() in ['quit', 'exit', '退出']:
                break
            
            if not user_input.strip():
                print("请输入有效的对话内容")
                continue
                
            results = model.predict(user_input)
            result = results[0]
            
            print(f"\n预测结果: {result['predicted_class']}")
            print("置信度分布:")
            for class_name, prob in result['probabilities'].items():
                print(f"  {class_name}: {prob:.3f}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"预测出错: {e}")
    
    print("\n演示结束，感谢使用！")

def show_model_performance():
    """显示模型性能"""
    
    print("\n" + "="*60)
    print("模型性能总结")
    print("="*60)
    
    try:
        import pandas as pd
        
        # 读取TF-IDF模型比较结果
        tfidf_results = pd.read_csv('results/tfidf_models_comparison.csv', index_col=0)
        print("\nTF-IDF + 传统机器学习模型结果:")
        print(tfidf_results.round(4))
        
        print("\n数据集信息:")
        print("- 训练集: 1207 样本 (429 患者)")
        print("- 测试集: 293 样本 (110 患者)")
        print("- 类别分布: 双相情感障碍(48.7%) > 抑郁相关(38.1%) > 焦虑相关(9.2%) > 多动障碍(4.0%)")
        
        print("\n主要特点:")
        print("- 使用jieba进行中文分词")
        print("- TF-IDF特征提取 (5000维)")
        print("- 网格搜索优化超参数")
        print("- 处理类别不平衡问题")
        
    except Exception as e:
        print(f"无法读取结果文件: {e}")

def main():
    """主函数"""
    
    print("精神疾病辅助诊断结果预测系统")
    print("支持4个诊断类别:")
    print("0: 抑郁相关 (重性抑郁发作等)")
    print("1: 双相情感障碍 (双相I型、双相II型等)")  
    print("2: 焦虑相关 (广泛性焦虑障碍等)")
    print("3: 多动障碍 (ADHD等)")
    
    # 显示模型性能
    show_model_performance()
    
    # 运行TF-IDF演示
    demo_tfidf_prediction()

if __name__ == "__main__":
    main() 