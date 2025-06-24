import json
import os
from collections import Counter

def create_4class_mapping():
    """
    将复杂的诊断结果映射为4个主要类别：
    0: 抑郁相关 (重性抑郁发作)
    1: 双相情感障碍 (双相I型/II型)
    2: 焦虑相关 (广泛性焦虑障碍)
    3: 多动障碍
    
    根据主要疾病分类，优先级：双相 > 抑郁 > 焦虑 > 多动
    """
    
    def classify_diagnosis(diagnosis):
        diagnosis = diagnosis.strip()
        
        if not diagnosis:  # 空诊断
            return 3  # 默认归类为多动障碍
            
        # 优先级分类
        if '双相' in diagnosis:
            return 1  # 双相情感障碍
        elif '重性抑郁' in diagnosis:
            return 0  # 抑郁相关
        elif '广泛性焦虑' in diagnosis:
            return 2  # 焦虑相关
        elif '多动' in diagnosis:
            return 3  # 多动障碍
        else:
            return 3  # 其他默认归类为多动障碍
    
    return classify_diagnosis

# 测试分类映射
if __name__ == "__main__":
    classify_func = create_4class_mapping()
    
    # 读取所有诊断并分类
    diagnoses = []
    for filename in os.listdir('DataSyn'):
        if filename.endswith('.json'):
            with open(f'DataSyn/{filename}', 'r', encoding='utf-8') as f:
                data = json.load(f)
                for conv in data.get('conversation', []):
                    if isinstance(conv, dict) and 'doctor' in conv:
                        if '诊断结束，你的诊断结果为：' in conv['doctor']:
                            diagnosis = conv['doctor'].replace('诊断结束，你的诊断结果为：', '').replace('。', '').strip()
                            diagnoses.append(diagnosis)
    
    # 统计4分类结果
    label_mapping = {0: '抑郁相关', 1: '双相情感障碍', 2: '焦虑相关', 3: '多动障碍'}
    class_counts = Counter()
    
    for diagnosis in diagnoses:
        label = classify_func(diagnosis)
        class_counts[label] += 1
    
    print("4分类结果统计:")
    for label, name in label_mapping.items():
        count = class_counts[label]
        print(f"{label}: {name} - {count} 样本 ({count/len(diagnoses)*100:.1f}%)")
    
    print(f"\n总样本数: {len(diagnoses)}")
    print(f"类别平衡度: {min(class_counts.values())}/{max(class_counts.values())} = {min(class_counts.values())/max(class_counts.values()):.2f}") 