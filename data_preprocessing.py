import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import re

class DataPreprocessor:
    def __init__(self, data_dir='DataSyn'):
        self.data_dir = data_dir
        # 定义4类疾病的关键词映射
        self.label_mapping = {
            0: '抑郁相关', 
            1: '双相情感障碍', 
            2: '焦虑相关', 
            3: '多动障碍'
        }
        self.keyword_mapping = {
            0: ['抑郁', '重性抑郁'],  # 抑郁相关
            1: ['双相'],  # 双相情感障碍
            2: ['焦虑', '广泛性焦虑'],  # 焦虑相关
            3: ['多动']  # 多动障碍
        }
        
    def extract_patient_id(self, filename):
        """从文件名提取patient_id"""
        # 文件格式: patient_10004com4_4_dialogue3_dialogue1_2.json
        parts = filename.split('_')
        if len(parts) >= 2:
            # 先得到10004com4，然后用com分割取第一部分10004
            patient_com = parts[1]  # 得到10004com4
            patient_id = patient_com.split('com')[0]  # 得到10004
            return patient_id
        return None
    
    def diagnosis_to_multilabel(self, diagnosis_text):
        """将诊断文本转换为多标签one-hot编码"""
        # 初始化4类标签为0
        labels = [0, 0, 0, 0]
        
        if not diagnosis_text:
            return labels
        
        # 按逗号、顿号、分号分割诊断文本
        diagnosis_parts = re.split('[，,、；;]', diagnosis_text)
        
        for part in diagnosis_parts:
            part = part.strip()
            if not part:
                continue
                
            # 检查每个类别的关键词
            for label_idx, keywords in self.keyword_mapping.items():
                for keyword in keywords:
                    if keyword in part:
                        labels[label_idx] = 1
                        break  # 找到一个关键词就够了
        
        return labels
        
    def load_data(self):
        """加载所有数据"""
        data_list = []
        
        for filename in os.listdir(self.data_dir):
            if not filename.endswith('.json'):
                continue
                
            patient_id = self.extract_patient_id(filename)
            if not patient_id:
                continue
                
            try:
                with open(os.path.join(self.data_dir, filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # 提取对话文本
                conversations = []
                diagnosis_text = None
                
                for conv in data.get('conversation', []):
                    if isinstance(conv, dict):
                        if 'doctor' in conv and 'patient' in conv:
                            # 普通对话
                            conversations.append(f"医生: {conv['doctor']}")
                            conversations.append(f"患者: {conv['patient']}")
                        elif 'doctor' in conv and '诊断结束' in conv['doctor']:
                            # 诊断结果
                            diagnosis_text = conv['doctor'].replace('诊断结束，你的诊断结果为：', '').replace('。', '').strip()
                            break
                
                if diagnosis_text is not None and conversations:
                    # 合并对话文本
                    full_text = '\n'.join(conversations)
                    
                    # 生成多标签
                    multilabels = self.diagnosis_to_multilabel(diagnosis_text)
                    
                    data_list.append({
                        'patient_id': patient_id,
                        'filename': filename,
                        'text': full_text,
                        'labels': multilabels,  # 多标签列表
                        'label_0': multilabels[0],  # 抑郁相关
                        'label_1': multilabels[1],  # 双相情感障碍
                        'label_2': multilabels[2],  # 焦虑相关
                        'label_3': multilabels[3],  # 多动障碍
                        'original_diagnosis': diagnosis_text
                    })
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
                
        return pd.DataFrame(data_list)
    
    def split_by_patient(self, df, test_size=0.2, random_state=42):
        """按患者ID分割数据，确保同一患者的数据不会同时出现在训练集和测试集中"""
        unique_patients = df['patient_id'].unique()
        
        # 简单随机分割患者（多标签分类不适合按标签分层）
        train_patients, test_patients = train_test_split(
            unique_patients, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # 根据患者ID分割数据
        train_df = df[df['patient_id'].isin(train_patients)].copy()
        test_df = df[df['patient_id'].isin(test_patients)].copy()
        
        return train_df, test_df
    
    def get_multilabel_stats(self, df):
        """获取多标签数据集统计信息"""
        stats = {
            'total_samples': len(df),
            'unique_patients': df['patient_id'].nunique(),
            'samples_per_patient': df.groupby('patient_id').size().describe().to_dict(),
        }
        
        # 计算每个标签的分布
        for i in range(4):
            label_col = f'label_{i}'
            stats[f'{self.label_mapping[i]}_count'] = int(df[label_col].sum())
            stats[f'{self.label_mapping[i]}_ratio'] = float(df[label_col].mean())
        
        # 计算标签组合的分布
        label_combinations = {}
        for _, row in df.iterrows():
            labels_tuple = tuple(row[['label_0', 'label_1', 'label_2', 'label_3']].values)
            label_combinations[str(labels_tuple)] = label_combinations.get(str(labels_tuple), 0) + 1
        
        stats['label_combinations'] = label_combinations
        
        # 计算每个样本的标签数量
        df_copy = df.copy()
        df_copy['num_labels'] = df_copy[['label_0', 'label_1', 'label_2', 'label_3']].sum(axis=1)
        stats['labels_per_sample'] = df_copy['num_labels'].value_counts().to_dict()
        
        return stats
    
    def preprocess_text(self, text):
        """文本预处理"""
        # 去除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # 去除特殊字符，保留中文、英文、数字和基本标点
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：""''（）【】\\s]', '', text)
        
        return text
    
    def save_processed_data(self, train_df, test_df, output_dir='processed_data'):
        """保存处理后的数据"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存训练集和测试集
        train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False, encoding='utf-8')
        test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False, encoding='utf-8')
        
        # 保存标签映射
        with open(os.path.join(output_dir, 'label_mapping.json'), 'w', encoding='utf-8') as f:
            json.dump(self.label_mapping, f, ensure_ascii=False, indent=2)
        
        # 保存数据集统计信息
        train_stats = self.get_multilabel_stats(train_df)
        test_stats = self.get_multilabel_stats(test_df)
        
        stats = {
            'train_stats': train_stats,
            'test_stats': test_stats,
            'label_mapping': self.label_mapping,
            'keyword_mapping': self.keyword_mapping
        }
        
        with open(os.path.join(output_dir, 'dataset_stats.json'), 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"处理后的数据已保存到 {output_dir}")
        print(f"训练集样本数: {len(train_df)}")
        print(f"测试集样本数: {len(test_df)}")
        print(f"训练集患者数: {train_df['patient_id'].nunique()}")
        print(f"测试集患者数: {test_df['patient_id'].nunique()}")
        
        # 打印多标签统计
        print("\n多标签统计:")
        for i in range(4):
            label_name = self.label_mapping[i]
            train_count = train_df[f'label_{i}'].sum()
            test_count = test_df[f'label_{i}'].sum()
            print(f"{label_name}: 训练集 {train_count}, 测试集 {test_count}")

def main():
    # 创建数据预处理器
    preprocessor = DataPreprocessor()
    
    # 加载数据
    print("正在加载数据...")
    df = preprocessor.load_data()
    
    if len(df) == 0:
        print("未找到任何有效数据！")
        return
    
    # 文本预处理
    print("正在预处理文本...")
    df['text'] = df['text'].apply(preprocessor.preprocess_text)
    
    # 按患者分割数据
    print("正在分割数据集...")
    train_df, test_df = preprocessor.split_by_patient(df)
    
    # 打印分割后的统计信息
    print(f"\n数据集分割结果:")
    print(f"总样本数: {len(df)}")
    print(f"训练集样本数: {len(train_df)}")
    print(f"测试集样本数: {len(test_df)}")
    
    # 打印多标签分布
    print("\n训练集标签分布:")
    for i in range(4):
        label_name = preprocessor.label_mapping[i]
        count = train_df[f'label_{i}'].sum()
        ratio = train_df[f'label_{i}'].mean()
        print(f"{label_name}: {count} ({ratio:.2%})")
    
    print("\n测试集标签分布:")
    for i in range(4):
        label_name = preprocessor.label_mapping[i]
        count = test_df[f'label_{i}'].sum()
        ratio = test_df[f'label_{i}'].mean()
        print(f"{label_name}: {count} ({ratio:.2%})")
    
    # 显示一些样例
    print("\n样例数据:")
    for idx, row in df.head(3).iterrows():
        print(f"诊断: {row['original_diagnosis']}")
        print(f"标签: {row['labels']} ({[preprocessor.label_mapping[i] for i, label in enumerate(row['labels']) if label == 1]})")
        print("-" * 50)
    
    # 保存处理后的数据
    preprocessor.save_processed_data(train_df, test_df)

if __name__ == "__main__":
    main() 