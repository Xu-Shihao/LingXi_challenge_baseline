import os
import pandas as pd
import numpy as np
import json
import time
import requests
from typing import List, Dict
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class Qwen3ZeroShot:
    def __init__(self, api_base="http://127.0.0.1:9019/v1", model_name="/tcci_mnt/shihao/models/Qwen3-8B"):
        self.api_base = api_base
        self.model_name = model_name
        self.headers = {
            "Content-Type": "application/json",
        }
        
        self.label_mapping = {
            0: '抑郁相关', 
            1: '双相情感障碍', 
            2: '焦虑相关', 
            3: '多动障碍'
        }
        
        # 反向映射用于解析模型输出
        self.reverse_label_mapping = {
            '抑郁相关': 0,
            '抑郁': 0,
            '重性抑郁': 0,
            '抑郁症': 0,
            '双相情感障碍': 1,
            '双相': 1,
            '双相I型': 1,
            '双相II型': 1,
            '双相障碍': 1,
            '焦虑相关': 2,
            '焦虑': 2,
            '焦虑症': 2,
            '广泛性焦虑': 2,
            '焦虑障碍': 2,
            '多动障碍': 3,
            '多动': 3,
            'ADHD': 3,
            '注意缺陷多动障碍': 3
        }
        
        # 验证API连接
        self.test_connection()
    
    def test_connection(self):
        """测试API连接"""
        try:
            response = requests.get(f"{self.api_base}/models", headers=self.headers, timeout=10)
            if response.status_code == 200:
                print(f"✓ API连接成功: {self.api_base}")
                print(f"✓ 模型: {self.model_name}")
            else:
                print(f"⚠️ API连接警告: 状态码 {response.status_code}")
        except Exception as e:
            print(f"❌ API连接失败: {e}")
            print("请确保Qwen3模型服务正在运行")
    
    def create_prompt(self, dialogue_text):
        """创建用于诊断分类的提示"""
        
        prompt = f"""你是一位专业的精神科医生，请根据以下医患对话内容，判断患者最可能的精神疾病诊断类别。

对话内容：
{dialogue_text}

请从以下4个类别中选择一个最符合的诊断：
1. 抑郁相关 - 包括重性抑郁发作等抑郁症状为主的疾病
2. 双相情感障碍 - 包括双相I型、双相II型等情绪波动明显的疾病  
3. 焦虑相关 - 包括广泛性焦虑障碍等以焦虑症状为主的疾病
4. 多动障碍 - 包括注意缺陷多动障碍(ADHD)等注意力和行为问题

要求：
1. 仔细分析对话中患者的症状描述
2. 考虑症状的严重程度、持续时间和功能影响
3. 只输出一个类别名称，如："抑郁相关"、"双相情感障碍"、"焦虑相关"或"多动障碍"
4. 不要输出其他解释或分析

诊断类别："""
        
        return prompt
    
    def call_api(self, prompt, max_retries=3, retry_delay=1):
        """调用Qwen3 API"""
        
        data = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.1,  # 低温度保证输出稳定
            "max_tokens": 50,
            "top_p": 0.9
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=self.headers,
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if 'choices' in result and len(result['choices']) > 0:
                        content = result['choices'][0]['message']['content'].strip()
                        return content
                    else:
                        print(f"API响应格式错误: {result}")
                        
                else:
                    print(f"API请求失败: 状态码 {response.status_code}")
                    print(f"响应内容: {response.text}")
                    
            except Exception as e:
                print(f"API调用异常 (尝试 {attempt + 1}/{max_retries}): {e}")
                
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数退避
        
        return None
    
    def parse_prediction(self, response_text):
        """解析模型输出，映射为数字标签"""
        if not response_text:
            return 3  # 默认返回多动障碍
        
        response_text = response_text.strip().lower()
        
        # 直接匹配
        for key, value in self.reverse_label_mapping.items():
            if key.lower() in response_text:
                return value
        
        # 模糊匹配
        if '抑郁' in response_text:
            return 0
        elif '双相' in response_text:
            return 1
        elif '焦虑' in response_text:
            return 2
        elif '多动' in response_text or 'adhd' in response_text:
            return 3
        
        # 默认返回多动障碍
        print(f"无法解析的响应: {response_text}")
        return 3
    
    def load_data(self, data_dir='processed_data'):
        """加载预处理后的数据"""
        train_path = os.path.join(data_dir, 'train.csv')
        test_path = os.path.join(data_dir, 'test.csv')
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(f"请先运行data_preprocessing.py生成处理后的数据")
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # 加载标签映射
        with open(os.path.join(data_dir, 'label_mapping.json'), 'r', encoding='utf-8') as f:
            self.label_mapping = json.load(f)
            # 将键转换为整数
            self.label_mapping = {int(k): v for k, v in self.label_mapping.items()}
        
        return train_df, test_df
    
    def predict_batch(self, texts, batch_size=1, save_interval=50):
        """批量预测（zero-shot不需要训练）"""
        predictions = []
        failed_indices = []
        
        # 尝试加载之前的进度
        progress_file = 'qwen3_progress.json'
        start_idx = 0
        
        if os.path.exists(progress_file):
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
                predictions = progress.get('predictions', [])
                failed_indices = progress.get('failed_indices', [])
                start_idx = len(predictions)
                print(f"从第 {start_idx} 个样本继续...")
        
        total = len(texts)
        
        for i in tqdm(range(start_idx, total), desc="Zero-shot预测"):
            text = texts[i]
            
            # 创建提示
            prompt = self.create_prompt(text)
            
            # 调用API
            response = self.call_api(prompt)
            
            if response:
                # 解析预测结果
                pred_label = self.parse_prediction(response)
                predictions.append({
                    'index': i,
                    'prediction': pred_label,
                    'response': response,
                    'confidence': 1.0  # Zero-shot默认置信度
                })
            else:
                failed_indices.append(i)
                predictions.append({
                    'index': i,
                    'prediction': 3,  # 默认标签
                    'response': None,
                    'confidence': 0.0
                })
            
            # 定期保存进度
            if (i + 1) % save_interval == 0:
                progress = {
                    'predictions': predictions,
                    'failed_indices': failed_indices,
                    'completed': i + 1,
                    'total': total
                }
                with open(progress_file, 'w', encoding='utf-8') as f:
                    json.dump(progress, f, ensure_ascii=False, indent=2)
                print(f"进度已保存: {i + 1}/{total}")
            
            # 添加延迟避免API限制
            time.sleep(0.1)
        
        # 删除进度文件
        if os.path.exists(progress_file):
            os.remove(progress_file)
        
        print(f"预测完成! 成功: {len(predictions) - len(failed_indices)}, 失败: {len(failed_indices)}")
        
        return predictions
    
    def evaluate(self, test_df):
        """评估模型"""
        print("开始Qwen3 Zero-shot评估...")
        
        # 批量预测
        predictions_data = self.predict_batch(test_df['text'].tolist())
        
        # 提取预测结果
        y_pred = [item['prediction'] for item in predictions_data]
        y_true = test_df['label'].tolist()
        
        # 计算指标
        accuracy = accuracy_score(y_true, y_pred)
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        
        print(f"\n=== Qwen3 Zero-shot 模型评估结果 ===")
        print(f"准确率: {accuracy:.4f}")
        print(f"F1 Score (weighted): {f1_weighted:.4f}")
        print(f"F1 Score (macro): {f1_macro:.4f}")
        
        # 详细分类报告
        class_names = [self.label_mapping[i] for i in sorted(self.label_mapping.keys())]
        print("\n分类报告:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        self.plot_confusion_matrix(cm, class_names)
        
        # 保存详细预测结果
        results_df = test_df.copy()
        results_df['predicted_label'] = y_pred
        results_df['predicted_class'] = [self.label_mapping[pred] for pred in y_pred]
        results_df['qwen3_response'] = [item['response'] for item in predictions_data]
        results_df['confidence'] = [item['confidence'] for item in predictions_data]
        
        os.makedirs('results', exist_ok=True)
        results_df.to_csv('results/qwen3_detailed_results.csv', index=False, encoding='utf-8')
        
        return {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'predictions': y_pred,
            'true_labels': y_true,
            'detailed_predictions': predictions_data,
            'classification_report': classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        }
    
    def plot_confusion_matrix(self, cm, class_names):
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - Qwen3 Zero-shot')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # 保存图片
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/qwen3_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict(self, texts):
        """预测新文本"""
        if isinstance(texts, str):
            texts = [texts]
        
        results = []
        for text in texts:
            prompt = self.create_prompt(text)
            response = self.call_api(prompt)
            
            if response:
                pred_label = self.parse_prediction(response)
                results.append({
                    'text': text,
                    'predicted_label': pred_label,
                    'predicted_class': self.label_mapping[pred_label],
                    'raw_response': response,
                    'confidence': 1.0
                })
            else:
                results.append({
                    'text': text,
                    'predicted_label': 3,
                    'predicted_class': self.label_mapping[3],
                    'raw_response': None,
                    'confidence': 0.0
                })
        
        return results
    
    def analyze_failure_cases(self, results):
        """分析失败案例"""
        failed_cases = []
        for item in results['detailed_predictions']:
            if item['confidence'] == 0.0:
                failed_cases.append(item)
        
        print(f"\n失败案例分析: {len(failed_cases)} 个样本")
        if failed_cases:
            print("失败原因可能包括:")
            print("1. API超时")
            print("2. 网络连接问题") 
            print("3. 模型服务异常")
            print("4. 响应格式错误")
        
        return failed_cases

def main():
    """主函数：运行Qwen3 Zero-shot评估"""
    
    # 创建模型
    model = Qwen3ZeroShot()
    
    # 加载数据
    print("正在加载数据...")
    train_df, test_df = model.load_data()
    
    # 由于是zero-shot，我们只使用测试集的一个小子集来快速验证
    # 可以根据需要调整样本数量
    sample_size = min(100, len(test_df))  # 使用前100个样本进行测试
    test_sample = test_df.head(sample_size).copy()
    
    print(f"使用 {sample_size} 个测试样本进行Zero-shot评估...")
    
    # 评估模型
    results = model.evaluate(test_sample)
    
    # 分析失败案例
    model.analyze_failure_cases(results)
    
    # 保存结果
    os.makedirs('results', exist_ok=True)
    results_summary = {
        'model': 'Qwen3 Zero-shot',
        'sample_size': sample_size,
        'accuracy': results['accuracy'],
        'f1_weighted': results['f1_weighted'],
        'f1_macro': results['f1_macro']
    }
    
    with open('results/qwen3_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    
    print("\nQwen3 Zero-shot评估完成!")
    print(f"结果已保存到 results/ 目录")
    
    # 示例预测
    print("\n=== 示例预测 ===")
    sample_text = test_sample.iloc[0]['text']
    pred_result = model.predict(sample_text)
    print(f"输入文本: {sample_text[:200]}...")
    print(f"预测结果: {pred_result[0]['predicted_class']}")
    print(f"原始响应: {pred_result[0]['raw_response']}")

if __name__ == "__main__":
    main() 