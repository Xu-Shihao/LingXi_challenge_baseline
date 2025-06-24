import os
import pandas as pd
import numpy as np
import torch
import json
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class MentalHealthDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 对文本进行编码
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class ChineseBERTBaseline:
    def __init__(self, model_name='hfl/chinese-bert-wwm-ext', num_labels=4, max_length=512):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"使用设备: {self.device}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
        
        # 初始化tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None  # 延迟初始化
        
        self.label_mapping = {
            0: '抑郁相关', 
            1: '双相情感障碍', 
            2: '焦虑相关', 
            3: '多动障碍'
        }
    
    def init_model(self):
        """初始化模型"""
        if self.model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                problem_type="single_label_classification"
            )
            self.model.to(self.device)
    
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
    
    def create_datasets(self, train_df, test_df):
        """创建PyTorch数据集"""
        train_dataset = MentalHealthDataset(
            train_df['text'].values,
            train_df['label'].values,
            self.tokenizer,
            self.max_length
        )
        
        test_dataset = MentalHealthDataset(
            test_df['text'].values,
            test_df['label'].values,
            self.tokenizer,
            self.max_length
        )
        
        return train_dataset, test_dataset
    
    def compute_metrics(self, eval_pred):
        """计算评估指标"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        f1_weighted = f1_score(labels, predictions, average='weighted')
        f1_macro = f1_score(labels, predictions, average='macro')
        
        return {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro
        }
    
    def train(self, train_dataset, eval_dataset=None, output_dir='models/chinesebert_finetuned'):
        """训练模型"""
        self.init_model()
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,  # 根据GPU内存调整
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=500 if eval_dataset else None,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="f1_weighted" if eval_dataset else None,
            greater_is_better=True,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to=None,  # 禁用wandb等
            seed=42,
            fp16=torch.cuda.is_available(),  # 使用半精度加速训练
        )
        
        # 初始化Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if eval_dataset else None,
        )
        
        print("开始训练ChineseBERT模型...")
        print(f"训练样本数: {len(train_dataset)}")
        if eval_dataset:
            print(f"验证样本数: {len(eval_dataset)}")
        
        # 训练模型
        trainer.train()
        
        # 保存模型
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"模型已保存到 {output_dir}")
        return trainer
    
    def evaluate(self, test_dataset):
        """评估模型"""
        if self.model is None:
            raise ValueError("模型未训练或加载，请先训练或加载模型")
        
        # 创建数据加载器
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # 预测
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="评估中"):
                inputs = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device)
                }
                
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                all_predictions.extend(torch.argmax(predictions, dim=-1).cpu().numpy())
                all_labels.extend(batch['labels'].numpy())
                all_probabilities.extend(predictions.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(all_labels, all_predictions)
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
        f1_macro = f1_score(all_labels, all_predictions, average='macro')
        
        print(f"\n=== ChineseBERT 模型评估结果 ===")
        print(f"准确率: {accuracy:.4f}")
        print(f"F1 Score (weighted): {f1_weighted:.4f}")
        print(f"F1 Score (macro): {f1_macro:.4f}")
        
        # 详细分类报告
        class_names = [self.label_mapping[i] for i in sorted(self.label_mapping.keys())]
        print("\n分类报告:")
        print(classification_report(all_labels, all_predictions, target_names=class_names))
        
        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions)
        self.plot_confusion_matrix(cm, class_names)
        
        return {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'true_labels': all_labels,
            'classification_report': classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)
        }
    
    def plot_confusion_matrix(self, cm, class_names):
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - ChineseBERT')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # 保存图片
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/chinesebert_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict(self, texts):
        """预测新文本"""
        if self.model is None:
            raise ValueError("模型未训练或加载，请先训练或加载模型")
        
        if isinstance(texts, str):
            texts = [texts]
        
        self.model.eval()
        results = []
        
        with torch.no_grad():
            for text in texts:
                # 编码文本
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # 移动到设备
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # 预测
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_label = torch.argmax(probabilities, dim=-1).item()
                
                results.append({
                    'text': text,
                    'predicted_label': predicted_label,
                    'predicted_class': self.label_mapping[predicted_label],
                    'probabilities': {self.label_mapping[i]: prob.item() for i, prob in enumerate(probabilities[0])}
                })
        
        return results
    
    def load_model(self, model_dir='models/chinesebert_finetuned'):
        """加载训练好的模型"""
        print(f"正在从 {model_dir} 加载模型...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        
        # 加载标签映射（如果存在）
        label_path = os.path.join(model_dir, 'label_mapping.json')
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                self.label_mapping = json.load(f)
                self.label_mapping = {int(k): v for k, v in self.label_mapping.items()}
        
        print("模型加载完成!")
        return self
    
    def save_label_mapping(self, output_dir):
        """单独保存标签映射"""
        label_path = os.path.join(output_dir, 'label_mapping.json')
        with open(label_path, 'w', encoding='utf-8') as f:
            json.dump(self.label_mapping, f, ensure_ascii=False, indent=2)

def main():
    """主函数：训练ChineseBERT模型"""
    
    # 创建模型
    model = ChineseBERTBaseline()
    
    # 加载数据
    print("正在加载数据...")
    train_df, test_df = model.load_data()
    
    # 创建数据集
    print("正在创建数据集...")
    train_dataset, test_dataset = model.create_datasets(train_df, test_df)
    
    # 训练模型（使用部分训练数据作为验证集）
    print("正在训练模型...")
    # 可以选择使用验证集进行早停
    trainer = model.train(train_dataset, eval_dataset=None)  # 设置为test_dataset进行验证
    
    # 保存标签映射
    model.save_label_mapping('models/chinesebert_finetuned')
    
    # 评估模型
    print("正在评估模型...")
    results = model.evaluate(test_dataset)
    
    # 保存结果
    os.makedirs('results', exist_ok=True)
    results_summary = {
        'model': 'ChineseBERT',
        'accuracy': results['accuracy'],
        'f1_weighted': results['f1_weighted'],
        'f1_macro': results['f1_macro']
    }
    
    with open('results/chinesebert_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    
    print("\nChineseBERT模型训练和评估完成!")
    print(f"结果已保存到 results/ 目录")

if __name__ == "__main__":
    main() 