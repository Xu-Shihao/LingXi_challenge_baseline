import os
import pandas as pd
import numpy as np
import pickle
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import multilabel_confusion_matrix, hamming_loss, jaccard_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

class TFIDFBaseline:
    def __init__(self, model_type='logistic', max_features=5000):
        """
        TF-IDF + 分类器的多标签分类Baseline
        
        Args:
            model_type: 分类器类型 ('logistic', 'rf', 'svm')
            max_features: TF-IDF最大特征数
        """
        self.model_type = model_type
        self.max_features = max_features
        
        # 初始化TF-IDF向量化器
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            tokenizer=self.tokenize_chinese,
            lowercase=False,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        # 初始化分类器
        if model_type == 'logistic':
            base_classifier = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'rf':
            base_classifier = RandomForestClassifier(random_state=42, n_estimators=100)
        elif model_type == 'svm':
            base_classifier = SVC(random_state=42, probability=True)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 使用MultiOutputClassifier进行多标签分类
        self.classifier = MultiOutputClassifier(base_classifier)
        
        self.label_mapping = None
        self.label_columns = ['label_0', 'label_1', 'label_2', 'label_3']
        
    def tokenize_chinese(self, text):
        """中文分词"""
        return list(jieba.cut(text, cut_all=False))
    
    def load_data(self, data_dir='processed_data'):
        """加载数据"""
        train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
        test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
        
        # 加载标签映射
        with open(os.path.join(data_dir, 'label_mapping.json'), 'r', encoding='utf-8') as f:
            self.label_mapping = json.load(f)
            # 将键转换为整数
            self.label_mapping = {int(k): v for k, v in self.label_mapping.items()}
        
        return train_df, test_df
    
    def train(self, train_df, use_grid_search=False):
        """训练多标签模型"""
        print(f"正在训练 多标签 TF-IDF + {self.model_type.upper()} 模型...")
        
        X_train_text = train_df['text'].values
        # 获取多标签目标
        y_train = train_df[self.label_columns].values
        
        # 提取TF-IDF特征
        print("正在提取TF-IDF特征...")
        X_train = self.vectorizer.fit_transform(X_train_text)
        
        print(f"特征维度: {X_train.shape}")
        print(f"训练样本数: {len(y_train)}")
        print(f"标签分布:")
        for i, col in enumerate(self.label_columns):
            label_name = self.label_mapping[i]
            count = y_train[:, i].sum()
            ratio = count / len(y_train)
            print(f"  {label_name}: {count} ({ratio:.2%})")
        
        # 网格搜索超参数（可选）
        if use_grid_search:
            print("正在进行网格搜索...")
            if self.model_type == 'logistic':
                base_classifier = LogisticRegression(random_state=42, max_iter=1000)
                param_grid = {'estimator__C': [0.1, 1.0, 10.0]}
            elif self.model_type == 'rf':
                base_classifier = RandomForestClassifier(random_state=42)
                param_grid = {'estimator__n_estimators': [50, 100, 200], 'estimator__max_depth': [None, 10, 20]}
            elif self.model_type == 'svm':
                base_classifier = SVC(random_state=42, probability=True)
                param_grid = {'estimator__C': [0.1, 1.0, 10.0], 'estimator__gamma': ['scale', 'auto']}
            
            self.classifier = MultiOutputClassifier(base_classifier)
            grid_search = GridSearchCV(
                self.classifier,
                param_grid,
                cv=3,  # 减少CV折数以节省时间
                scoring='f1_weighted',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            self.classifier = grid_search.best_estimator_
            print(f"最佳参数: {grid_search.best_params_}")
        else:
            # 直接训练
            self.classifier.fit(X_train, y_train)
        
        print("多标签模型训练完成!")
        
        # 提取并显示重要特征
        self.extract_top_features()
        
        return self
    
    def extract_top_features(self, top_k=20):
        """提取并显示top k重要的TF-IDF特征"""
        print(f"\n=== Top {top_k} 重要特征分析 ===")
        
        # 获取特征名称
        feature_names = self.vectorizer.get_feature_names_out()
        
        # 对于多标签分类，我们需要分析每个标签的重要特征
        class_names = [self.label_mapping[i] for i in sorted(self.label_mapping.keys())]
        
        # 存储所有特征重要性信息
        all_features_info = {}
        
        for i, label_name in enumerate(class_names):
            print(f"\n--- {label_name} ---")
            
            # 获取当前标签对应的分类器
            if hasattr(self.classifier, 'estimators_'):
                current_classifier = self.classifier.estimators_[i]
            else:
                current_classifier = self.classifier
            
            # 根据分类器类型提取特征重要性
            if self.model_type == 'logistic':
                # 对于逻辑回归，使用系数的绝对值
                if hasattr(current_classifier, 'coef_'):
                    importance_scores = np.abs(current_classifier.coef_[0])
                else:
                    print(f"警告: 无法获取 {label_name} 的特征系数")
                    continue
            elif self.model_type == 'rf':
                # 对于随机森林，使用特征重要性
                if hasattr(current_classifier, 'feature_importances_'):
                    importance_scores = current_classifier.feature_importances_
                else:
                    print(f"警告: 无法获取 {label_name} 的特征重要性")
                    continue
            else:
                print(f"暂不支持 {self.model_type} 的特征重要性提取")
                continue
            
            # 获取top k特征的索引
            top_indices = np.argsort(importance_scores)[::-1][:top_k]
            
            # 存储当前标签的特征信息
            label_features = []
            
            print(f"Top {top_k} 重要特征:")
            for rank, idx in enumerate(top_indices, 1):
                feature_name = feature_names[idx]
                score = importance_scores[idx]
                print(f"  {rank:2d}. {feature_name:20s} (重要性: {score:.4f})")
                
                label_features.append({
                    'rank': rank,
                    'feature': feature_name,
                    'importance': float(score)
                })
            
            all_features_info[label_name] = label_features
        
        # 保存特征重要性信息到文件
        self.save_feature_importance(all_features_info)
        
        return all_features_info
    
    def save_feature_importance(self, features_info):
        """保存特征重要性信息"""
        os.makedirs('results', exist_ok=True)
        
        # 保存为JSON文件
        with open(f'results/tfidf_{self.model_type}_top_features.json', 'w', encoding='utf-8') as f:
            json.dump(features_info, f, ensure_ascii=False, indent=2)
        
        # 同时保存为CSV文件方便查看
        all_features_data = []
        for label_name, features in features_info.items():
            for feature_info in features:
                all_features_data.append({
                    'Label': label_name,
                    'Rank': feature_info['rank'],
                    'Feature': feature_info['feature'],
                    'Importance': feature_info['importance']
                })
        
        features_df = pd.DataFrame(all_features_data)
        features_df.to_csv(f'results/tfidf_{self.model_type}_top_features.csv', index=False, encoding='utf-8')
        
        print(f"\n特征重要性已保存到:")
        print(f"  - results/tfidf_{self.model_type}_top_features.json")
        print(f"  - results/tfidf_{self.model_type}_top_features.csv")
    
    def evaluate(self, test_df):
        """评估多标签模型"""
        X_test_text = test_df['text'].values
        y_test = test_df[self.label_columns].values
        
        # 提取测试集特征
        X_test = self.vectorizer.transform(X_test_text)
        
        # 预测
        y_pred = self.classifier.predict(X_test)
        
        # 计算多标签评估指标
        hamming_loss_score = hamming_loss(y_test, y_pred)
        jaccard_score_micro = jaccard_score(y_test, y_pred, average='micro')
        jaccard_score_macro = jaccard_score(y_test, y_pred, average='macro')
        f1_micro = f1_score(y_test, y_pred, average='micro')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        precision_micro = precision_score(y_test, y_pred, average='micro')
        recall_micro = recall_score(y_test, y_pred, average='micro')
        
        print(f"\n=== 多标签 TF-IDF + {self.model_type.upper()} 模型评估结果 ===")
        print(f"Hamming Loss: {hamming_loss_score:.4f}")
        print(f"Jaccard Score (micro): {jaccard_score_micro:.4f}")
        print(f"Jaccard Score (macro): {jaccard_score_macro:.4f}")
        print(f"F1 Score (micro): {f1_micro:.4f}")
        print(f"F1 Score (macro): {f1_macro:.4f}")
        print(f"F1 Score (weighted): {f1_weighted:.4f}")
        print(f"Precision (micro): {precision_micro:.4f}")
        print(f"Recall (micro): {recall_micro:.4f}")
        
        # 每个标签的详细评估
        print("\n各标签详细评估:")
        class_names = [self.label_mapping[i] for i in sorted(self.label_mapping.keys())]
        for i, label_name in enumerate(class_names):
            y_true_label = y_test[:, i]
            y_pred_label = y_pred[:, i]
            
            f1_label = f1_score(y_true_label, y_pred_label)
            precision_label = precision_score(y_true_label, y_pred_label, zero_division=0)
            recall_label = recall_score(y_true_label, y_pred_label, zero_division=0)
            
            print(f"{label_name}:")
            print(f"  F1: {f1_label:.4f}, Precision: {precision_label:.4f}, Recall: {recall_label:.4f}")
        
        # 绘制每个标签的混淆矩阵
        self.plot_multilabel_confusion_matrix(y_test, y_pred, class_names)
        
        return {
            'hamming_loss': hamming_loss_score,
            'jaccard_micro': jaccard_score_micro,
            'jaccard_macro': jaccard_score_macro,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'y_pred': y_pred
        }
    
    def plot_multilabel_confusion_matrix(self, y_test, y_pred, class_names):
        """绘制多标签混淆矩阵"""
        # 计算每个标签的混淆矩阵
        cm_multilabel = multilabel_confusion_matrix(y_test, y_pred)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i, (cm, label_name) in enumerate(zip(cm_multilabel, class_names)):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{label_name}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('True')
        
        plt.suptitle(f'Multilabel Confusion Matrix - TF-IDF + {self.model_type.upper()}')
        plt.tight_layout()
        
        # 保存图片
        os.makedirs('results', exist_ok=True)
        plt.savefig(f'results/tfidf_{self.model_type}_multilabel_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict(self, texts):
        """预测新文本的多标签"""
        if isinstance(texts, str):
            texts = [texts]
        
        X = self.vectorizer.transform(texts)
        predictions = self.classifier.predict(X)
        
        results = []
        for i, text in enumerate(texts):
            pred_labels = predictions[i]
            
            # 转换为标签名称
            predicted_classes = [self.label_mapping[j] for j, label in enumerate(pred_labels) if label == 1]
            
            results.append({
                'text': text,
                'predicted_labels': pred_labels.tolist(),
                'predicted_classes': predicted_classes
            })
        
        return results
    
    def save_model(self, model_dir='models/saved'):
        """保存模型"""
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存TF-IDF向量化器
        with open(os.path.join(model_dir, f'tfidf_{self.model_type}_vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # 保存分类器
        with open(os.path.join(model_dir, f'tfidf_{self.model_type}_classifier.pkl'), 'wb') as f:
            pickle.dump(self.classifier, f)
        
        # 保存标签映射
        with open(os.path.join(model_dir, f'tfidf_{self.model_type}_labels.json'), 'w', encoding='utf-8') as f:
            json.dump(self.label_mapping, f, ensure_ascii=False, indent=2)
        
        print(f"多标签模型已保存到 {model_dir}")
    
    def load_model(self, model_dir='models/saved'):
        """加载模型"""
        # 加载TF-IDF向量化器
        with open(os.path.join(model_dir, f'tfidf_{self.model_type}_vectorizer.pkl'), 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        # 加载分类器
        with open(os.path.join(model_dir, f'tfidf_{self.model_type}_classifier.pkl'), 'rb') as f:
            self.classifier = pickle.load(f)
        
        # 加载标签映射
        with open(os.path.join(model_dir, f'tfidf_{self.model_type}_labels.json'), 'r', encoding='utf-8') as f:
            self.label_mapping = json.load(f)
            self.label_mapping = {int(k): v for k, v in self.label_mapping.items()}
        
        print(f"多标签模型已从 {model_dir} 加载")
        return self

def main():
    """主函数：训练和评估所有多标签TF-IDF模型"""
    
    # 测试不同的分类器
    models = ['logistic', 'rf']  # 'svm' 在多标签情况下可能会很慢
    results = {}
    
    for model_type in models:
        print(f"\n{'='*50}")
        print(f"训练多标签 TF-IDF + {model_type.upper()} 模型")
        print(f"{'='*50}")
        
        # 创建模型
        model = TFIDFBaseline(model_type=model_type)
        
        # 加载数据
        train_df, test_df = model.load_data()
        
        # 训练模型
        model.train(train_df, use_grid_search=True)
        
        # 评估模型
        eval_results = model.evaluate(test_df)
        results[model_type] = eval_results
        
        # 保存模型
        model.save_model()
        
        print(f"多标签 {model_type.upper()} 模型完成!")
    
    # 比较结果
    print(f"\n{'='*50}")
    print("所有多标签模型结果比较")
    print(f"{'='*50}")
    
    comparison_data = []
    for model_name, result in results.items():
        comparison_data.append({
            'Model': f'TF-IDF + {model_name.upper()}',
            'Hamming Loss': result['hamming_loss'],
            'Jaccard (Micro)': result['jaccard_micro'],
            'F1 (Micro)': result['f1_micro'],
            'F1 (Macro)': result['f1_macro'],
            'F1 (Weighted)': result['f1_weighted'],
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # 保存比较结果
    comparison_df.to_csv('results/tfidf_multilabel_models_comparison.csv', index=False)
    print(f"\n结果已保存到 results/tfidf_multilabel_models_comparison.csv")

if __name__ == "__main__":
    main() 