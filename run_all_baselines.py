#!/usr/bin/env python3
"""
精神疾病辅助诊断结果预测 - Baseline运行脚本

运行所有baseline模型：
1. 数据预处理
2. TF-IDF + 传统机器学习
3. ChineseBERT fine-tuning
4. Qwen3 zero-shot

作者: Assistant
"""

import os
import sys
import json
import pandas as pd
import argparse
from datetime import datetime
import traceback

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_data_preprocessing():
    """运行数据预处理"""
    print("\n" + "="*60)
    print("步骤 1: 数据预处理")
    print("="*60)
    
    try:
        from data_preprocessing import main as preprocess_main
        preprocess_main()
        print("✓ 数据预处理完成")
        return True
    except Exception as e:
        print(f"❌ 数据预处理失败: {e}")
        traceback.print_exc()
        return False

def run_tfidf_baseline():
    """运行TF-IDF baseline"""
    print("\n" + "="*60)
    print("步骤 2: TF-IDF Baseline")
    print("="*60)
    
    try:
        from models.tfidf_baseline import main as tfidf_main
        tfidf_main()
        print("✓ TF-IDF baseline完成")
        return True
    except Exception as e:
        print(f"❌ TF-IDF baseline失败: {e}")
        traceback.print_exc()
        return False

def run_chinesebert_baseline():
    """运行ChineseBERT baseline"""
    print("\n" + "="*60)
    print("步骤 3: ChineseBERT Baseline")
    print("="*60)
    
    try:
        from models.chinesebert_baseline import main as bert_main
        bert_main()
        print("✓ ChineseBERT baseline完成")
        return True
    except Exception as e:
        print(f"❌ ChineseBERT baseline失败: {e}")
        traceback.print_exc()
        return False

def run_qwen3_baseline():
    """运行Qwen3 zero-shot baseline"""
    print("\n" + "="*60)
    print("步骤 4: Qwen3 Zero-shot Baseline")
    print("="*60)
    
    try:
        from models.qwen3_zeroshot import main as qwen3_main
        qwen3_main()
        print("✓ Qwen3 zero-shot baseline完成")
        return True
    except Exception as e:
        print(f"❌ Qwen3 zero-shot baseline失败: {e}")
        traceback.print_exc()
        return False

def display_feature_importance():
    """显示TF-IDF模型的特征重要性信息"""
    results_dir = 'results'
    
    # 查找所有特征重要性文件
    feature_files = []
    for filename in os.listdir(results_dir):
        if filename.startswith('tfidf_') and filename.endswith('_top_features.json'):
            feature_files.append(filename)
    
    if not feature_files:
        print("未找到特征重要性文件")
        return
    
    print(f"\n{'='*60}")
    print("TF-IDF 模型特征重要性分析")
    print(f"{'='*60}")
    
    for feature_file in feature_files:
        model_type = feature_file.replace('tfidf_', '').replace('_top_features.json', '')
        print(f"\n--- {model_type.upper()} 模型 ---")
        
        feature_path = os.path.join(results_dir, feature_file)
        try:
            with open(feature_path, 'r', encoding='utf-8') as f:
                features_info = json.load(f)
            
            for label_name, features in features_info.items():
                print(f"\n{label_name} - Top 10 重要特征:")
                for feature_info in features[:10]:  # 显示前10个
                    print(f"  {feature_info['rank']:2d}. {feature_info['feature']:25s} (重要性: {feature_info['importance']:.4f})")
        except Exception as e:
            print(f"读取特征文件 {feature_file} 时出错: {e}")

def collect_and_compare_results():
    """收集并比较所有模型的结果"""
    print("\n" + "="*60)
    print("步骤 5: 结果收集与比较")
    print("="*60)
    
    results_dir = 'results'
    if not os.path.exists(results_dir):
        print("❌ 结果目录不存在")
        return False
    
    # 收集结果
    all_results = {}
    
    # TF-IDF结果（多标签版本）
    tfidf_comparison_file = os.path.join(results_dir, 'tfidf_multilabel_models_comparison.csv')
    if os.path.exists(tfidf_comparison_file):
        tfidf_df = pd.read_csv(tfidf_comparison_file)
        for idx, row in tfidf_df.iterrows():
            model_name = row['Model']
            all_results[model_name] = {
                'Hamming Loss': row['Hamming Loss'],
                'Jaccard (Micro)': row['Jaccard (Micro)'],
                'F1 (Micro)': row['F1 (Micro)'],
                'F1 (Macro)': row['F1 (Macro)'],
                'F1 (Weighted)': row['F1 (Weighted)']
            }
    
    # 显示特征重要性信息
    display_feature_importance()
    
    # 兼容旧版TF-IDF结果
    old_tfidf_file = os.path.join(results_dir, 'tfidf_models_comparison.csv')
    if os.path.exists(old_tfidf_file):
        old_tfidf_df = pd.read_csv(old_tfidf_file, index_col=0)  # 指定第一列为索引
        for model_type in old_tfidf_df.index:
            model_name = f"TF-IDF + {str(model_type).upper()}"
            row = old_tfidf_df.loc[model_type]
            all_results[model_name] = {
                'Accuracy': row['Accuracy'],
                'F1 (weighted)': row['F1 (weighted)'],
                'F1 (macro)': row['F1 (macro)']
            }
    
    # ChineseBERT结果
    bert_results_file = os.path.join(results_dir, 'chinesebert_results.json')
    if os.path.exists(bert_results_file):
        with open(bert_results_file, 'r', encoding='utf-8') as f:
            bert_results = json.load(f)
            all_results['ChineseBERT'] = {
                'Accuracy': bert_results['accuracy'],
                'F1 (weighted)': bert_results['f1_weighted'],
                'F1 (macro)': bert_results['f1_macro']
            }
    
    # Qwen3结果
    qwen3_results_file = os.path.join(results_dir, 'qwen3_results.json')
    if os.path.exists(qwen3_results_file):
        with open(qwen3_results_file, 'r', encoding='utf-8') as f:
            qwen3_results = json.load(f)
            all_results['Qwen3 Zero-shot'] = {
                'Accuracy': qwen3_results['accuracy'],
                'F1 (weighted)': qwen3_results['f1_weighted'],
                'F1 (macro)': qwen3_results['f1_macro']
            }
    
    if not all_results:
        print("❌ 未找到任何结果文件")
        return False
    
    # 创建比较表格
    comparison_df = pd.DataFrame(all_results).T
    comparison_df = comparison_df.round(4)
    
    print("\n=== 所有模型结果比较 ===")
    print(comparison_df.to_string())
    
    # 找出最佳模型（基于不同指标）
    best_models = {}
    
    # 检查有哪些指标可用
    available_metrics = comparison_df.columns.tolist()
    
    if 'F1 (Micro)' in available_metrics:
        best_f1_micro = comparison_df['F1 (Micro)'].idxmax()
        best_models['f1_micro'] = {'model': best_f1_micro, 'score': float(comparison_df.loc[best_f1_micro, 'F1 (Micro)'])}
    
    if 'F1 (Macro)' in available_metrics:
        best_f1_macro = comparison_df['F1 (Macro)'].idxmax()
        best_models['f1_macro'] = {'model': best_f1_macro, 'score': float(comparison_df.loc[best_f1_macro, 'F1 (Macro)'])}
        
    if 'F1 (Weighted)' in available_metrics:
        best_f1_weighted = comparison_df['F1 (Weighted)'].idxmax()
        best_models['f1_weighted'] = {'model': best_f1_weighted, 'score': float(comparison_df.loc[best_f1_weighted, 'F1 (Weighted)'])}
    
    if 'Jaccard (Micro)' in available_metrics:
        best_jaccard = comparison_df['Jaccard (Micro)'].idxmax()
        best_models['jaccard_micro'] = {'model': best_jaccard, 'score': float(comparison_df.loc[best_jaccard, 'Jaccard (Micro)'])}
    
    if 'Hamming Loss' in available_metrics:
        best_hamming = comparison_df['Hamming Loss'].idxmin()  # 越小越好
        best_models['hamming_loss'] = {'model': best_hamming, 'score': float(comparison_df.loc[best_hamming, 'Hamming Loss'])}
    
    # 兼容单标签指标
    if 'Accuracy' in available_metrics:
        best_accuracy = comparison_df['Accuracy'].idxmax()
        best_models['accuracy'] = {'model': best_accuracy, 'score': float(comparison_df.loc[best_accuracy, 'Accuracy'])}
    
    print(f"\n=== 最佳模型 ===")
    for metric, info in best_models.items():
        print(f"{metric}: {info['model']} ({info['score']:.4f})")
    
    # 保存总结果
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'model_comparison': comparison_df.to_dict(),
        'best_models': best_models
    }
    
    with open(os.path.join(results_dir, 'final_comparison.json'), 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    comparison_df.to_csv(os.path.join(results_dir, 'all_models_comparison.csv'))
    
    print(f"\n✓ 最终比较结果已保存到 {results_dir}/")
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行精神疾病诊断预测baseline')
    parser.add_argument('--skip-preprocessing', action='store_true', 
                       help='跳过数据预处理步骤')
    parser.add_argument('--only-tfidf', action='store_true',
                       help='只运行TF-IDF模型')
    parser.add_argument('--only-bert', action='store_true',
                       help='只运行ChineseBERT模型')
    parser.add_argument('--only-qwen3', action='store_true',
                       help='只运行Qwen3模型')
    parser.add_argument('--compare-only', action='store_true',
                       help='只进行结果比较，不运行模型')
    
    args = parser.parse_args()
    
    print("精神疾病辅助诊断结果预测 - Baseline运行")
    print("="*60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success_count = 0
    total_steps = 0
    
    # 只比较结果
    if args.compare_only:
        collect_and_compare_results()
        return
    
    # 数据预处理
    if not args.skip_preprocessing:
        total_steps += 1
        if run_data_preprocessing():
            success_count += 1
    
    # 检查数据是否存在
    if not os.path.exists('processed_data/train.csv'):
        print("❌ 未找到预处理后的数据，请先运行数据预处理")
        return
    
    # 运行指定的模型
    if args.only_tfidf:
        total_steps += 1
        if run_tfidf_baseline():
            success_count += 1
    elif args.only_bert:
        total_steps += 1
        if run_chinesebert_baseline():
            success_count += 1
    elif args.only_qwen3:
        total_steps += 1
        if run_qwen3_baseline():
            success_count += 1
    else:
        # 运行所有模型
        total_steps += 3
        
        if run_tfidf_baseline():
            success_count += 1
        
        if run_chinesebert_baseline():
            success_count += 1
        
        if run_qwen3_baseline():
            success_count += 1
    
    # 结果比较
    total_steps += 1
    if collect_and_compare_results():
        success_count += 1
    
    # 总结
    print("\n" + "="*60)
    print("运行总结")
    print("="*60)
    print(f"完成步骤: {success_count}/{total_steps}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success_count == total_steps:
        print("🎉 所有步骤成功完成!")
    else:
        print("⚠️  部分步骤失败，请检查错误信息")

if __name__ == "__main__":
    main() 