#!/usr/bin/env python3
"""
Experiment Monitor
監控訓練實驗進度
"""

import os
import time
import json
from datetime import datetime

def check_experiment_progress():
    """檢查實驗進度"""
    base_dir = "./experiments/comprehensive_training"
    
    if not os.path.exists(base_dir):
        print("❌ 實驗目錄不存在")
        return
    
    print("🔍 檢查實驗進度...")
    print("=" * 60)
    
    # 檢查各模型目錄
    models = ['qasa', 'qksan', 'qrwkv']
    datasets = ['simple_sentences', 'short_stories', 'quantum_phrases', 'haiku', 'proverbs']
    
    completed_experiments = []
    running_experiments = []
    
    for model in models:
        model_dir = os.path.join(base_dir, model)
        if not os.path.exists(model_dir):
            print(f"📁 {model} 模型目錄不存在")
            continue
        
        print(f"\n🤖 {model.upper()} 模型:")
        for dataset in datasets:
            # 查找該數據集的實驗目錄
            dataset_experiments = []
            for item in os.listdir(model_dir):
                if item.startswith(dataset):
                    dataset_experiments.append(item)
            
            if dataset_experiments:
                # 取最新的實驗
                latest_experiment = sorted(dataset_experiments)[-1]
                experiment_path = os.path.join(model_dir, latest_experiment)
                
                # 檢查是否完成
                model_file = os.path.join(experiment_path, 'model.pth')
                history_file = os.path.join(experiment_path, 'training_history.json')
                texts_file = os.path.join(experiment_path, 'generated_texts.txt')
                
                if all(os.path.exists(f) for f in [model_file, history_file, texts_file]):
                    # 讀取訓練結果
                    try:
                        with open(history_file, 'r') as f:
                            history = json.load(f)
                        
                        best_val_loss = history.get('best_val_loss', 'N/A')
                        final_train_loss = history.get('final_train_loss', 'N/A')
                        
                        completed_experiments.append({
                            'model': model,
                            'dataset': dataset,
                            'best_val_loss': best_val_loss,
                            'final_train_loss': final_train_loss,
                            'path': experiment_path
                        })
                        
                        print(f"  ✅ {dataset}: 完成 (最佳驗證損失: {best_val_loss:.4f})")
                    except:
                        print(f"  ⚠️  {dataset}: 文件存在但讀取失敗")
                else:
                    print(f"  🔄 {dataset}: 進行中...")
                    running_experiments.append(f"{model}_{dataset}")
            else:
                print(f"  ❌ {dataset}: 未開始")
    
    # 統計信息
    total_experiments = len(models) * len(datasets)
    completed_count = len(completed_experiments)
    running_count = len(running_experiments)
    
    print(f"\n{'='*60}")
    print("📊 實驗統計:")
    print(f"  總實驗數: {total_experiments}")
    print(f"  已完成: {completed_count}")
    print(f"  進行中: {running_count}")
    print(f"  未開始: {total_experiments - completed_count - running_count}")
    print(f"  完成率: {completed_count/total_experiments*100:.1f}%")
    
    if completed_experiments:
        print(f"\n🏆 最佳性能:")
        best_experiment = min(completed_experiments, key=lambda x: x['best_val_loss'])
        print(f"  {best_experiment['model'].upper()} on {best_experiment['dataset']}: {best_experiment['best_val_loss']:.4f}")
    
    return completed_experiments, running_experiments

def show_recent_results():
    """顯示最近的實驗結果"""
    base_dir = "./experiments/comprehensive_training"
    
    if not os.path.exists(base_dir):
        return
    
    print("\n📋 最近的實驗結果:")
    print("=" * 60)
    
    # 查找最近的比較報告
    report_dirs = []
    for item in os.listdir(base_dir):
        if item.startswith('comparison_report_'):
            report_dirs.append(item)
    
    if report_dirs:
        latest_report = sorted(report_dirs)[-1]
        report_path = os.path.join(base_dir, latest_report, 'comparison_report.md')
        
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                content = f.read()
            
            # 提取關鍵信息
            lines = content.split('\n')
            for line in lines:
                if '實驗時間:' in line or '訓練輪數:' in line or '數據集數量:' in line:
                    print(line.strip())
                elif '平均最佳驗證損失=' in line:
                    print(line.strip())
    
    # 顯示最近的生成文本
    print("\n📝 最近的生成文本示例:")
    print("-" * 40)
    
    models = ['qasa', 'qksan', 'qrwkv']
    for model in models:
        model_dir = os.path.join(base_dir, model)
        if not os.path.exists(model_dir):
            continue
        
        # 查找最新的實驗
        experiments = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
        if experiments:
            latest_exp = sorted(experiments)[-1]
            texts_file = os.path.join(model_dir, latest_exp, 'generated_texts.txt')
            
            if os.path.exists(texts_file):
                print(f"\n🤖 {model.upper()} 模型:")
                with open(texts_file, 'r') as f:
                    content = f.read()
                    lines = content.split('\n')
                    for line in lines[:10]:  # 只顯示前10行
                        if line.strip() and not line.startswith('='):
                            print(f"  {line}")

def main():
    """主函數"""
    print("🔍 量子文字生成實驗監控器")
    print("=" * 60)
    
    while True:
        try:
            completed, running = check_experiment_progress()
            show_recent_results()
            
            print(f"\n⏰ 檢查時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("按 Ctrl+C 退出監控")
            
            # 等待30秒後再次檢查
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("\n👋 監控結束")
            break
        except Exception as e:
            print(f"❌ 監控錯誤: {e}")
            time.sleep(10)

if __name__ == '__main__':
    main()
