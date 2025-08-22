#!/usr/bin/env python3
"""
Optimized Full Experiment
優化的完整實驗，使用較小配置並包含完整評估指標
"""

import os
import sys
import torch
import json
import time
import numpy as np
from datetime import datetime

sys.path.append('.')

from data.datasets import create_dataloader
from models.quantum_text_models import QASATextGenerator
from qksan_fixed import create_fixed_qksan_model
from qrwkv_model import QuantumRWKVModel, ModelConfig
from evaluation.multi_metrics import MultiMetricEvaluator

def create_model(model_type, vocab_size, device, **kwargs):
    """創建模型 - 使用較小配置"""
    if model_type == 'qasa':
        return QASATextGenerator(
            vocab_size=vocab_size,
            hidden_dim=kwargs.get('hidden_dim', 64),  # 減少到64
            seq_len=kwargs.get('seq_len', 16),        # 減少到16
            n_qubits=kwargs.get('n_qubits', 4),       # 減少到4
            device=device
        )
    elif model_type == 'qksan':
        return create_fixed_qksan_model(
            vocab_size=vocab_size,
            embedding_dim=kwargs.get('embedding_dim', 64),  # 減少到64
            n_qubits=kwargs.get('n_qubits', 4),            # 減少到4
            device=device
        )
    elif model_type == 'qrwkv':
        config = ModelConfig(
            vocab_size=vocab_size,
            n_embd=kwargs.get('embedding_dim', 64),    # 減少到64
            n_head=8,                                   # 確保64能被8整除
            n_layer=kwargs.get('n_layers', 2),         # 減少到2層
            n_qubits=kwargs.get('n_qubits', 2),        # 減少到2
            block_size=256
        )
        return QuantumRWKVModel(config)
    else:
        raise ValueError(f"不支持的模型類型: {model_type}")

def calculate_perplexity(model, data_loader, device):
    """計算困惑度"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            outputs = model(input_ids)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            logits = outputs.view(-1, outputs.size(-1))
            targets = target_ids.view(-1)
            
            loss = torch.nn.functional.cross_entropy(logits, targets, reduction='sum')
            total_loss += loss.item()
            total_tokens += targets.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity

def generate_texts_for_evaluation(model, tokenizer, prompts, device, max_length=20):
    """生成文本用於評估"""
    model.eval()
    generated_texts = []
    
    with torch.no_grad():
        for prompt in prompts:
            try:
                prompt_tokens = tokenizer.encode(prompt)
                input_ids = torch.tensor([prompt_tokens], device=device)
                
                # 貪心解碼
                generated = input_ids.clone()
                for step in range(max_length):
                    outputs = model(generated)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    next_token_logits = outputs[0, -1, :]
                    next_token = torch.argmax(next_token_logits).unsqueeze(0)
                    generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
                generated_text = tokenizer.decode(generated[0].tolist())
                generated_texts.append(generated_text)
            except Exception as e:
                print(f"生成失敗: {e}")
                generated_texts.append(f"[生成失敗: {prompt}]")
    
    return generated_texts

def train_model(model, train_loader, val_loader, device, epochs=8):
    """訓練模型"""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    
    print(f"🚀 開始訓練，共{epochs}個epoch...")
    print("=" * 60)
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            logits = outputs.view(-1, outputs.size(-1))
            targets = target_ids.view(-1)
            loss = criterion(logits, targets)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                target_ids = batch['target_ids'].to(device)
                
                outputs = model(input_ids)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                logits = outputs.view(-1, outputs.size(-1))
                targets = target_ids.view(-1)
                loss = criterion(logits, targets)
                val_loss += loss.item()
                val_batches += 1
        
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        epoch_time = time.time() - epoch_start_time
        
        # 學習率調度
        scheduler.step(avg_val_loss)
        
        # 檢查是否是最佳模型
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
        
        # 顯示詳細的epoch信息
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Time: {epoch_time:.2f}s | "
              f"{'🏆 BEST' if is_best else '     '}")
        
        # 每4個epoch顯示更詳細的信息
        if (epoch + 1) % 4 == 0:
            print(f"  📊 訓練批次數: {train_batches}, 驗證批次數: {val_batches}")
            print(f"  📈 當前最佳驗證損失: {best_val_loss:.4f}")
            print(f"  📉 學習率: {optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 60)
    
    print("=" * 60)
    print(f"✅ 訓練完成! 最佳驗證損失: {best_val_loss:.4f}")
    
    # 恢復最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1]
    }

def evaluate_model_comprehensive(model, tokenizer, test_loader, device):
    """綜合評估模型"""
    print("📊 開始綜合評估...")
    
    # 計算困惑度
    perplexity = calculate_perplexity(model, test_loader, device)
    print(f"  📈 困惑度: {perplexity:.4f}")
    
    # 生成文本用於評估
    test_prompts = ["the", "birds", "water", "spring", "stars", "quantum", "light", "time"]
    generated_texts = generate_texts_for_evaluation(model, tokenizer, test_prompts, device)
    
    # 使用多指標評估器
    evaluator = MultiMetricEvaluator()
    
    # 準備參考文本（使用測試數據集的一些樣本）
    reference_texts = []
    with torch.no_grad():
        for batch in test_loader:
            if len(reference_texts) >= len(generated_texts):  # 確保參考文本數量與生成文本相同
                break
            input_ids = batch['input_ids']
            for i in range(min(2, input_ids.size(0))):  # 每個batch取2個樣本
                if len(reference_texts) >= len(generated_texts):  # 再次檢查
                    break
                text = tokenizer.decode(input_ids[i].tolist())
                reference_texts.append(text)
    
    # 確保參考文本和生成文本數量相同
    if len(reference_texts) != len(generated_texts):
        # 如果參考文本不夠，重複使用現有的
        while len(reference_texts) < len(generated_texts):
            reference_texts.extend(reference_texts[:len(generated_texts) - len(reference_texts)])
        # 如果參考文本太多，截斷
        reference_texts = reference_texts[:len(generated_texts)]
    
    print(f"  📝 生成文本數量: {len(generated_texts)}")
    print(f"  📝 參考文本數量: {len(reference_texts)}")
    
    # 評估生成文本
    evaluation_results = evaluator.evaluate_generated_texts(generated_texts, reference_texts)
    
    # 添加困惑度到結果中
    evaluation_results['perplexity'] = perplexity
    
    print(f"  📊 評估完成")
    for metric, value in evaluation_results.items():
        if isinstance(value, float):
            print(f"    {metric}: {value:.4f}")
        else:
            print(f"    {metric}: {value}")
    
    return evaluation_results, generated_texts

def run_experiment(model_type, dataset_name, device, epochs=8):
    """運行單個實驗"""
    print(f"\n{'='*80}")
    print(f"🎯 開始實驗: {model_type.upper()} on {dataset_name}")
    print(f"{'='*80}")
    
    # 創建數據加載器 - 使用較小配置
    print("📊 創建數據加載器...")
    train_loader, tokenizer = create_dataloader(
        dataset_name=dataset_name,
        split='train',
        batch_size=16,
        max_length=32,
        vocab_size=500,
        max_samples=200,  # 減少樣本數量
        shuffle=True
    )
    
    val_loader, _ = create_dataloader(
        dataset_name=dataset_name,
        split='val',
        batch_size=16,
        max_length=32,
        vocab_size=500,
        max_samples=50,
        shuffle=False
    )
    
    test_loader, _ = create_dataloader(
        dataset_name=dataset_name,
        split='test',
        batch_size=16,
        max_length=32,
        vocab_size=500,
        max_samples=50,
        shuffle=False
    )
    
    print(f"✅ 數據加載器創建成功")
    print(f"📝 詞彙大小: {tokenizer.get_vocab_size()}")
    print(f"📦 訓練批次數: {len(train_loader)}")
    print(f"📦 驗證批次數: {len(val_loader)}")
    print(f"📦 測試批次數: {len(test_loader)}")
    
    # 創建模型
    print(f"\n🤖 創建{model_type.upper()}模型...")
    vocab_size = tokenizer.get_vocab_size()
    model = create_model(model_type, vocab_size, device)
    
    print(f"✅ 模型創建成功")
    print(f"📊 模型參數數量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 訓練模型
    print(f"\n🎯 開始訓練...")
    start_time = time.time()
    training_results = train_model(model, train_loader, val_loader, device, epochs)
    training_time = time.time() - start_time
    
    # 綜合評估
    evaluation_results, generated_texts = evaluate_model_comprehensive(
        model, tokenizer, test_loader, device
    )
    
    # 保存結果
    print(f"\n💾 保存實驗結果...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"./experiments/optimized_full/{model_type}/{dataset_name}_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_results': training_results,
        'evaluation_results': evaluation_results,
        'config': {
            'model_type': model_type,
            'dataset_name': dataset_name,
            'vocab_size': vocab_size,
            'epochs': epochs
        }
    }, os.path.join(model_dir, 'model.pth'))
    
    # 保存生成文本
    with open(os.path.join(model_dir, 'generated_texts.txt'), 'w') as f:
        f.write(f"Generated Texts for {model_type.upper()} on {dataset_name}\n")
        f.write("=" * 50 + "\n\n")
        test_prompts = ["the", "birds", "water", "spring", "stars", "quantum", "light", "time"]
        for i, (prompt, text) in enumerate(zip(test_prompts, generated_texts)):
            f.write(f"Prompt {i+1}: '{prompt}'\n")
            f.write(f"Generated: '{text}'\n\n")
    
    # 保存訓練歷史
    with open(os.path.join(model_dir, 'training_history.json'), 'w') as f:
        json.dump(training_results, f, indent=2)
    
    # 保存評估結果
    with open(os.path.join(model_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    result = {
        'model_type': model_type,
        'dataset_name': dataset_name,
        'training_results': training_results,
        'evaluation_results': evaluation_results,
        'generated_texts': generated_texts,
        'training_time': training_time,
        'vocab_size': vocab_size,
        'model_dir': model_dir
    }
    
    print(f"✅ 實驗完成: {model_type.upper()} on {dataset_name}")
    print(f"⏱️  總訓練時間: {training_time:.2f}秒")
    print(f"📊 最佳驗證損失: {training_results['best_val_loss']:.4f}")
    print(f"📈 困惑度: {evaluation_results['perplexity']:.4f}")
    print(f"📁 結果保存到: {model_dir}")
    
    return result

def main():
    """主函數"""
    print("🚀 量子文字生成優化完整實驗")
    print("=" * 80)
    
    # 設置設備
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  使用設備: {device}")
    
    # 實驗配置 - 使用較小配置
    datasets = ['simple_sentences', 'short_stories', 'quantum_phrases', 'haiku', 'proverbs']
    models = ['qasa', 'qksan', 'qrwkv']
    epochs = 8  # 減少到8個epoch
    
    print(f"📊 數據集: {datasets}")
    print(f"🤖 模型: {models}")
    print(f"🔄 訓練輪數: {epochs} epochs")
    print(f"⚙️  配置: 較小模型配置（64維隱藏層，較少量子比特）")
    
    # 運行實驗
    all_results = {}
    
    for model_type in models:
        for dataset_name in datasets:
            try:
                result = run_experiment(model_type, dataset_name, device, epochs)
                experiment_key = f"{model_type}_{dataset_name}"
                all_results[experiment_key] = result
            except Exception as e:
                print(f"❌ 實驗失敗 {model_type} on {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 生成綜合比較報告
    print("\n📊 生成綜合比較報告...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = f"./experiments/optimized_full/comprehensive_report_{timestamp}"
    os.makedirs(report_dir, exist_ok=True)
    
    # 保存所有結果
    with open(os.path.join(report_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # 生成詳細報告
    report = generate_comprehensive_report(all_results, datasets, models, epochs)
    
    with open(os.path.join(report_dir, 'comprehensive_report.md'), 'w') as f:
        f.write(report)
    
    print(f"📄 報告已保存到: {report_dir}")
    print(f"\n{'='*80}")
    print("🎉 優化完整實驗完成!")
    print(f"📊 共完成了 {len(all_results)} 個實驗")
    print(f"{'='*80}")

def generate_comprehensive_report(all_results, datasets, models, epochs):
    """生成綜合報告"""
    report = f"""
# 量子文字生成模型綜合比較報告

## 實驗概覽
- **實驗時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **訓練輪數**: {epochs} epochs
- **數據集數量**: {len(datasets)}
- **模型數量**: {len(models)}個
- **配置**: 優化的小型配置（64維隱藏層，較少量子比特）

## 實驗結果

### 按模型類型比較
"""
    
    for model in models:
        model_data = [v for k, v in all_results.items() if v['model_type'] == model]
        if model_data:
            report += f"\n#### {model.upper()} 模型\n"
            report += "| 數據集 | 最佳驗證損失 | 困惑度 | BLEU | ROUGE-L | 多樣性 | 訓練時間(秒) |\n"
            report += "|--------|-------------|--------|------|---------|--------|------------|\n"
            
            for data in model_data:
                eval_results = data['evaluation_results']
                report += f"| {data['dataset_name']} | {data['training_results']['best_val_loss']:.4f} | {eval_results.get('perplexity', 'N/A'):.4f} | {eval_results.get('bleu_score', 'N/A'):.4f} | {eval_results.get('rouge_l', 'N/A'):.4f} | {eval_results.get('distinct_1', 'N/A'):.4f} | {data['training_time']:.1f} |\n"
    
    # 按數據集比較
    report += "\n### 按數據集比較\n"
    for dataset in datasets:
        dataset_data = [v for k, v in all_results.items() if v['dataset_name'] == dataset]
        if dataset_data:
            report += f"\n#### {dataset}\n"
            report += "| 模型 | 最佳驗證損失 | 困惑度 | BLEU | ROUGE-L | 多樣性 | 訓練時間(秒) |\n"
            report += "|------|-------------|--------|------|---------|--------|------------|\n"
            
            for data in dataset_data:
                eval_results = data['evaluation_results']
                report += f"| {data['model_type']} | {data['training_results']['best_val_loss']:.4f} | {eval_results.get('perplexity', 'N/A'):.4f} | {eval_results.get('bleu_score', 'N/A'):.4f} | {eval_results.get('rouge_l', 'N/A'):.4f} | {eval_results.get('distinct_1', 'N/A'):.4f} | {data['training_time']:.1f} |\n"
    
    # 生成文本示例
    report += "\n### 生成文本示例\n"
    for model in models:
        model_data = [v for k, v in all_results.items() if v['model_type'] == model]
        if model_data:
            report += f"\n#### {model.upper()} 生成示例\n"
            for data in model_data[:2]:  # 只顯示前2個數據集
                report += f"**{data['dataset_name']}**:\n"
                for i, text in enumerate(data['generated_texts'][:3]):  # 只顯示前3個生成文本
                    report += f"- {text}\n"
                report += "\n"
    
    # 性能總結
    report += "\n## 性能總結\n\n"
    
    for model in models:
        model_data = [v for k, v in all_results.items() if v['model_type'] == model]
        if model_data:
            avg_val_loss = sum(d['training_results']['best_val_loss'] for d in model_data) / len(model_data)
            avg_perplexity = sum(d['evaluation_results'].get('perplexity', 0) for d in model_data) / len(model_data)
            avg_bleu = sum(d['evaluation_results'].get('bleu_score', 0) for d in model_data) / len(model_data)
            avg_time = sum(d['training_time'] for d in model_data) / len(model_data)
            
            report += f"- **{model.upper()}**: 平均最佳驗證損失={avg_val_loss:.4f}, 平均困惑度={avg_perplexity:.4f}, 平均BLEU={avg_bleu:.4f}, 平均訓練時間={avg_time:.1f}秒\n"
    
    return report

if __name__ == '__main__':
    main()
