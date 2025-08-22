#!/usr/bin/env python3
"""
Simplified Full Experiment
簡化的完整實驗，避免評估指標問題
"""

import os
import sys
import torch
import json
import time
from datetime import datetime

sys.path.append('.')

from data.datasets import create_dataloader
from models.quantum_text_models import QASATextGenerator
from qksan_fixed import create_fixed_qksan_model
from qrwkv_model import QuantumRWKVModel, ModelConfig

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
            
            # 檢查詞彙大小是否匹配
            vocab_size = model.embedding.num_embeddings if hasattr(model, 'embedding') else model.config.vocab_size
            max_valid_idx = vocab_size - 1
            
            # 過濾超出詞彙範圍的索引
            valid_mask = (input_ids <= max_valid_idx) & (target_ids <= max_valid_idx)
            if not valid_mask.any():
                continue  # 跳過這個batch
            
            # 只處理有效的token
            valid_input_ids = input_ids[valid_mask]
            valid_target_ids = target_ids[valid_mask]
            
            if len(valid_input_ids) == 0:
                continue
            
            try:
                outputs = model(valid_input_ids.unsqueeze(0) if valid_input_ids.dim() == 1 else valid_input_ids)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                logits = outputs.view(-1, outputs.size(-1))
                targets = valid_target_ids.view(-1)
                
                loss = torch.nn.functional.cross_entropy(logits, targets, reduction='sum')
                total_loss += loss.item()
                total_tokens += targets.numel()
            except Exception as e:
                print(f"計算困惑度時跳過batch: {e}")
                continue
    
    if total_tokens == 0:
        return float('inf')  # 如果沒有有效token，返回無窮大
    
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
                
                # 檢查詞彙大小是否匹配
                vocab_size = model.embedding.num_embeddings if hasattr(model, 'embedding') else model.config.vocab_size
                max_valid_idx = vocab_size - 1
                
                # 過濾超出詞彙範圍的token
                valid_tokens = [t for t in prompt_tokens if t <= max_valid_idx]
                if not valid_tokens:
                    valid_tokens = [0]  # 使用PAD token作為默認
                
                input_ids = torch.tensor([valid_tokens], device=device)
                
                # 貪心解碼
                generated = input_ids.clone()
                for step in range(max_length):
                    try:
                        outputs = model(generated)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        
                        next_token_logits = outputs[0, -1, :]
                        next_token = torch.argmax(next_token_logits).unsqueeze(0)
                        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                    except Exception as e:
                        print(f"生成步驟失敗: {e}")
                        break
                
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

def evaluate_model_simple(model, tokenizer, test_loader, device):
    """簡單評估模型"""
    print("📊 開始簡單評估...")
    
    # 計算困惑度
    perplexity = calculate_perplexity(model, test_loader, device)
    print(f"  📈 困惑度: {perplexity:.4f}")
    
    # 生成文本
    test_prompts = ["the", "birds", "water", "spring", "stars"]
    generated_texts = generate_texts_for_evaluation(model, tokenizer, test_prompts, device)
    
    print(f"  📝 生成文本示例:")
    for i, text in enumerate(generated_texts):
        print(f"    {i+1}. {text}")
    
    return {
        'perplexity': perplexity,
        'generated_texts': generated_texts
    }

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
    
    val_loader, val_tokenizer = create_dataloader(
        dataset_name=dataset_name,
        split='val',
        batch_size=16,
        max_length=32,
        vocab_size=500,
        max_samples=50,
        shuffle=False
    )
    
    test_loader, test_tokenizer = create_dataloader(
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
    
    # 簡單評估 - 使用訓練時的tokenizer
    evaluation_results = evaluate_model_simple(model, tokenizer, test_loader, device)
    
    # 保存結果
    print(f"\n💾 保存實驗結果...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"./experiments/simplified_full/{model_type}/{dataset_name}_{timestamp}"
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
        test_prompts = ["the", "birds", "water", "spring", "stars"]
        for i, (prompt, text) in enumerate(zip(test_prompts, evaluation_results['generated_texts'])):
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
    print("🚀 量子文字生成簡化完整實驗")
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
    
    # 生成簡單比較報告
    print("\n📊 生成簡單比較報告...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = f"./experiments/simplified_full/simple_report_{timestamp}"
    os.makedirs(report_dir, exist_ok=True)
    
    # 保存所有結果
    with open(os.path.join(report_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # 生成簡單報告
    report = f"""
# 量子文字生成模型簡單比較報告

## 實驗概覽
- **實驗時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **訓練輪數**: {epochs} epochs
- **數據集數量**: {len(datasets)}
- **模型數量**: {len(models)}個
- **配置**: 簡化的小型配置（64維隱藏層，較少量子比特）

## 實驗結果

### 按模型類型比較
"""
    
    for model in models:
        model_data = [v for k, v in all_results.items() if v['model_type'] == model]
        if model_data:
            report += f"\n#### {model.upper()} 模型\n"
            report += "| 數據集 | 最佳驗證損失 | 困惑度 | 訓練時間(秒) |\n"
            report += "|--------|-------------|--------|------------|\n"
            
            for data in model_data:
                report += f"| {data['dataset_name']} | {data['training_results']['best_val_loss']:.4f} | {data['evaluation_results']['perplexity']:.4f} | {data['training_time']:.1f} |\n"
    
    # 按數據集比較
    report += "\n### 按數據集比較\n"
    for dataset in datasets:
        dataset_data = [v for k, v in all_results.items() if v['dataset_name'] == dataset]
        if dataset_data:
            report += f"\n#### {dataset}\n"
            report += "| 模型 | 最佳驗證損失 | 困惑度 | 訓練時間(秒) |\n"
            report += "|------|-------------|--------|------------|\n"
            
            for data in dataset_data:
                report += f"| {data['model_type']} | {data['training_results']['best_val_loss']:.4f} | {data['evaluation_results']['perplexity']:.4f} | {data['training_time']:.1f} |\n"
    
    # 生成文本示例
    report += "\n### 生成文本示例\n"
    for model in models:
        model_data = [v for k, v in all_results.items() if v['model_type'] == model]
        if model_data:
            report += f"\n#### {model.upper()} 生成示例\n"
            for data in model_data[:2]:  # 只顯示前2個數據集
                report += f"**{data['dataset_name']}**:\n"
                for i, text in enumerate(data['evaluation_results']['generated_texts'][:3]):  # 只顯示前3個生成文本
                    report += f"- {text}\n"
                report += "\n"
    
    # 性能總結
    report += "\n## 性能總結\n\n"
    
    for model in models:
        model_data = [v for k, v in all_results.items() if v['model_type'] == model]
        if model_data:
            avg_val_loss = sum(d['training_results']['best_val_loss'] for d in model_data) / len(model_data)
            avg_perplexity = sum(d['evaluation_results']['perplexity'] for d in model_data) / len(model_data)
            avg_time = sum(d['training_time'] for d in model_data) / len(model_data)
            
            report += f"- **{model.upper()}**: 平均最佳驗證損失={avg_val_loss:.4f}, 平均困惑度={avg_perplexity:.4f}, 平均訓練時間={avg_time:.1f}秒\n"
    
    with open(os.path.join(report_dir, 'simple_report.md'), 'w') as f:
        f.write(report)
    
    print(f"📄 報告已保存到: {report_dir}")
    print(f"\n{'='*80}")
    print("🎉 簡化完整實驗完成!")
    print(f"📊 共完成了 {len(all_results)} 個實驗")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
