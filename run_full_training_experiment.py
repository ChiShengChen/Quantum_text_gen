#!/usr/bin/env python3
"""
Full Training Experiment
完整的訓練實驗，在所有數據集上訓練所有模型30個epoch
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
    """創建模型"""
    if model_type == 'qasa':
        return QASATextGenerator(
            vocab_size=vocab_size,
            hidden_dim=kwargs.get('hidden_dim', 128),
            seq_len=kwargs.get('seq_len', 32),
            n_qubits=kwargs.get('n_qubits', 8),
            device=device
        )
    elif model_type == 'qksan':
        return create_fixed_qksan_model(
            vocab_size=vocab_size,
            embedding_dim=kwargs.get('embedding_dim', 128),
            n_qubits=kwargs.get('n_qubits', 6),
            device=device
        )
    elif model_type == 'qrwkv':
        config = ModelConfig(
            vocab_size=vocab_size,
            n_embd=kwargs.get('embedding_dim', 128),
            n_layer=kwargs.get('n_layers', 4),
            n_qubits=kwargs.get('n_qubits', 4),
            block_size=512
        )
        return QuantumRWKVModel(config)
    else:
        raise ValueError(f"不支持的模型類型: {model_type}")

def train_model(model, train_loader, val_loader, device, epochs=30):
    """訓練模型"""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"開始訓練，共{epochs}個epoch...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
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
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
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
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'final_train_loss': train_losses[-1]
    }

def generate_texts(model, tokenizer, prompts, device):
    """生成文本"""
    model.eval()
    generated_texts = []
    
    with torch.no_grad():
        for prompt in prompts:
            try:
                prompt_tokens = tokenizer.encode(prompt)
                input_ids = torch.tensor([prompt_tokens], device=device)
                
                # 簡單的貪心解碼
                generated = input_ids.clone()
                for _ in range(10):  # 生成10個token
                    outputs = model(generated)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    # 取最後一個位置的logits
                    next_token_logits = outputs[0, -1, :]
                    
                    # 貪心選擇
                    next_token = torch.argmax(next_token_logits).unsqueeze(0)
                    generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
                generated_text = tokenizer.decode(generated[0].tolist())
                generated_texts.append(generated_text)
            except Exception as e:
                print(f"生成失敗: {e}")
                generated_texts.append(f"[生成失敗: {prompt}]")
    
    return generated_texts

def run_experiment(model_type, dataset_name, device, epochs=30):
    """運行單個實驗"""
    print(f"\n{'='*60}")
    print(f"開始實驗: {model_type} on {dataset_name}")
    print(f"{'='*60}")
    
    # 創建數據加載器
    train_loader, tokenizer = create_dataloader(
        dataset_name=dataset_name,
        split='train',
        batch_size=16,
        max_length=32,
        vocab_size=500,
        max_samples=500,
        shuffle=True
    )
    
    val_loader, _ = create_dataloader(
        dataset_name=dataset_name,
        split='val',
        batch_size=16,
        max_length=32,
        vocab_size=500,
        max_samples=100,
        shuffle=False
    )
    
    # 創建模型
    vocab_size = tokenizer.get_vocab_size()
    model = create_model(model_type, vocab_size, device)
    
    # 訓練模型
    start_time = time.time()
    training_results = train_model(model, train_loader, val_loader, device, epochs)
    training_time = time.time() - start_time
    
    # 生成文本
    test_prompts = ["the", "birds", "water", "spring", "stars"]
    generated_texts = generate_texts(model, tokenizer, test_prompts, device)
    
    # 保存結果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"./experiments/comprehensive_training/{model_type}/{dataset_name}_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_results': training_results,
        'config': {
            'model_type': model_type,
            'dataset_name': dataset_name,
            'vocab_size': vocab_size,
            'epochs': epochs
        }
    }, os.path.join(model_dir, 'model.pth'))
    
    # 保存生成文本
    with open(os.path.join(model_dir, 'generated_texts.txt'), 'w') as f:
        f.write(f"Generated Texts for {model_type} on {dataset_name}\n")
        f.write("=" * 50 + "\n\n")
        for i, (prompt, text) in enumerate(zip(test_prompts, generated_texts)):
            f.write(f"Prompt {i+1}: '{prompt}'\n")
            f.write(f"Generated: '{text}'\n\n")
    
    # 保存訓練歷史
    with open(os.path.join(model_dir, 'training_history.json'), 'w') as f:
        json.dump(training_results, f, indent=2)
    
    result = {
        'model_type': model_type,
        'dataset_name': dataset_name,
        'training_results': training_results,
        'generated_texts': generated_texts,
        'training_time': training_time,
        'vocab_size': vocab_size,
        'model_dir': model_dir
    }
    
    print(f"✅ 實驗完成: {model_type} on {dataset_name}")
    print(f"⏱️  訓練時間: {training_time:.2f}秒")
    print(f"📊 最佳驗證損失: {training_results['best_val_loss']:.4f}")
    print(f"📁 結果保存到: {model_dir}")
    
    return result

def main():
    """主函數"""
    print("🚀 量子文字生成全面訓練實驗")
    print("=" * 60)
    
    # 設置設備
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  使用設備: {device}")
    
    # 實驗配置
    datasets = ['simple_sentences', 'short_stories', 'quantum_phrases', 'haiku', 'proverbs']
    models = ['qasa', 'qksan', 'qrwkv']
    epochs = 30
    
    print(f"📊 數據集: {datasets}")
    print(f"🤖 模型: {models}")
    print(f"🔄 訓練輪數: {epochs} epochs")
    
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
                continue
    
    # 生成比較報告
    print("\n📊 生成比較報告...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = f"./experiments/comprehensive_training/comparison_report_{timestamp}"
    os.makedirs(report_dir, exist_ok=True)
    
    # 保存所有結果
    with open(os.path.join(report_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # 生成報告
    report = f"""
# 量子文字生成模型全面比較報告

## 實驗概覽
- **實驗時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **訓練輪數**: {epochs} epochs
- **數據集數量**: {len(datasets)}
- **模型數量**: {len(models)}個

## 實驗結果

### 按模型類型比較
"""
    
    for model in models:
        model_data = [v for k, v in all_results.items() if v['model_type'] == model]
        if model_data:
            report += f"\n#### {model.upper()} 模型\n"
            report += "| 數據集 | 最佳驗證損失 | 最終訓練損失 | 訓練時間(秒) |\n"
            report += "|--------|-------------|-------------|------------|\n"
            
            for data in model_data:
                report += f"| {data['dataset_name']} | {data['training_results']['best_val_loss']:.4f} | {data['training_results']['final_train_loss']:.4f} | {data['training_time']:.1f} |\n"
    
    # 按數據集比較
    report += "\n### 按數據集比較\n"
    for dataset in datasets:
        dataset_data = [v for k, v in all_results.items() if v['dataset_name'] == dataset]
        if dataset_data:
            report += f"\n#### {dataset}\n"
            report += "| 模型 | 最佳驗證損失 | 最終訓練損失 | 訓練時間(秒) |\n"
            report += "|------|-------------|-------------|------------|\n"
            
            for data in dataset_data:
                report += f"| {data['model_type']} | {data['training_results']['best_val_loss']:.4f} | {data['training_results']['final_train_loss']:.4f} | {data['training_time']:.1f} |\n"
    
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
    
    # 總結
    report += "\n## 實驗總結\n\n"
    
    for model in models:
        model_data = [v for k, v in all_results.items() if v['model_type'] == model]
        if model_data:
            avg_val_loss = sum(d['training_results']['best_val_loss'] for d in model_data) / len(model_data)
            avg_train_loss = sum(d['training_results']['final_train_loss'] for d in model_data) / len(model_data)
            avg_time = sum(d['training_time'] for d in model_data) / len(model_data)
            
            report += f"- **{model.upper()}**: 平均最佳驗證損失={avg_val_loss:.4f}, 平均最終訓練損失={avg_train_loss:.4f}, 平均訓練時間={avg_time:.1f}秒\n"
    
    with open(os.path.join(report_dir, 'comparison_report.md'), 'w') as f:
        f.write(report)
    
    print(f"📄 報告已保存到: {report_dir}")
    print(f"\n{'='*60}")
    print("🎉 全面訓練實驗完成!")
    print(f"📊 共完成了 {len(all_results)} 個實驗")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
