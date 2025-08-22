#!/usr/bin/env python3
import os
import sys
import json
import torch
import time
from datetime import datetime

# 添加當前目錄到Python路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

def run_experiment(model_type, dataset_name, config, device):
    """運行單個實驗"""
    print(f"\n{'='*80}")
    print(f"🎯 開始實驗: {model_type.upper()} on {dataset_name}")
    print(f"{'='*80}")
    
    try:
        # 導入必要的模塊
        from data.datasets import create_dataloader
        
        if model_type == 'qksan':
            from qksan_fixed import create_fixed_qksan_model
        elif model_type == 'qrwkv':
            from qrwkv_model import QuantumRWKVModel, ModelConfig
        elif model_type == 'qasa':
            from models.quantum_text_models import QASATextGenerator
        
        # 創建數據加載器
        print("📊 創建數據加載器...")
        train_loader, tokenizer = create_dataloader(
            dataset_name=dataset_name,
            split='train',
            batch_size=config['batch_size'],
            max_length=config['max_length'],
            vocab_size=config['vocab_size'],
            max_samples=config['max_samples'],
            shuffle=True
        )
        
        val_loader, _ = create_dataloader(
            dataset_name=dataset_name,
            split='val',
            batch_size=config['batch_size'],
            max_length=config['max_length'],
            vocab_size=config['vocab_size'],
            max_samples=config['max_samples'] // 4,
            shuffle=False
        )
        
        test_loader, _ = create_dataloader(
            dataset_name=dataset_name,
            split='test',
            batch_size=config['batch_size'],
            max_length=config['max_length'],
            vocab_size=config['vocab_size'],
            max_samples=config['max_samples'] // 4,
            shuffle=False
        )
        
        print(f"✅ 數據加載器創建成功")
        print(f"📝 詞彙大小: {tokenizer.get_vocab_size()}")
        
        # 創建模型
        print(f"\n🤖 創建{model_type.upper()}模型...")
        vocab_size = tokenizer.get_vocab_size()
        
        if model_type == 'qksan':
            model = create_fixed_qksan_model(
                vocab_size=vocab_size,
                embedding_dim=64,
                n_qubits=4,
                device=device
            )
        elif model_type == 'qrwkv':
            model_config = ModelConfig(
                vocab_size=vocab_size,
                n_embd=64,
                n_head=8,
                n_layer=2,
                n_qubits=2,
                block_size=256
            )
            model = QuantumRWKVModel(model_config)
        elif model_type == 'qasa':
            model = QASATextGenerator(
                vocab_size=vocab_size,
                hidden_dim=64,
                seq_len=config['max_length'],
                n_qubits=4,
                device=device
            )
        
        print(f"✅ 模型創建成功")
        print(f"📊 模型參數數量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 訓練模型
        print(f"\n🎯 開始訓練...")
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        criterion = torch.nn.CrossEntropyLoss()
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(config['epochs']):
            # Training
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch in train_loader:
                try:
                    input_ids = batch['input_ids'].to(device)
                    target_ids = batch['target_ids'].to(device)
                    
                    # 檢查索引範圍
                    if input_ids.max() >= vocab_size or target_ids.max() >= vocab_size:
                        print(f"⚠️  跳過包含超出詞彙範圍的batch")
                        continue
                    
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
                    train_batches += 1
                except Exception as e:
                    print(f"⚠️  訓練batch失敗: {e}")
                    continue
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    try:
                        input_ids = batch['input_ids'].to(device)
                        target_ids = batch['target_ids'].to(device)
                        
                        # 檢查索引範圍
                        if input_ids.max() >= vocab_size or target_ids.max() >= vocab_size:
                            continue
                        
                        outputs = model(input_ids)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        
                        logits = outputs.view(-1, outputs.size(-1))
                        targets = target_ids.view(-1)
                        loss = criterion(logits, targets)
                        val_loss += loss.item()
                        val_batches += 1
                    except Exception as e:
                        print(f"⚠️  驗證batch失敗: {e}")
                        continue
            
            if train_batches > 0 and val_batches > 0:
                avg_train_loss = train_loss / train_batches
                avg_val_loss = val_loss / val_batches
                
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                
                print(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{config['epochs']} | 跳過 (無有效數據)")
        
        training_time = time.time() - start_time
        print(f"✅ 訓練完成! 最佳驗證損失: {best_val_loss:.4f}")
        print(f"⏱️  訓練時間: {training_time:.2f}秒")
        
        # 評估模型
        print(f"\n📊 開始評估...")
        
        # 計算困惑度
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in test_loader:
                try:
                    input_ids = batch['input_ids'].to(device)
                    target_ids = batch['target_ids'].to(device)
                    
                    # 檢查索引範圍
                    if input_ids.max() >= vocab_size or target_ids.max() >= vocab_size:
                        continue
                    
                    outputs = model(input_ids)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    logits = outputs.view(-1, outputs.size(-1))
                    targets = target_ids.view(-1)
                    
                    loss = torch.nn.functional.cross_entropy(logits, targets, reduction='sum')
                    total_loss += loss.item()
                    total_tokens += targets.numel()
                except Exception as e:
                    print(f"⚠️  評估batch失敗: {e}")
                    continue
        
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
        else:
            perplexity = float('inf')
        
        print(f"📈 困惑度: {perplexity:.4f}")
        
        # 生成文本示例
        test_prompts = ["the", "birds", "water", "spring", "stars"]
        generated_texts = []
        
        model.eval()
        with torch.no_grad():
            for prompt in test_prompts:
                try:
                    prompt_tokens = tokenizer.encode(prompt)
                    # 檢查prompt tokens是否在詞彙範圍內
                    if max(prompt_tokens) >= vocab_size:
                        generated_texts.append(f"[詞彙超出範圍: {prompt}]")
                        continue
                    
                    input_ids = torch.tensor([prompt_tokens], device=device)
                    
                    generated = input_ids.clone()
                    for step in range(20):
                        try:
                            outputs = model(generated)
                            if isinstance(outputs, tuple):
                                outputs = outputs[0]
                            
                            next_token_logits = outputs[0, -1, :]
                            next_token = torch.argmax(next_token_logits).unsqueeze(0)
                            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                        except Exception as e:
                            print(f"⚠️  生成步驟失敗: {e}")
                            break
                    
                    generated_text = tokenizer.decode(generated[0].tolist())
                    generated_texts.append(generated_text)
                except Exception as e:
                    print(f"⚠️  生成失敗: {e}")
                    generated_texts.append(f"[生成失敗: {prompt}]")
        
        print(f"📝 生成文本示例:")
        for i, text in enumerate(generated_texts):
            print(f"    {i+1}. {text}")
        
        # 保存結果
        result = {
            'model_type': model_type,
            'dataset_name': dataset_name,
            'training_results': {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
                'final_train_loss': train_losses[-1] if train_losses else float('inf'),
                'final_val_loss': val_losses[-1] if val_losses else float('inf'),
                'training_time': training_time
            },
            'evaluation_results': {
                'perplexity': perplexity,
                'generated_texts': generated_texts
            },
            'config': config,
            'vocab_size': vocab_size,
            'status': 'success'
        }
        
        return result
        
    except Exception as e:
        print(f"❌ 實驗失敗: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'model_type': model_type,
            'dataset_name': dataset_name,
            'status': 'failed',
            'error': str(e),
            'config': config
        }

def main():
    # 配置
    config = {
        'epochs': int(os.environ.get('EPOCHS', 8)),
        'batch_size': int(os.environ.get('BATCH_SIZE', 16)),
        'max_length': int(os.environ.get('MAX_LENGTH', 32)),
        'vocab_size': int(os.environ.get('VOCAB_SIZE', 500)),
        'max_samples': int(os.environ.get('MAX_SAMPLES', 200)),
        'learning_rate': float(os.environ.get('LEARNING_RATE', 0.001))
    }
    
    # 模型和數據集
    models = os.environ.get('MODELS', 'qksan,qrwkv,qasa').split(',')
    datasets = os.environ.get('DATASETS', 'simple_sentences,short_stories,quantum_phrases,haiku,proverbs').split(',')
    
    # 強制使用CPU以避免CUDA錯誤
    device = 'cpu'
    print(f"🖥️  使用設備: {device} (強制CPU模式)")
    
    print(f"🤖 模型: {models}")
    print(f"📊 數據集: {datasets}")
    print(f"🔄 訓練輪數: {config['epochs']} epochs")
    
    # 創建輸出目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # 運行實驗
    all_results = {}
    
    for model_type in models:
        for dataset_name in datasets:
            try:
                print(f"\n🚀 開始實驗: {model_type.upper()} on {dataset_name}")
                result = run_experiment(model_type, dataset_name, config, device)
                
                # 保存單個實驗結果
                experiment_key = f"{model_type}_{dataset_name}"
                all_results[experiment_key] = result
                
                # 保存到文件
                result_file = os.path.join(results_dir, f"{experiment_key}.json")
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                
                if result.get('status') == 'success':
                    print(f"✅ 實驗完成: {model_type.upper()} on {dataset_name}")
                else:
                    print(f"❌ 實驗失敗: {model_type.upper()} on {dataset_name}")
                
            except Exception as e:
                print(f"❌ 實驗失敗 {model_type} on {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 保存所有結果
    all_results_file = os.path.join(results_dir, "all_results.json")
    with open(all_results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # 生成簡單報告
    report_file = os.path.join(results_dir, "experiment_report.md")
    with open(report_file, 'w') as f:
        f.write(f"# 量子文字生成模型實驗報告 (最終修復版)\n\n")
        f.write(f"## 實驗配置\n")
        f.write(f"- 實驗時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- 模型: {models}\n")
        f.write(f"- 數據集: {datasets}\n")
        f.write(f"- 訓練輪數: {config['epochs']}\n")
        f.write(f"- 批次大小: {config['batch_size']}\n")
        f.write(f"- 學習率: {config['learning_rate']}\n")
        f.write(f"- 設備: CPU (強制模式)\n\n")
        
        f.write(f"## 實驗結果\n\n")
        f.write(f"| 模型 | 數據集 | 狀態 | 最佳驗證損失 | 困惑度 | 訓練時間(秒) |\n")
        f.write(f"|------|--------|------|-------------|--------|------------|\n")
        
        for key, result in all_results.items():
            model_type = result['model_type']
            dataset_name = result['dataset_name']
            status = result.get('status', 'unknown')
            
            if status == 'success':
                best_val_loss = result['training_results']['best_val_loss']
                perplexity = result['evaluation_results']['perplexity']
                training_time = result['training_results']['training_time']
                f.write(f"| {model_type} | {dataset_name} | ✅ 成功 | {best_val_loss:.4f} | {perplexity:.4f} | {training_time:.1f} |\n")
            else:
                error = result.get('error', '未知錯誤')
                f.write(f"| {model_type} | {dataset_name} | ❌ 失敗 | - | - | - |\n")
    
    print(f"\n🎉 所有實驗完成!")
    print(f"📁 結果保存到: {results_dir}")
    print(f"📄 報告文件: {report_file}")

if __name__ == '__main__':
    main()
