#!/usr/bin/env python3
"""
Test Single Experiment
測試單個實驗，確保基本功能正常
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

def test_qasa_experiment():
    """測試QASA模型實驗"""
    print("🧪 測試QASA模型實驗")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  使用設備: {device}")
    
    # 創建數據加載器
    print("📊 創建數據加載器...")
    train_loader, tokenizer = create_dataloader(
        dataset_name='simple_sentences',
        split='train',
        batch_size=8,
        max_length=32,
        vocab_size=500,
        max_samples=100,
        shuffle=True
    )
    
    val_loader, _ = create_dataloader(
        dataset_name='simple_sentences',
        split='val',
        batch_size=8,
        max_length=32,
        vocab_size=500,
        max_samples=20,
        shuffle=False
    )
    
    print(f"✅ 數據加載器創建成功")
    print(f"📝 詞彙大小: {tokenizer.get_vocab_size()}")
    print(f"📦 訓練批次數: {len(train_loader)}")
    print(f"📦 驗證批次數: {len(val_loader)}")
    
    # 創建模型
    print("\n🤖 創建QASA模型...")
    vocab_size = tokenizer.get_vocab_size()
    model = QASATextGenerator(
        vocab_size=vocab_size,
        hidden_dim=128,
        seq_len=32,
        n_qubits=8,
        device=device
    )
    model.to(device)
    
    print(f"✅ 模型創建成功")
    print(f"📊 模型參數數量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 測試前向傳播
    print("\n🔄 測試前向傳播...")
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            outputs = model(input_ids)
            print(f"✅ 前向傳播成功")
            print(f"📊 輸入形狀: {input_ids.shape}")
            print(f"📊 輸出形狀: {outputs.shape}")
            print(f"📊 目標形狀: {target_ids.shape}")
            break
    
    # 訓練測試
    print("\n🎯 開始訓練測試 (5個epoch)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    start_time = time.time()
    
    for epoch in range(5):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            
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
                logits = outputs.view(-1, outputs.size(-1))
                targets = target_ids.view(-1)
                loss = criterion(logits, targets)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/5: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    training_time = time.time() - start_time
    print(f"✅ 訓練測試完成，耗時: {training_time:.2f}秒")
    
    # 生成文本測試
    print("\n📝 測試文本生成...")
    test_prompts = ["the", "birds", "water"]
    generated_texts = []
    
    model.eval()
    with torch.no_grad():
        for prompt in test_prompts:
            try:
                prompt_tokens = tokenizer.encode(prompt)
                input_ids = torch.tensor([prompt_tokens], device=device)
                
                if hasattr(model, 'generate'):
                    generated = model.generate(
                        input_ids=input_ids,
                        max_length=20,
                        temperature=1.0,
                        do_sample=True,
                        top_k=50
                    )
                else:
                    generated = input_ids
                
                generated_text = tokenizer.decode(generated[0].tolist())
                generated_texts.append(generated_text)
                print(f"Prompt: '{prompt}' -> Generated: '{generated_text}'")
            except Exception as e:
                print(f"生成失敗: {e}")
                generated_texts.append(f"[生成失敗: {prompt}]")
    
    # 保存結果
    print("\n💾 保存實驗結果...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"./experiments/test_results/qasa_simple_sentences_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'model_type': 'qasa',
            'dataset_name': 'simple_sentences',
            'vocab_size': vocab_size,
            'epochs': 5
        }
    }, os.path.join(result_dir, 'model.pth'))
    
    # 保存生成文本
    with open(os.path.join(result_dir, 'generated_texts.txt'), 'w') as f:
        f.write("QASA Model Test Results\n")
        f.write("=" * 30 + "\n\n")
        for i, (prompt, text) in enumerate(zip(test_prompts, generated_texts)):
            f.write(f"Prompt {i+1}: '{prompt}'\n")
            f.write(f"Generated: '{text}'\n\n")
    
    # 保存實驗信息
    experiment_info = {
        'model_type': 'qasa',
        'dataset_name': 'simple_sentences',
        'training_time': training_time,
        'vocab_size': vocab_size,
        'generated_texts': generated_texts,
        'result_dir': result_dir
    }
    
    with open(os.path.join(result_dir, 'experiment_info.json'), 'w') as f:
        json.dump(experiment_info, f, indent=2)
    
    print(f"✅ 實驗結果已保存到: {result_dir}")
    print("\n🎉 QASA模型實驗測試完成!")
    
    return experiment_info

def main():
    """主函數"""
    print("🚀 量子文字生成單個實驗測試")
    print("=" * 60)
    
    try:
        result = test_qasa_experiment()
        print(f"\n{'='*60}")
        print("✅ 所有測試通過!")
        print(f"📁 結果保存在: {result['result_dir']}")
        print(f"{'='*60}")
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
