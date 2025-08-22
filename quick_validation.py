#!/usr/bin/env python3
"""
Quick Validation
快速驗證實驗流程
"""

import os
import sys
import torch
import json
import time
from datetime import datetime

sys.path.append('.')

from data.datasets import create_dataloader
from qrwkv_model import QuantumRWKVModel, ModelConfig

def quick_test():
    """快速測試"""
    print("🚀 快速驗證實驗")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  使用設備: {device}")
    
    # 只測試QRWKV模型和simple_sentences數據集
    dataset_name = 'simple_sentences'
    model_type = 'qrwkv'
    epochs = 3
    
    print(f"📊 數據集: {dataset_name}")
    print(f"🤖 模型: {model_type}")
    print(f"🔄 訓練輪數: {epochs} epochs")
    
    try:
        # 創建數據加載器
        print("\n📊 創建數據加載器...")
        train_loader, tokenizer = create_dataloader(
            dataset_name=dataset_name,
            split='train',
            batch_size=8,
            max_length=16,
            vocab_size=200,
            max_samples=50,  # 很少的樣本
            shuffle=True
        )
        
        val_loader, _ = create_dataloader(
            dataset_name=dataset_name,
            split='val',
            batch_size=8,
            max_length=16,
            vocab_size=200,
            max_samples=10,
            shuffle=False
        )
        
        print(f"✅ 數據加載器創建成功")
        print(f"📝 詞彙大小: {tokenizer.get_vocab_size()}")
        print(f"📦 訓練批次數: {len(train_loader)}")
        print(f"📦 驗證批次數: {len(val_loader)}")
        
        # 創建QRWKV模型
        print(f"\n🤖 創建QRWKV模型...")
        vocab_size = tokenizer.get_vocab_size()
        config = ModelConfig(
            vocab_size=vocab_size,
            n_embd=64,
            n_head=8,
            n_layer=2,
            n_qubits=2,
            block_size=256
        )
        model = QuantumRWKVModel(config)
        model.to(device)
        
        print(f"✅ 模型創建成功")
        print(f"📊 模型參數數量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 簡單訓練
        print(f"\n🎯 開始訓練...")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training
            model.train()
            train_loss = 0.0
            train_batches = 0
            
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
            epoch_time = time.time() - epoch_start_time
            
            print(f"Epoch {epoch+1:2d}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Time: {epoch_time:.2f}s")
        
        training_time = time.time() - start_time
        
        # 生成文本
        print(f"\n📝 生成文本...")
        test_prompts = ["the", "birds"]
        generated_texts = []
        
        model.eval()
        with torch.no_grad():
            for prompt in test_prompts:
                try:
                    prompt_tokens = tokenizer.encode(prompt)
                    input_ids = torch.tensor([prompt_tokens], device=device)
                    
                    generated = input_ids.clone()
                    for step in range(5):
                        outputs = model(generated)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        
                        next_token_logits = outputs[0, -1, :]
                        next_token = torch.argmax(next_token_logits).unsqueeze(0)
                        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                    
                    generated_text = tokenizer.decode(generated[0].tolist())
                    generated_texts.append(generated_text)
                    print(f"  '{prompt}' -> '{generated_text}'")
                except Exception as e:
                    print(f"  ❌ 生成失敗: {e}")
                    generated_texts.append(f"[生成失敗: {prompt}]")
        
        # 保存結果
        print(f"\n💾 保存結果...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = f"./experiments/quick_validation_{timestamp}"
        os.makedirs(result_dir, exist_ok=True)
        
        results = {
            'model_type': model_type,
            'dataset_name': dataset_name,
            'final_train_loss': avg_train_loss,
            'final_val_loss': avg_val_loss,
            'training_time': training_time,
            'generated_texts': generated_texts,
            'vocab_size': vocab_size
        }
        
        with open(os.path.join(result_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ 快速驗證完成!")
        print(f"⏱️  總訓練時間: {training_time:.2f}秒")
        print(f"📊 最終驗證損失: {avg_val_loss:.4f}")
        print(f"📁 結果保存到: {result_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ 快速驗證失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = quick_test()
    if success:
        print("\n🎉 快速驗證成功！實驗流程正常。")
    else:
        print("\n⚠️  快速驗證失敗，需要檢查問題。")
