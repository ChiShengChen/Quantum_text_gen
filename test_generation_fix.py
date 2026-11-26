#!/usr/bin/env python3
"""
測試生成邏輯修復效果
"""

import torch
import sys
import os

# 添加當前目錄到路徑
sys.path.append('.')

from data.datasets import create_dataloader
from models.quantum_text_models import create_quantum_text_model
from comprehensive_evaluation import generate_texts_for_evaluation, get_reference_texts

def test_generation_fix():
    """測試生成邏輯修復"""
    print("🧪 測試生成邏輯修復效果")
    print("=" * 50)
    
    # 設置設備
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  使用設備: {device}")
    
    # 創建數據加載器和tokenizer
    print("\n📊 創建數據加載器...")
    train_loader, tokenizer = create_dataloader(
        dataset_name='simple_sentences',
        split='train',
        batch_size=4,
        max_length=16,
        vocab_size=200,
        max_samples=100,
        shuffle=True
    )
    
    print(f"✅ 詞彙大小: {tokenizer.get_vocab_size()}")
    
    # 測試不同模型
    models_to_test = ['qksan', 'qrwkv', 'qasa']
    
    for model_type in models_to_test:
        print(f"\n🤖 測試 {model_type.upper()} 模型...")
        
        try:
            # 創建模型
            model = create_quantum_text_model(
                model_type=model_type,
                vocab_size=tokenizer.get_vocab_size(),
                hidden_dim=64,
                n_qubits=4,
                device=device
            )
            
            model.to(device)
            print(f"✅ 模型創建成功，參數數量: {sum(p.numel() for p in model.parameters()):,}")
            
            # 簡單訓練幾個epoch
            print("🎯 進行簡單訓練...")
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()
            
            model.train()
            for epoch in range(3):
                total_loss = 0
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
                    
                    total_loss += loss.item()
                
                print(f"  Epoch {epoch+1}/3, Loss: {total_loss/len(train_loader):.4f}")
            
            # 測試生成
            print("📝 測試文本生成...")
            prompts = ["the", "birds", "water", "spring", "stars"]
            
            generated_texts = generate_texts_for_evaluation(
                model, tokenizer, prompts, device, max_length=10
            )
            
            print(f"✅ {model_type.upper()} 生成結果:")
            for i, (prompt, generated) in enumerate(zip(prompts, generated_texts)):
                print(f"  {i+1}. 提示: '{prompt}' -> 生成: '{generated}'")
            
        except Exception as e:
            print(f"❌ {model_type.upper()} 測試失敗: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n🎉 測試完成!")

if __name__ == "__main__":
    test_generation_fix()
