#!/usr/bin/env python3
"""
測試不同Top-K和溫度參數對生成效果的影響
"""

import torch
import sys
import os

# 添加當前目錄到路徑
sys.path.append('.')

from data.datasets import create_dataloader
from models.quantum_text_models import create_quantum_text_model

def test_parameter_effects():
    """測試不同參數對生成效果的影響"""
    print("🧪 測試Top-K和溫度參數對生成效果的影響")
    print("=" * 60)
    
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
    
    # 創建模型
    print("\n🤖 創建QKSAN模型...")
    model = create_quantum_text_model(
        model_type='qksan',
        vocab_size=tokenizer.get_vocab_size(),
        hidden_dim=64,
        n_qubits=4,
        device=device
    )
    
    model.to(device)
    print(f"✅ 模型創建成功，參數數量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 簡單訓練
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
    
    # 測試不同參數組合
    print("\n🔬 測試不同參數組合...")
    prompt = "the"
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    # 參數組合
    parameter_combinations = [
        {"temperature": 0.5, "top_k": 5, "name": "低溫度+小Top-K (保守)"},
        {"temperature": 0.5, "top_k": 20, "name": "低溫度+大Top-K (保守但多樣)"},
        {"temperature": 1.0, "top_k": 5, "name": "標準溫度+小Top-K (平衡)"},
        {"temperature": 1.0, "top_k": 20, "name": "標準溫度+大Top-K (平衡多樣)"},
        {"temperature": 1.5, "top_k": 5, "name": "高溫度+小Top-K (隨機但限制)"},
        {"temperature": 1.5, "top_k": 20, "name": "高溫度+大Top-K (高隨機性)"},
    ]
    
    for params in parameter_combinations:
        print(f"\n--- {params['name']} ---")
        print(f"溫度: {params['temperature']}, Top-K: {params['top_k']}")
        
        try:
            # 生成文本
            generated = model.generate(
                input_tensor,
                max_length=8,
                temperature=params['temperature'],
                top_k=params['top_k'],
                do_sample=True
            )
            
            # 解碼
            generated_text = tokenizer.decode(generated[0].tolist())
            print(f"生成結果: '{generated_text}'")
            
        except Exception as e:
            print(f"生成失敗: {e}")
    
    # 測試溫度對概率分布的影響
    print("\n🌡️ 測試溫度對概率分布的影響...")
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        logits = outputs[:, -1, :]  # 取最後一個位置的logits
        
        temperatures = [0.5, 1.0, 1.5, 2.0]
        for temp in temperatures:
            print(f"\n溫度 = {temp}:")
            scaled_logits = logits / temp
            probs = torch.softmax(scaled_logits, dim=-1)
            
            # 顯示前5個最高概率的詞
            top_probs, top_indices = torch.topk(probs, 5)
            for i in range(5):
                word = tokenizer.id_to_token(top_indices[0, i].item())
                prob = top_probs[0, i].item()
                print(f"  '{word}': {prob:.4f}")
    
    # 測試Top-K對詞彙選擇的影響
    print("\n🔝 測試Top-K對詞彙選擇的影響...")
    
    top_k_values = [1, 3, 5, 10, 20]
    for k in top_k_values:
        print(f"\nTop-K = {k}:")
        
        # 應用Top-K過濾
        top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
        filtered_logits = torch.full_like(logits, float('-inf'))
        filtered_logits.scatter_(1, top_k_indices, top_k_logits)
        
        # 計算概率
        probs = torch.softmax(filtered_logits, dim=-1)
        
        # 顯示可選的詞
        print(f"  可選詞彙 ({k}個):")
        for i in range(k):
            word = tokenizer.id_to_token(top_k_indices[0, i].item())
            prob = probs[0, top_k_indices[0, i]].item()
            print(f"    '{word}': {prob:.4f}")
    
    print("\n🎉 測試完成!")

if __name__ == "__main__":
    test_parameter_effects()
