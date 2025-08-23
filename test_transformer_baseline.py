#!/usr/bin/env python3
"""
測試Transformer Baseline模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformer_baseline import create_transformer_baseline_model
import numpy as np

def test_transformer_model():
    """測試Transformer模型的基本功能"""
    print("🧪 測試Transformer Baseline模型")
    print("="*50)
    
    # 模型參數
    vocab_size = 100
    d_model = 128
    n_heads = 8
    n_layers = 2
    max_len = 32
    batch_size = 4
    seq_len = 10
    
    print(f"📊 模型參數:")
    print(f"   - 詞彙大小: {vocab_size}")
    print(f"   - 模型維度: {d_model}")
    print(f"   - 注意力頭數: {n_heads}")
    print(f"   - 層數: {n_layers}")
    print(f"   - 最大長度: {max_len}")
    
    # 創建模型
    model = create_transformer_baseline_model(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_len=max_len
    )
    
    print(f"✅ 模型創建成功")
    print(f"📈 模型參數數量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 測試前向傳播
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"📥 輸入形狀: {input_ids.shape}")
    
    try:
        logits = model(input_ids)
        print(f"📤 輸出形狀: {logits.shape}")
        print(f"✅ 前向傳播測試通過")
    except Exception as e:
        print(f"❌ 前向傳播測試失敗: {e}")
        return False
    
    # 測試生成功能
    try:
        generated = model.generate(input_ids, max_length=15, temperature=1.0)
        print(f"🎯 生成形狀: {generated.shape}")
        print(f"✅ 生成功能測試通過")
    except Exception as e:
        print(f"❌ 生成功能測試失敗: {e}")
        return False
    
    # 測試訓練
    print(f"\n🏋️  測試訓練功能...")
    
    # 創建簡單的訓練數據
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 定義損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 訓練幾個步驟
    model.train()
    for step in range(5):
        optimizer.zero_grad()
        
        # 前向傳播
        outputs = model(input_ids)
        loss = criterion(outputs.view(-1, vocab_size), target_ids.view(-1))
        
        # 反向傳播
        loss.backward()
        optimizer.step()
        
        if step % 2 == 0:
            print(f"   步驟 {step}: 損失 = {loss.item():.4f}")
    
    print(f"✅ 訓練功能測試通過")
    
    # 測試困惑度計算
    print(f"\n📊 測試困惑度計算...")
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.view(-1, vocab_size)
        targets = target_ids.view(-1)
        
        loss = criterion(logits, targets)
        perplexity = torch.exp(loss)
        
        print(f"   困惑度: {perplexity.item():.4f}")
        print(f"✅ 困惑度計算測試通過")
    
    print(f"\n🎉 所有測試通過！Transformer Baseline模型工作正常")
    return True

def test_with_small_dataset():
    """使用小數據集測試模型"""
    print(f"\n📚 使用小數據集測試...")
    
    # 創建簡單的文本數據
    texts = [
        "hello world",
        "quantum computing",
        "machine learning",
        "artificial intelligence",
        "deep learning"
    ]
    
    # 簡單的詞彙表
    vocab = {}
    for text in texts:
        for word in text.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    
    vocab_size = len(vocab)
    print(f"   詞彙大小: {vocab_size}")
    print(f"   詞彙表: {list(vocab.keys())}")
    
    # 創建模型
    model = create_transformer_baseline_model(
        vocab_size=vocab_size,
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_len=16
    )
    
    # 創建訓練數據
    batch_size = 2
    seq_len = 4
    
    # 簡單的輸入輸出對
    input_ids = torch.tensor([
        [vocab['hello'], vocab['world'], vocab['quantum'], vocab['computing']],
        [vocab['machine'], vocab['learning'], vocab['artificial'], vocab['intelligence']]
    ])
    
    target_ids = torch.tensor([
        [vocab['world'], vocab['quantum'], vocab['computing'], vocab['machine']],
        [vocab['learning'], vocab['artificial'], vocab['intelligence'], vocab['deep']]
    ])
    
    # 訓練
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print(f"   開始訓練...")
    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(input_ids)
        loss = criterion(outputs.view(-1, vocab_size), target_ids.view(-1))
        
        loss.backward()
        optimizer.step()
        
        if epoch % 3 == 0:
            print(f"   Epoch {epoch}: 損失 = {loss.item():.4f}")
    
    # 測試生成
    model.eval()
    with torch.no_grad():
        generated = model.generate(input_ids[:, :2], max_length=6)
        print(f"   生成結果形狀: {generated.shape}")
        print(f"✅ 小數據集測試通過")
    
    return True

if __name__ == "__main__":
    print("🚀 開始測試Transformer Baseline模型")
    print("="*60)
    
    # 基本功能測試
    success1 = test_transformer_model()
    
    # 小數據集測試
    success2 = test_with_small_dataset()
    
    if success1 and success2:
        print(f"\n{'='*60}")
        print("🎉 所有測試通過！Transformer Baseline模型已準備就緒")
        print("💡 現在可以運行完整的評估實驗了")
        print(f"{'='*60}")
    else:
        print(f"\n❌ 部分測試失敗，請檢查模型實現")
