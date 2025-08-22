#!/usr/bin/env python3
"""
Simple Test
簡單測試QRWKV模型
"""

import torch
import sys
sys.path.append('.')

from qrwkv_model import QuantumRWKVModel, ModelConfig

def test_qrwkv():
    print("🧪 測試QRWKV模型...")
    
    # 創建配置
    config = ModelConfig(
        vocab_size=100,
        n_embd=64,
        n_head=8,  # 確保64能被8整除
        n_layer=2,
        n_qubits=2,
        block_size=256
    )
    
    print(f"✅ 配置創建成功")
    print(f"📊 n_embd: {config.n_embd}, n_head: {config.n_head}")
    
    # 創建模型
    model = QuantumRWKVModel(config)
    print(f"✅ 模型創建成功")
    print(f"📊 模型參數數量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 測試前向傳播
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    
    print(f"📊 輸入形狀: {input_ids.shape}")
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        print(f"📊 輸出形狀: {outputs.shape}")
    
    print("✅ QRWKV模型測試成功!")

if __name__ == '__main__':
    test_qrwkv()
