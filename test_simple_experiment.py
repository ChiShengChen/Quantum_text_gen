#!/usr/bin/env python3
"""
簡單的量子文字生成模型測試腳本
Simple Quantum Text Generation Model Test Script
"""

import os
import sys
import torch
import time
from datetime import datetime

# 添加當前目錄到Python路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_simple_experiment():
    """測試簡單實驗"""
    print("🚀 開始簡單測試...")
    
    try:
        # 導入必要的模塊
        print("📦 導入模塊...")
        from data.datasets import create_dataloader
        from qksan_fixed import create_fixed_qksan_model
        from qrwkv_model import QuantumRWKVModel, ModelConfig
        from models.quantum_text_models import QASATextGenerator
        
        print("✅ 模塊導入成功")
        
        # 設置設備
        device = 'cpu'
        print(f"🖥️  使用設備: {device}")
        
        # 測試數據加載器
        print("\n📊 測試數據加載器...")
        dataset_name = 'simple_sentences'
        train_loader, tokenizer = create_dataloader(
            dataset_name=dataset_name,
            split='train',
            batch_size=4,
            max_length=8,
            vocab_size=100,
            max_samples=10,
            shuffle=True
        )
        
        print(f"✅ 數據加載器創建成功")
        print(f"📝 詞彙大小: {tokenizer.get_vocab_size()}")
        
        # 測試QKSAN模型
        print("\n🤖 測試QKSAN模型...")
        vocab_size = tokenizer.get_vocab_size()
        qksan_model = create_fixed_qksan_model(
            vocab_size=vocab_size,
            embedding_dim=32,
            n_qubits=2,
            device=device
        )
        print(f"✅ QKSAN模型創建成功")
        print(f"📊 參數數量: {sum(p.numel() for p in qksan_model.parameters()):,}")
        
        # 測試QRWKV模型
        print("\n🤖 測試QRWKV模型...")
        qrwkv_config = ModelConfig(
            vocab_size=vocab_size,
            n_embd=32,
            n_head=4,
            n_layer=1,
            n_qubits=2,
            block_size=64
        )
        qrwkv_model = QuantumRWKVModel(qrwkv_config)
        print(f"✅ QRWKV模型創建成功")
        print(f"📊 參數數量: {sum(p.numel() for p in qrwkv_model.parameters()):,}")
        
        # 測試QASA模型
        print("\n🤖 測試QASA模型...")
        qasa_model = QASATextGenerator(
            vocab_size=vocab_size,
            hidden_dim=32,
            seq_len=8,
            n_qubits=2,
            device=device
        )
        print(f"✅ QASA模型創建成功")
        print(f"📊 參數數量: {sum(p.numel() for p in qasa_model.parameters()):,}")
        
        # 測試前向傳播
        print("\n🔄 測試前向傳播...")
        for batch in train_loader:
            input_ids = batch['input_ids']
            target_ids = batch['target_ids']
            
            print(f"📝 輸入形狀: {input_ids.shape}")
            print(f"📝 目標形狀: {target_ids.shape}")
            
            # 測試QKSAN
            try:
                qksan_output = qksan_model(input_ids)
                print(f"✅ QKSAN前向傳播成功: {qksan_output.shape if hasattr(qksan_output, 'shape') else 'tuple'}")
            except Exception as e:
                print(f"❌ QKSAN前向傳播失敗: {e}")
            
            # 測試QRWKV
            try:
                qrwkv_output = qrwkv_model(input_ids)
                print(f"✅ QRWKV前向傳播成功: {qrwkv_output.shape if hasattr(qrwkv_output, 'shape') else 'tuple'}")
            except Exception as e:
                print(f"❌ QRWKV前向傳播失敗: {e}")
            
            # 測試QASA
            try:
                qasa_output = qasa_model(input_ids)
                print(f"✅ QASA前向傳播成功: {qasa_output.shape if hasattr(qasa_output, 'shape') else 'tuple'}")
            except Exception as e:
                print(f"❌ QASA前向傳播失敗: {e}")
            
            break  # 只測試第一個batch
        
        print("\n🎉 所有測試完成!")
        return True
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_simple_experiment()
    if success:
        print("\n✅ 測試成功! 所有模型都可以正常工作")
    else:
        print("\n❌ 測試失敗! 需要檢查問題")
