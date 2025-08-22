#!/usr/bin/env python3
"""
Test All Models
測試所有模型的基本功能
"""

import torch
import sys
sys.path.append('.')

from models.quantum_text_models import QASATextGenerator
from qksan_fixed import create_fixed_qksan_model
from qrwkv_model import QuantumRWKVModel, ModelConfig

def test_qasa():
    """測試QASA模型"""
    print("🧪 測試QASA模型...")
    try:
        model = QASATextGenerator(
            vocab_size=100,
            hidden_dim=64,
            seq_len=16,
            n_qubits=4,
            device='cpu'
        )
        
        input_ids = torch.randint(0, 100, (2, 10))
        outputs = model(input_ids)
        
        print(f"✅ QASA模型測試成功")
        print(f"📊 輸入形狀: {input_ids.shape}")
        print(f"📊 輸出形狀: {outputs.shape}")
        return True
    except Exception as e:
        print(f"❌ QASA模型測試失敗: {e}")
        return False

def test_qksan():
    """測試QKSAN模型"""
    print("\n🧪 測試QKSAN模型...")
    try:
        model = create_fixed_qksan_model(
            vocab_size=100,
            embedding_dim=64,
            n_qubits=4,
            device='cpu'
        )
        
        input_ids = torch.randint(0, 100, (2, 10))
        outputs = model(input_ids)
        
        print(f"✅ QKSAN模型測試成功")
        print(f"📊 輸入形狀: {input_ids.shape}")
        print(f"📊 輸出形狀: {outputs.shape}")
        return True
    except Exception as e:
        print(f"❌ QKSAN模型測試失敗: {e}")
        return False

def test_qrwkv():
    """測試QRWKV模型"""
    print("\n🧪 測試QRWKV模型...")
    try:
        config = ModelConfig(
            vocab_size=100,
            n_embd=64,
            n_head=8,
            n_layer=2,
            n_qubits=2,
            block_size=256
        )
        model = QuantumRWKVModel(config)
        
        input_ids = torch.randint(0, 100, (2, 10))
        outputs = model(input_ids)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        print(f"✅ QRWKV模型測試成功")
        print(f"📊 輸入形狀: {input_ids.shape}")
        print(f"📊 輸出形狀: {outputs.shape}")
        return True
    except Exception as e:
        print(f"❌ QRWKV模型測試失敗: {e}")
        return False

def main():
    """主函數"""
    print("🚀 量子文字生成模型測試")
    print("=" * 50)
    
    results = {}
    
    # 測試所有模型
    results['qasa'] = test_qasa()
    results['qksan'] = test_qksan()
    results['qrwkv'] = test_qrwkv()
    
    # 總結
    print("\n" + "=" * 50)
    print("📊 測試總結:")
    
    passed = sum(results.values())
    total = len(results)
    
    for model, success in results.items():
        status = "✅ 通過" if success else "❌ 失敗"
        print(f"  {model.upper()}: {status}")
    
    print(f"\n🎯 總體結果: {passed}/{total} 個模型測試通過")
    
    if passed == total:
        print("🎉 所有模型測試通過！可以開始訓練實驗。")
    else:
        print("⚠️  部分模型測試失敗，需要修復後再進行訓練。")

if __name__ == '__main__':
    main()
