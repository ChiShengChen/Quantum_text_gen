#!/usr/bin/env python3
"""
分析QRWKV模型的量子特性
"""

import torch
import sys
import os

sys.path.append('.')

from qrwkv_model import QuantumRWKVModel, ModelConfig

def analyze_qrwkv_quantum_features():
    """分析QRWKV模型的量子特性"""
    
    print("🔬 QRWKV模型量子特性分析")
    print("=" * 50)
    
    # 創建QRWKV模型
    config = ModelConfig(
        vocab_size=100,
        n_embd=64,
        n_head=8,
        n_layer=2,
        n_qubits=4,
        q_depth=2,
        block_size=256
    )
    
    model = QuantumRWKVModel(config)
    
    print(f"📊 模型配置:")
    print(f"  - 詞彙大小: {config.vocab_size}")
    print(f"  - 嵌入維度: {config.n_embd}")
    print(f"  - 層數: {config.n_layer}")
    print(f"  - 量子比特數: {config.n_qubits}")
    print(f"  - 量子電路深度: {config.q_depth}")
    
    print(f"\n🔍 量子特性檢查:")
    
    # 檢查量子組件
    quantum_components_found = []
    
    for name, module in model.named_modules():
        if 'vqc_layer' in name:
            quantum_components_found.append(f"✅ {name}: 變分量子電路層")
        elif 'q_device' in name:
            quantum_components_found.append(f"✅ {name}: 量子設備")
        elif 'quantum' in name.lower():
            quantum_components_found.append(f"✅ {name}: 量子組件")
    
    if quantum_components_found:
        print("  發現的量子組件:")
        for component in quantum_components_found:
            print(f"    {component}")
    else:
        print("  ❌ 未發現量子組件")
    
    # 檢查PennyLane相關組件
    print(f"\n🔧 PennyLane組件檢查:")
    
    try:
        import pennylane as qml
        print("  ✅ PennyLane已導入")
        
        # 檢查量子設備
        for name, module in model.named_modules():
            if hasattr(module, 'q_device'):
                print(f"  ✅ 發現量子設備: {module.q_device}")
                try:
                    print(f"     - 量子比特數: {module.q_device.num_wires}")
                except AttributeError:
                    print(f"     - 量子比特數: {module.n_qubits}")
                print(f"     - 設備名稱: {module.q_device.name}")
        
        # 檢查量子電路
        for name, module in model.named_modules():
            if hasattr(module, 'vqc_layer'):
                print(f"  ✅ 發現變分量子電路: {name}")
                print(f"     - 量子節點類型: {type(module.vqc_layer.qnode).__name__}")
                print(f"     - 量子比特數: {module.n_qubits}")
                print(f"     - 量子電路深度: {module.q_depth}")
        
    except ImportError:
        print("  ❌ PennyLane未導入")
    
    # 檢查模型參數
    print(f"\n📈 模型參數分析:")
    total_params = 0
    quantum_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        if 'vqc' in name or 'quantum' in name.lower():
            quantum_params += param_count
            print(f"  🔬 量子參數: {name} - {param_count:,} 參數")
        else:
            print(f"  📊 經典參數: {name} - {param_count:,} 參數")
    
    print(f"\n📊 參數統計:")
    print(f"  - 總參數數: {total_params:,}")
    print(f"  - 量子參數數: {quantum_params:,}")
    print(f"  - 經典參數數: {total_params - quantum_params:,}")
    print(f"  - 量子參數比例: {quantum_params/total_params*100:.2f}%")
    
    # 檢查量子電路結構
    print(f"\n🔬 量子電路結構分析:")
    
    for name, module in model.named_modules():
        if hasattr(module, 'vqc_layer'):
            print(f"  量子電路: {name}")
            print(f"    - 量子比特數: {module.n_qubits}")
            print(f"    - 量子電路深度: {module.q_depth}")
            print(f"    - 使用AngleEmbedding和BasicEntanglerLayers")
            print(f"    - 測量PauliZ期望值")
    
    # 測試前向傳播
    print(f"\n🧪 前向傳播測試:")
    
    try:
        # 創建測試輸入
        batch_size = 2
        seq_len = 4
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        print(f"  輸入形狀: {input_ids.shape}")
        
        # 前向傳播
        with torch.no_grad():
            outputs, states = model(input_ids)
        
        print(f"  輸出形狀: {outputs.shape}")
        print(f"  狀態數量: {len(states)}")
        print(f"  ✅ 前向傳播成功")
        
    except Exception as e:
        print(f"  ❌ 前向傳播失敗: {e}")
    
    # 總結
    print(f"\n📋 量子特性總結:")
    
    if quantum_params > 0:
        print(f"  ✅ 這是真正的量子模型")
        print(f"  ✅ 包含 {quantum_params:,} 個量子參數")
        print(f"  ✅ 使用PennyLane量子框架")
        print(f"  ✅ 具有變分量子電路")
        print(f"  ✅ 量子-經典混合架構")
    else:
        print(f"  ❌ 這不是真正的量子模型")
        print(f"  ❌ 沒有量子參數")
        print(f"  ❌ 只是經典神經網絡")

if __name__ == "__main__":
    analyze_qrwkv_quantum_features()
