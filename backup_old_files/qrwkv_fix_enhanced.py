#!/usr/bin/env python3
"""
Enhanced QRWKV Model Compatibility Fix
基於架構分析的增強版QRWKV修復工具
"""

import torch
import torch.nn as nn
import json
import os
import sys
from typing import Dict, Any, Optional, Tuple

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qrwkv_model import QuantumRWKVModel, ModelConfig

class EnhancedQRWKVCheckpointFixer:
    """增強版QRWKV檢查點修復工具"""
    
    def __init__(self):
        self.supported_versions = {
            'v1': {'hidden_dim': 256, 'n_layers': 6, 'n_qubits': 4},
            'v2': {'hidden_dim': 128, 'n_layers': 4, 'n_qubits': 6}
        }
    
    def improved_detect_checkpoint_version(self, checkpoint_path: str) -> Dict[str, Any]:
        """改進的檢查點版本檢測 - 基於架構分析"""
        print(f"🔍 改進檢測檢查點版本: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model_state_dict']
        
        print(f"📊 檢查點包含 {len(state_dict)} 個參數")
        
        # 1. 檢測詞彙表大小
        vocab_size = None
        if 'qrwkv_model.wte.weight' in state_dict:
            vocab_size = state_dict['qrwkv_model.wte.weight'].shape[0]
            print(f"📝 檢測到詞彙表大小: {vocab_size}")
        else:
            print("⚠️  未找到詞嵌入層，無法檢測詞彙表大小")
        
        # 2. 檢測嵌入維度
        hidden_dim = None
        if 'qrwkv_model.wte.weight' in state_dict:
            hidden_dim = state_dict['qrwkv_model.wte.weight'].shape[1]
            print(f"🔢 檢測到嵌入維度: {hidden_dim}")
        else:
            print("⚠️  未找到詞嵌入層，無法檢測嵌入維度")
        
        # 3. 改進層數檢測
        max_layer = -1
        layer_patterns = [
            ('blocks.', '.ln1.weight'),
            ('blocks.', '.ln2.weight'),
            ('blocks.', '.att.time_mix_k'),
            ('blocks.', '.ffn.time_mix_k')
        ]
        
        for key in state_dict.keys():
            for prefix, suffix in layer_patterns:
                if key.startswith(prefix) and key.endswith(suffix):
                    try:
                        # 提取層號: blocks.X.xxx -> X
                        parts = key.split('.')
                        if len(parts) >= 2:
                            layer_num = int(parts[1])
                            max_layer = max(max_layer, layer_num)
                            break
                    except (ValueError, IndexError):
                        continue
        
        n_layers = max_layer + 1 if max_layer >= 0 else 0
        print(f"🏗️  檢測到層數: {n_layers}")
        
        # 4. 檢測量子比特數
        n_qubits = None
        for key in state_dict.keys():
            if 'vqc_layer.weights' in key:
                n_qubits = state_dict[key].shape[1]
                print(f"⚛️  檢測到量子比特數: {n_qubits}")
                break
        
        if n_qubits is None:
            print("⚠️  未找到量子電路參數，使用默認值")
            n_qubits = 4
        
        # 5. 檢測其他參數
        n_head = 12  # 默認值
        q_depth = 2  # 默認值
        n_intermediate = hidden_dim * 4 if hidden_dim else 1024
        
        # 檢測中間層維度
        for key in state_dict.keys():
            if 'classical_fc1.weight' in key:
                n_intermediate = state_dict[key].shape[0]
                print(f"🔗 檢測到中間層維度: {n_intermediate}")
                break
        
        detected_config = {
            'vocab_size': vocab_size,
            'hidden_dim': hidden_dim,
            'n_layers': n_layers,
            'n_qubits': n_qubits,
            'n_head': n_head,
            'q_depth': q_depth,
            'n_intermediate': n_intermediate
        }
        
        print(f"✅ 檢測完成: {detected_config}")
        return detected_config
    
    def create_compatible_model(self, 
                            checkpoint_config: Dict[str, Any]) -> QuantumRWKVModel:
        """創建與檢查點兼容的模型 - 使用檢測到的配置"""
        
        print("🔧 創建兼容模型...")
        
        # 驗證必要參數
        required_params = ['vocab_size', 'hidden_dim', 'n_layers', 'n_qubits']
        for param in required_params:
            if checkpoint_config.get(param) is None:
                raise ValueError(f"缺少必要參數: {param}")
        
        model_config = ModelConfig(
            vocab_size=checkpoint_config['vocab_size'],
            n_embd=checkpoint_config['hidden_dim'],
            n_layer=checkpoint_config['n_layers'],
            n_qubits=checkpoint_config['n_qubits'],
            n_head=checkpoint_config.get('n_head', 12),
            q_depth=checkpoint_config.get('q_depth', 2),
            n_intermediate=checkpoint_config.get('n_intermediate', checkpoint_config['hidden_dim'] * 4),
            block_size=512
        )
        
        model = QuantumRWKVModel(model_config)
        print(f"✅ 模型創建完成: {model_config}")
        return model
    
    def handle_vocab_size_mismatch(self, old_key: str, old_value: torch.Tensor, 
                                new_state_dict: Dict[str, torch.Tensor],
                                target_vocab_size: int) -> bool:
        """處理詞彙表大小不匹配"""
        
        if 'wte.weight' in old_key:
            # 詞嵌入層
            old_vocab_size, hidden_dim = old_value.shape
            new_weight = new_state_dict[old_key]
            
            if old_vocab_size < target_vocab_size:
                # 需要擴展詞彙表
                print(f"📈 擴展詞彙表: {old_vocab_size} -> {target_vocab_size}")
                # 複製現有權重，新詞彙使用隨機初始化
                new_weight[:old_vocab_size] = old_value
                # 新詞彙使用小隨機值初始化
                new_weight[old_vocab_size:] = torch.randn(target_vocab_size - old_vocab_size, hidden_dim) * 0.02
            else:
                # 需要截斷詞彙表
                print(f"📉 截斷詞彙表: {old_vocab_size} -> {target_vocab_size}")
                new_weight[:target_vocab_size] = old_value[:target_vocab_size]
            
            return True
            
        elif 'lm_head.weight' in old_key:
            # 語言模型頭
            old_vocab_size, hidden_dim = old_value.shape
            new_weight = new_state_dict[old_key]
            
            if old_vocab_size < target_vocab_size:
                # 需要擴展
                print(f"📈 擴展語言模型頭: {old_vocab_size} -> {target_vocab_size}")
                new_weight[:old_vocab_size] = old_value
                new_weight[old_vocab_size:] = torch.randn(target_vocab_size - old_vocab_size, hidden_dim) * 0.02
            else:
                # 需要截斷
                print(f"📉 截斷語言模型頭: {old_vocab_size} -> {target_vocab_size}")
                new_weight[:target_vocab_size] = old_value[:target_vocab_size]
            
            return True
        
        return False
    
    def optimize_parameter_mapping(self, old_state_dict: Dict[str, torch.Tensor],
                                new_state_dict: Dict[str, torch.Tensor],
                                target_vocab_size: int = None) -> Tuple[int, int, int]:
        """優化參數映射策略"""
        
        print("🔄 開始參數映射...")
        
        # 移除前綴
        cleaned_old_dict = {}
        for key, value in old_state_dict.items():
            if key.startswith('qrwkv_model.'):
                new_key = key.replace('qrwkv_model.', '')
                cleaned_old_dict[new_key] = value
            else:
                cleaned_old_dict[key] = value
        
        mapped_count = 0
        skipped_count = 0
        error_count = 0
        
        for old_key, old_value in cleaned_old_dict.items():
            if old_key in new_state_dict:
                new_value = new_state_dict[old_key]
                
                if old_value.shape == new_value.shape:
                    # 形狀匹配，直接複製
                    new_state_dict[old_key] = old_value
                    mapped_count += 1
                else:
                    # 形狀不匹配，嘗試處理
                    if target_vocab_size and ('wte.weight' in old_key or 'lm_head.weight' in old_key):
                        if self.handle_vocab_size_mismatch(old_key, old_value, new_state_dict, target_vocab_size):
                            mapped_count += 1
                        else:
                            print(f"⚠️  詞彙表處理失敗: {old_key} {old_value.shape} -> {new_value.shape}")
                            skipped_count += 1
                    else:
                        print(f"⚠️  形狀不匹配跳過: {old_key} {old_value.shape} -> {new_value.shape}")
                        skipped_count += 1
            else:
                print(f"⚠️  鍵不存在跳過: {old_key}")
                skipped_count += 1
        
        print(f"✅ 參數映射完成: {mapped_count} 個成功, {skipped_count} 個跳過, {error_count} 個錯誤")
        return mapped_count, skipped_count, error_count
    
    def migrate_checkpoint(self, 
                        old_checkpoint_path: str,
                        new_checkpoint_path: str,
                        target_vocab_size: int = None) -> bool:
        """遷移檢查點到新的配置"""
        
        print(f"🔄 遷移檢查點: {old_checkpoint_path} -> {new_checkpoint_path}")
        
        # 檢測檢查點配置
        detected_config = self.improved_detect_checkpoint_version(old_checkpoint_path)
        
        # 如果沒有指定目標詞彙表大小，使用檢測到的
        if target_vocab_size is None:
            target_vocab_size = detected_config['vocab_size']
        
        # 創建新模型
        new_model = self.create_compatible_model(detected_config)
        new_state_dict = new_model.state_dict()
        
        # 加載舊檢查點
        old_checkpoint = torch.load(old_checkpoint_path, map_location='cpu', weights_only=False)
        old_state_dict = old_checkpoint['model_state_dict']
        
        # 優化參數映射
        mapped_count, skipped_count, error_count = self.optimize_parameter_mapping(
            old_state_dict, new_state_dict, target_vocab_size
        )
        
        # 保存新檢查點
        new_checkpoint = {
            'model_state_dict': new_state_dict,
            'config': detected_config,
            'target_vocab_size': target_vocab_size,
            'migration_info': {
                'original_checkpoint': old_checkpoint_path,
                'mapped_params': mapped_count,
                'skipped_params': skipped_count,
                'error_params': error_count,
                'migration_success_rate': mapped_count / (mapped_count + skipped_count + error_count) if (mapped_count + skipped_count + error_count) > 0 else 0
            }
        }
        
        torch.save(new_checkpoint, new_checkpoint_path)
        
        print(f"✅ 遷移完成: {mapped_count} 個參數遷移, {skipped_count} 個參數跳過")
        print(f"📊 遷移成功率: {new_checkpoint['migration_info']['migration_success_rate']:.2%}")
        
        return True
    
    def fix_experiment_checkpoints(self, experiment_dir: str = "./experiments/qrwkv"):
        """修復實驗目錄中的所有檢查點"""
        
        print(f"🔧 修復實驗目錄: {experiment_dir}")
        
        if not os.path.exists(experiment_dir):
            print(f"❌ 目錄不存在: {experiment_dir}")
            return
        
        fixed_count = 0
        total_count = 0
        
        # 遍歷所有子目錄
        for dataset_dir in os.listdir(experiment_dir):
            dataset_path = os.path.join(experiment_dir, dataset_dir)
            if not os.path.isdir(dataset_path):
                continue
            
            print(f"📁 處理數據集: {dataset_dir}")
            
            # 查找檢查點文件
            checkpoint_files = []
            for file in os.listdir(dataset_path):
                if file.endswith('.pth') or file.endswith('.pt'):
                    if not file.endswith('_fixed.pth') and not file.endswith('_fixed.pt'):
                        checkpoint_files.append(os.path.join(dataset_path, file))
            
            for checkpoint_path in checkpoint_files:
                total_count += 1
                try:
                    print(f"🔧 修復檢查點: {os.path.basename(checkpoint_path)}")
                    
                    # 創建修復版本路徑
                    fixed_path = checkpoint_path.replace('.pth', '_fixed.pth').replace('.pt', '_fixed.pt')
                    
                    # 遷移檢查點
                    success = self.migrate_checkpoint(checkpoint_path, fixed_path)
                    
                    if success:
                        fixed_count += 1
                        print(f"✅ 修復成功: {os.path.basename(fixed_path)}")
                    else:
                        print(f"❌ 修復失敗: {os.path.basename(checkpoint_path)}")
                        
                except Exception as e:
                    print(f"❌ 修復失敗 {os.path.basename(checkpoint_path)}: {e}")
        
        print(f"🎉 修復完成: {fixed_count}/{total_count} 個檢查點修復成功")

def create_qrwkv_compatible_model(checkpoint_path: str, 
                                target_vocab_size: int = None,
                                device: str = 'cpu') -> QuantumRWKVModel:
    """創建與檢查點兼容的QRWKV模型 - 增強版"""
    
    fixer = EnhancedQRWKVCheckpointFixer()
    
    # 檢測檢查點配置
    detected_config = fixer.improved_detect_checkpoint_version(checkpoint_path)
    
    # 如果沒有指定目標詞彙表大小，使用檢測到的
    if target_vocab_size is None:
        target_vocab_size = detected_config['vocab_size']
    
    # 創建兼容模型
    model = fixer.create_compatible_model(detected_config)
    model.to(device)
    
    # 加載檢查點
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    # 優化參數映射
    new_state_dict = model.state_dict()
    mapped_count, skipped_count, error_count = fixer.optimize_parameter_mapping(
        state_dict, new_state_dict, target_vocab_size
    )
    
    # 加載到模型
    model.load_state_dict(new_state_dict)
    
    print(f"✅ QRWKV模型加載成功: {detected_config}")
    print(f"📊 參數加載: {mapped_count} 個成功, {skipped_count} 個跳過")
    
    return model

def test_enhanced_qrwkv_fix():
    """測試增強版QRWKV修復功能"""
    
    print("🧪 測試增強版QRWKV修復功能")
    print("=" * 60)
    
    # 查找QRWKV檢查點
    experiment_dir = "./experiments/qrwkv"
    checkpoint_path = None
    
    if os.path.exists(experiment_dir):
        for dataset_dir in os.listdir(experiment_dir):
            dataset_path = os.path.join(experiment_dir, dataset_dir)
            if os.path.isdir(dataset_path):
                for file in os.listdir(dataset_path):
                    if file.endswith('.pth') or file.endswith('.pt'):
                        if not file.endswith('_fixed.pth') and not file.endswith('_fixed.pt'):
                            checkpoint_path = os.path.join(dataset_path, file)
                            break
                if checkpoint_path:
                    break
    
    if not checkpoint_path:
        print("❌ 未找到QRWKV檢查點文件")
        return False
    
    print(f"📁 找到檢查點: {checkpoint_path}")
    
    try:
        # 測試修復
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🖥️  使用設備: {device}")
        
        # 使用檢測到的詞彙表大小
        model = create_qrwkv_compatible_model(checkpoint_path, device=device)
        
        # 測試前向傳播
        batch_size = 2
        seq_len = 16
        
        # 獲取實際詞彙表大小
        vocab_size = model.config.vocab_size
        print(f"📝 使用詞彙表大小: {vocab_size}")
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        with torch.no_grad():
            outputs, states = model(input_ids)
        
        print(f"✅ 前向傳播成功: {outputs.shape}")
        print("🎉 增強版QRWKV修復測試通過!")
        return True
        
    except Exception as e:
        print(f"❌ 增強版QRWKV修復測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("🚀 啟動增強版QRWKV修復工具")
    print("=" * 60)
    
    # 修復所有實驗檢查點
    fixer = EnhancedQRWKVCheckpointFixer()
    fixer.fix_experiment_checkpoints()
    
    # 測試修復功能
    test_enhanced_qrwkv_fix()
