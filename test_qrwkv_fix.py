#!/usr/bin/env python3
"""
測試QRWKV模型修復
"""

import torch
import sys
import os

# 添加項目路徑
sys.path.append('/media/meow/Transcend/Quantum_text_gen')

def test_qrwkv_model():
    """測試QRWKV模型"""
    print("🧪 測試QRWKV模型修復...")
    
    try:
        # 導入必要的模塊
        from data.datasets import create_dataloader
        from qrwkv_model import QuantumRWKVModel, ModelConfig
        
        # 創建數據加載器
        print("📊 創建數據加載器...")
        train_loader, tokenizer = create_dataloader(
            dataset_name='simple_sentences',
            split='train',
            batch_size=4,
            max_length=16,
            vocab_size=100,
            max_samples=50,
            shuffle=True
        )
        
        print(f"✅ 數據加載器創建成功，詞彙大小: {tokenizer.get_vocab_size()}")
        
        # 創建QRWKV模型
        print("🤖 創建QRWKV模型...")
        config = ModelConfig(
            vocab_size=tokenizer.get_vocab_size(),
            n_embd=64,
            n_layer=2,
            n_head=4,
            block_size=16,
            n_qubits=4,
            q_depth=2
        )
        
        model = QuantumRWKVModel(config)
        print(f"✅ QRWKV模型創建成功，參數數量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 測試前向傳播
        print("🔄 測試前向傳播...")
        for batch in train_loader:
            input_ids = batch['input_ids']
            print(f"輸入形狀: {input_ids.shape}")
            
            outputs = model(input_ids)
            print(f"輸出類型: {type(outputs)}")
            
            if isinstance(outputs, tuple):
                print(f"輸出元組長度: {len(outputs)}")
                print(f"第一個元素形狀: {outputs[0].shape}")
            else:
                print(f"輸出張量形狀: {outputs.shape}")
            
            break
        
        print("✅ QRWKV模型測試成功！")
        return True
        
    except Exception as e:
        print(f"❌ QRWKV模型測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generation_function():
    """測試生成函數"""
    print("\n🧪 測試生成函數...")
    
    try:
        from comprehensive_evaluation import generate_texts_for_evaluation
        
        # 創建一個簡單的測試模型
        class TestModel(torch.nn.Module):
            def __init__(self, vocab_size):
                super().__init__()
                self.embedding = torch.nn.Embedding(vocab_size, 64)
                self.linear = torch.nn.Linear(64, vocab_size)
            
            def forward(self, x):
                # 返回元組格式的輸出
                embeddings = self.embedding(x)
                logits = self.linear(embeddings)
                return (logits,)  # 返回元組
        
        # 創建簡單的tokenizer
        class TestTokenizer:
            def __init__(self, vocab_size):
                self.vocab_size = vocab_size
                self.token_to_id = lambda x: 0
                self.decode = lambda x: "test"
            
            def encode(self, text):
                return [0, 1, 2]
            
            def get_vocab_size(self):
                return self.vocab_size
        
        model = TestModel(100)
        tokenizer = TestTokenizer(100)
        device = 'cpu'
        
        # 測試生成
        prompts = ["the", "birds"]
        generated_texts = generate_texts_for_evaluation(model, tokenizer, prompts, device, max_length=5)
        
        print(f"✅ 生成函數測試成功！生成了 {len(generated_texts)} 個文本")
        return True
        
    except Exception as e:
        print(f"❌ 生成函數測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 開始QRWKV修復測試...")
    
    # 測試QRWKV模型
    model_test = test_qrwkv_model()
    
    # 測試生成函數
    generation_test = test_generation_function()
    
    if model_test and generation_test:
        print("\n🎉 所有測試通過！QRWKV模型修復成功！")
    else:
        print("\n❌ 部分測試失敗，需要進一步修復")
