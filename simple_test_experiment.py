#!/usr/bin/env python3
import torch
import sys
sys.path.append(".")

from data.datasets import create_dataloader
from qrwkv_model import QuantumRWKVModel, ModelConfig

print("🚀 簡單實驗開始...")

# 創建數據加載器
train_loader, tokenizer = create_dataloader(
    dataset_name="simple_sentences",
    split="train", 
    batch_size=4,
    max_length=8,
    vocab_size=100,
    max_samples=10,
    shuffle=True
)

print(f"✅ 數據加載成功，詞彙大小: {tokenizer.get_vocab_size()}")

# 創建模型
config = ModelConfig(
    vocab_size=tokenizer.get_vocab_size(),
    n_embd=32,
    n_head=4,
    n_layer=1,
    n_qubits=2,
    block_size=128
)
model = QuantumRWKVModel(config)

print(f"✅ 模型創建成功，參數數量: {sum(p.numel() for p in model.parameters()):,}")

# 簡單訓練
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

print("🎯 開始訓練...")
for epoch in range(3):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"]
        target_ids = batch["target_ids"]
        
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
    
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

print("🎉 簡單實驗完成！")

