#!/usr/bin/env python3
"""
改進的量子文字生成訓練腳本
包含更好的生成策略、訓練方法和評估指標
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from datetime import datetime
import json
import random

# 設置隨機種子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def improved_text_generation(model, tokenizer, prompts, device, max_length=20):
    """改進的文本生成函數"""
    model.eval()
    generated_texts = []
    
    with torch.no_grad():
        for prompt in prompts:
            # 編碼提示詞
            input_ids = tokenizer.encode(prompt)
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
            
            generated = []
            current_input = input_tensor.clone()
            
            for _ in range(max_length):
                outputs = model(current_input)
                logits = outputs[:, -1, :]
                
                # 使用溫度採樣
                temperature = 0.7
                logits = logits / temperature
                
                # Top-k 採樣
                top_k = 15
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                
                # 計算概率
                probs = torch.softmax(top_k_logits, dim=-1)
                
                # 採樣
                next_token_idx = torch.multinomial(probs, 1)
                next_token_id = top_k_indices[0, next_token_idx.item()]
                
                # 檢查結束條件
                if (next_token_id.item() == tokenizer.token_to_id('<PAD>') or 
                    len(generated) >= max_length - 1):
                    break
                
                generated.append(next_token_id.item())
                current_input = torch.cat([current_input, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)
            
            # 解碼
            generated_text = tokenizer.decode(generated)
            full_text = prompt + " " + generated_text if generated_text else prompt
            generated_texts.append(full_text)
    
    return generated_texts

def improved_training_loop(model, train_loader, val_loader, epochs, device, learning_rate=0.001):
    """改進的訓練循環"""
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    for epoch in range(epochs):
        # 訓練階段
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            
            # 重塑輸出和目標
            outputs = outputs.view(-1, outputs.size(-1))
            targets = target_ids.view(-1)
            
            loss = criterion(outputs, targets)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 驗證階段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                target_ids = batch['target_ids'].to(device)
                
                outputs = model(input_ids)
                outputs = outputs.view(-1, outputs.size(-1))
                targets = target_ids.view(-1)
                
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # 學習率調度
        scheduler.step(val_loss)
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    return train_losses, val_losses, best_val_loss

def main():
    """主函數"""
    set_seed(42)
    
    # 設置設備
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用設備: {device}")
    
    # 這裡需要導入你的模型和數據加載器
    # 示例配置
    config = {
        'epochs': 30,
        'batch_size': 16,
        'learning_rate': 0.001,
        'max_length': 32,
        'vocab_size': 500,
        'max_samples': 1000
    }
    
    print("改進的訓練配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 這裡添加你的模型訓練代碼
    print("\n請將此腳本與你的模型和數據加載器結合使用")

if __name__ == "__main__":
    main()
