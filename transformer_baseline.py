#!/usr/bin/env python3
"""
Transformer Decoder Baseline Model
作為量子模型的baseline比較
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    """位置編碼"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """多頭注意力機制"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """縮放點積注意力"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 線性變換
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 注意力計算
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 重塑輸出
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 最終線性變換
        output = self.w_o(attention_output)
        
        return output, attention_weights

class FeedForward(nn.Module):
    """前饋神經網絡"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerDecoderLayer(nn.Module):
    """Transformer解碼器層"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 自注意力
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前饋網絡
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class TransformerDecoder(nn.Module):
    """Transformer解碼器"""
    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 8, 
                 n_layers: int = 6, d_ff: int = 1024, max_len: int = 512, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # 詞嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # 解碼器層
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 輸出層
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """創建因果掩碼"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        
        # 嵌入和位置編碼
        embeddings = self.embedding(input_ids) * math.sqrt(self.d_model)
        embeddings = self.pos_encoding(embeddings.transpose(0, 1)).transpose(0, 1)
        embeddings = self.dropout(embeddings)
        
        # 創建因果掩碼
        causal_mask = self.create_causal_mask(seq_len).to(input_ids.device)
        
        # 通過解碼器層
        x = embeddings
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, causal_mask)
        
        # 輸出層
        logits = self.output_layer(x)
        
        return logits

class TransformerBaselineModel(nn.Module):
    """Transformer Baseline模型包裝器"""
    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 8,
                 n_layers: int = 6, d_ff: int = 1024, max_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout
        )
        
        self.vocab_size = vocab_size
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.decoder(input_ids)
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 50, 
                temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9) -> torch.Tensor:
        """生成文本"""
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # 獲取最後的序列
                current_input = generated[:, -input_ids.size(1):]
                
                # 前向傳播
                logits = self.forward(current_input)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-k 和 Top-p 採樣
                if top_k > 0:
                    # 確保top_k不超過詞彙大小
                    actual_top_k = min(top_k, next_token_logits.size(-1))
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, actual_top_k)
                    next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # 採樣下一個token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 添加到生成序列
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated

def create_transformer_baseline_model(vocab_size: int, **kwargs) -> TransformerBaselineModel:
    """創建Transformer baseline模型"""
    return TransformerBaselineModel(
        vocab_size=vocab_size,
        d_model=kwargs.get('d_model', 256),
        n_heads=kwargs.get('n_heads', 8),
        n_layers=kwargs.get('n_layers', 6),
        d_ff=kwargs.get('d_ff', 1024),
        max_len=kwargs.get('max_len', 512),
        dropout=kwargs.get('dropout', 0.1)
    )

if __name__ == "__main__":
    # 測試模型
    vocab_size = 1000
    model = create_transformer_baseline_model(vocab_size, d_model=128, n_layers=2)
    
    # 測試前向傳播
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    logits = model(input_ids)
    print(f"輸入形狀: {input_ids.shape}")
    print(f"輸出形狀: {logits.shape}")
    
    # 測試生成
    generated = model.generate(input_ids, max_length=15)
    print(f"生成形狀: {generated.shape}")
    print("測試完成！")
