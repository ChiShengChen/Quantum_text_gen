# models/transformer_text_models.py
"""
Transformer Model Adapters for Text Generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import sys
import os

sys.path.append('..')
from transformer_model import TransformerModel, TransformerConfig

class TransformerTextGenerator:
    """Transformer文本生成器適配器"""
    
    def __init__(self, model: TransformerModel, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def generate(self, prompt: str, max_length: int = 50, temperature: float = 1.0, 
                top_k: int = 50, top_p: float = 0.9, do_sample: bool = True) -> str:
        """
        生成文本
        
        Args:
            prompt: 輸入提示
            max_length: 最大生成長度
            temperature: 溫度參數
            top_k: Top-K採樣
            top_p: Top-P採樣
            do_sample: 是否使用採樣
        
        Returns:
            生成的文本
        """
        # 編碼輸入
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        # 計算需要生成的新token數量
        max_new_tokens = max_length - len(input_ids)
        if max_new_tokens <= 0:
            return prompt
        
        # 生成文本
        with torch.no_grad():
            if do_sample:
                # 使用採樣
                generated_ids = self.model.generate(
                    input_tensor,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k
                )
            else:
                # 使用貪心解碼
                generated_ids = input_tensor
                for _ in range(max_new_tokens):
                    logits, _ = self.model(generated_ids)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)
        
        # 解碼輸出
        generated_text = self.tokenizer.decode(generated_ids[0].tolist())
        return generated_text

def create_transformer_model(vocab_size: int, device: str = 'cpu', **kwargs) -> TransformerModel:
    """
    創建Transformer模型
    
    Args:
        vocab_size: 詞彙大小
        device: 設備
        **kwargs: 其他參數
    
    Returns:
        Transformer模型
    """
    config = TransformerConfig(
        vocab_size=vocab_size,
        n_embd=kwargs.get('n_embd', 256),  # 較小的嵌入維度以匹配其他模型
        n_head=kwargs.get('n_head', 8),
        n_layer=kwargs.get('n_layer', 4),   # 較少的層數以匹配其他模型
        block_size=kwargs.get('block_size', 256),
        n_intermediate=kwargs.get('n_intermediate', 1024),
        dropout=kwargs.get('dropout', 0.1)
    )
    
    model = TransformerModel(config)
    return model

def create_transformer_generator(model: TransformerModel, tokenizer, device: str = 'cpu') -> TransformerTextGenerator:
    """
    創建Transformer文本生成器
    
    Args:
        model: Transformer模型
        tokenizer: 分詞器
        device: 設備
    
    Returns:
        Transformer文本生成器
    """
    return TransformerTextGenerator(model, tokenizer, device)
