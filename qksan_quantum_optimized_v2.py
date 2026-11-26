import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from typing import List, Tuple, Optional

class QuantumStateEncoder(nn.Module):
    """量子態編碼器：將文本嵌入編碼到量子態"""
    def __init__(self, embedding_dim, n_qubits):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_qubits = n_qubits
        
        # 編碼器網絡
        self.encoder_net = nn.Sequential(
            nn.Linear(embedding_dim, n_qubits * 2),
            nn.Tanh(),
            nn.Linear(n_qubits * 2, n_qubits)
        )
        
    def forward(self, embeddings):
        """將嵌入編碼為量子態參數"""
        batch_size, seq_len, _ = embeddings.shape
        encoded = self.encoder_net(embeddings)  # [batch, seq, n_qubits]
        return torch.tanh(encoded)  # 限制在[-1, 1]範圍

class QuantumSelfAttention(nn.Module):
    """量子自注意力機制"""
    def __init__(self, embedding_dim, n_qubits, n_heads=4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_qubits = n_qubits
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads
        
        # 量子設備
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # 注意力參數
        self.query_encoder = QuantumStateEncoder(embedding_dim, n_qubits)
        self.key_encoder = QuantumStateEncoder(embedding_dim, n_qubits)
        self.value_encoder = QuantumStateEncoder(embedding_dim, n_qubits)
        
        # 量子電路參數
        self.attention_weights = nn.Parameter(torch.randn(n_heads, n_qubits, 3))
        
        # 輸出投影
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        
    def quantum_attention_circuit(self, query, key, value, weights):
        """量子注意力電路"""
        # 編碼查詢、鍵、值到量子態
        for i in range(self.n_qubits):
            qml.RY(query[i], wires=i)
            qml.RZ(key[i], wires=i)
        
        # 變分注意力層
        for i in range(self.n_qubits):
            qml.Rot(*weights[i], wires=i)
        
        # 創建量子糾纏
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        
        # 測量並與值結合
        measurements = [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        return measurements
    
    def forward(self, x, mask=None):
        batch_size, seq_len, embedding_dim = x.shape
        
        # 編碼查詢、鍵、值
        queries = self.query_encoder(x)
        keys = self.key_encoder(x)
        values = self.value_encoder(x)
        
        # 重塑為多頭
        queries = queries.view(batch_size, seq_len, self.n_heads, self.n_qubits // self.n_heads)
        keys = keys.view(batch_size, seq_len, self.n_heads, self.n_qubits // self.n_heads)
        values = values.view(batch_size, seq_len, self.n_heads, self.n_qubits // self.n_heads)
        
        outputs = []
        
        for head in range(self.n_heads):
            head_outputs = []
            for seq_idx in range(seq_len):
                seq_outputs = []
                for batch_idx in range(batch_size):
                    # 運行量子注意力電路
                    qnode = qml.QNode(self.quantum_attention_circuit, self.dev, interface="torch")
                    
                    query = queries[batch_idx, seq_idx, head]
                    key = keys[batch_idx, seq_idx, head]
                    value = values[batch_idx, seq_idx, head]
                    weights = self.attention_weights[head]
                    
                    quantum_output = qnode(query, key, value, weights)
                    quantum_output = torch.stack(quantum_output)
                    
                    # 與值結合
                    attended_value = quantum_output * value
                    seq_outputs.append(attended_value)
                
                head_outputs.append(torch.stack(seq_outputs))
            
            outputs.append(torch.stack(head_outputs))
        
        # 合併多頭
        output = torch.stack(outputs, dim=-1)  # [batch, seq, n_qubits//n_heads, n_heads]
        output = output.view(batch_size, seq_len, embedding_dim)
        
        return self.output_proj(output)

class QuantumLanguageModel(nn.Module):
    """量子語言模型：專門針對文字生成優化"""
    def __init__(self, embedding_dim, n_qubits, vocab_size):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_qubits = n_qubits
        self.vocab_size = vocab_size
        
        # 量子設備
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # 語言模型參數
        self.language_weights = nn.Parameter(torch.randn(3, n_qubits, 3))  # 3層變分電路
        
        # 編碼和解碼
        self.language_encoder = nn.Linear(embedding_dim, n_qubits)
        self.language_decoder = nn.Linear(n_qubits, vocab_size)
        
    def quantum_language_circuit(self, inputs, weights):
        """量子語言建模電路"""
        # 編碼輸入
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)
        
        # 多層變分電路
        for layer in range(3):
            # 旋轉門
            for i in range(self.n_qubits):
                qml.Rot(*weights[layer, i], wires=i)
            
            # 糾纏層
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[self.n_qubits - 1, 0])
        
        # 測量
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def forward(self, x):
        batch_size, seq_len, embedding_dim = x.shape
        outputs = []
        
        for seq_idx in range(seq_len):
            seq_outputs = []
            for batch_idx in range(batch_size):
                # 編碼
                encoded = self.language_encoder(x[batch_idx, seq_idx])
                encoded = torch.tanh(encoded)
                
                # 量子語言建模
                qnode = qml.QNode(self.quantum_language_circuit, self.dev, interface="torch")
                quantum_output = qnode(encoded, self.language_weights)
                quantum_output = torch.stack(quantum_output)
                
                # 解碼為詞彙分布
                vocab_logits = self.language_decoder(quantum_output)
                seq_outputs.append(vocab_logits)
            
            outputs.append(torch.stack(seq_outputs))
        
        return torch.stack(outputs, dim=1)

class OptimizedQuantumQKSAN(nn.Module):
    """優化的量子QKSAN模型，專門針對文字生成"""
    def __init__(self, vocab_size, embedding_dim=128, n_qubits=6, device="cpu"):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_qubits = n_qubits
        self.device = device
        
        # 詞嵌入
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 位置編碼
        self.pos_encoding = nn.Parameter(torch.randn(1024, embedding_dim))
        
        # 量子自注意力
        self.quantum_attention = QuantumSelfAttention(embedding_dim, n_qubits, n_heads=8)
        
        # 量子語言模型
        self.quantum_language = QuantumLanguageModel(embedding_dim, n_qubits, vocab_size)
        
        # 層歸一化
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        
        # 前饋網絡
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
        
        # 最終輸出層
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # 詞嵌入 + 位置編碼
        embeddings = self.embedding(input_ids)
        embeddings = embeddings + self.pos_encoding[:seq_len].unsqueeze(0)
        embeddings = self.dropout(embeddings)
        
        # 量子自注意力
        attention_output = self.quantum_attention(embeddings, attention_mask)
        attention_output = self.layer_norm1(attention_output + embeddings)
        
        # 前饋網絡
        ffn_output = self.ffn(attention_output)
        ffn_output = self.layer_norm2(ffn_output + attention_output)
        
        # 量子語言建模
        language_output = self.quantum_language(ffn_output)
        
        # 最終輸出
        logits = self.output_layer(ffn_output) + language_output
        
        return logits
    
    def generate(self, input_ids, max_length=50, temperature=1.0, top_k=50, top_p=0.9):
        """量子文本生成"""
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # 獲取最後的序列
                current_input = generated[:, -self.pos_encoding.size(0):]
                
                # 前向傳播
                logits = self.forward(current_input)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-k 和 Top-p 採樣
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # 採樣下一個token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 添加到生成序列
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated

def create_optimized_quantum_qksan(vocab_size, embedding_dim=128, n_qubits=6, device="cpu"):
    """創建優化的量子QKSAN模型"""
    return OptimizedQuantumQKSAN(vocab_size, embedding_dim, n_qubits, device)

# 測試和演示函數
def test_optimized_quantum_model():
    """測試優化的量子模型"""
    print("🧪 測試優化的量子QKSAN模型...")
    
    # 創建模型
    model = create_optimized_quantum_qksan(
        vocab_size=1000,
        embedding_dim=128,
        n_qubits=6,
        device="cpu"
    )
    
    # 測試輸入
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # 前向傳播
    with torch.no_grad():
        output = model(input_ids)
    
    print(f"✅ 輸入形狀: {input_ids.shape}")
    print(f"✅ 輸出形狀: {output.shape}")
    print(f"✅ 模型參數數量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 測試生成
    print("\n🚀 測試量子文本生成...")
    start_tokens = torch.tensor([[1, 2, 3, 4, 5]])  # 開始token
    generated = model.generate(start_tokens, max_length=15, temperature=0.8)
    print(f"✅ 生成序列: {generated[0].tolist()}")
    
    return model

if __name__ == "__main__":
    test_optimized_quantum_model()
