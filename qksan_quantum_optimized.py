import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from typing import List, Tuple

class QuantumAttention(nn.Module):
    """量子注意力機制"""
    def __init__(self, embedding_dim, n_qubits, n_layers=2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # 創建量子設備
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # 量子電路參數
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3))  # 每層每量子比特3個旋轉參數
        
        # 編碼器：將經典數據編碼到量子態
        self.encoder = nn.Linear(embedding_dim, n_qubits)
        
        # 解碼器：從量子態提取信息
        self.decoder = nn.Linear(n_qubits, embedding_dim)
        
    def quantum_circuit(self, inputs, weights):
        """量子電路：包含編碼、變分層和測量"""
        # 編碼輸入數據到量子態
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)
            qml.RZ(inputs[i], wires=i)
        
        # 變分層
        for layer in range(self.n_layers):
            # 旋轉門
            for i in range(self.n_qubits):
                qml.Rot(*weights[layer, i], wires=i)
            
            # 糾纏層
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[self.n_qubits - 1, 0])
        
        # 測量所有量子比特
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def forward(self, x):
        batch_size, seq_len, embedding_dim = x.shape
        outputs = []
        
        for seq_idx in range(seq_len):
            seq_outputs = []
            for batch_idx in range(batch_size):
                # 編碼到量子態
                encoded = self.encoder(x[batch_idx, seq_idx])
                encoded = torch.tanh(encoded)  # 限制在[-1, 1]範圍
                
                # 運行量子電路
                qnode = qml.QNode(self.quantum_circuit, self.dev, interface="torch")
                quantum_output = qnode(encoded, self.weights)
                quantum_output = torch.stack(quantum_output)
                
                # 解碼量子輸出
                decoded = self.decoder(quantum_output)
                seq_outputs.append(decoded)
            
            outputs.append(torch.stack(seq_outputs))
        
        return torch.stack(outputs, dim=1)

class QuantumKernel(nn.Module):
    """量子核函數：用於序列建模"""
    def __init__(self, embedding_dim, n_qubits, kernel_size=3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_qubits = n_qubits
        self.kernel_size = kernel_size
        
        # 量子設備
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # 量子核參數
        self.kernel_weights = nn.Parameter(torch.randn(kernel_size, n_qubits, 3))
        
        # 編碼和解碼
        self.kernel_encoder = nn.Linear(embedding_dim * kernel_size, n_qubits)
        self.kernel_decoder = nn.Linear(n_qubits, embedding_dim)
        
    def quantum_kernel_circuit(self, inputs, weights):
        """量子核電路"""
        # 編碼
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)
        
        # 多層變分電路
        for layer in range(self.kernel_size):
            for i in range(self.n_qubits):
                qml.Rot(*weights[layer, i], wires=i)
            
            # 創建糾纏
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def forward(self, x):
        batch_size, seq_len, embedding_dim = x.shape
        outputs = []
        
        for i in range(seq_len - self.kernel_size + 1):
            batch_outputs = []
            for batch_idx in range(batch_size):
                # 提取kernel窗口
                kernel_input = x[batch_idx, i:i+self.kernel_size].flatten()
                
                # 編碼
                encoded = self.kernel_encoder(kernel_input)
                encoded = torch.tanh(encoded)
                
                # 量子核計算
                qnode = qml.QNode(self.quantum_kernel_circuit, self.dev, interface="torch")
                quantum_output = qnode(encoded, self.kernel_weights)
                quantum_output = torch.stack(quantum_output)
                
                # 解碼
                decoded = self.kernel_decoder(quantum_output)
                batch_outputs.append(decoded)
            
            outputs.append(torch.stack(batch_outputs))
        
        return torch.stack(outputs, dim=1)

class QuantumQKSANModel(nn.Module):
    """真正的量子QKSAN模型"""
    def __init__(self, vocab_size, embedding_dim=64, n_qubits=4, device="cpu"):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_qubits = n_qubits
        self.device = device
        
        # 詞嵌入
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 位置編碼
        self.pos_encoding = nn.Parameter(torch.randn(512, embedding_dim))
        
        # 量子注意力機制
        self.quantum_attention = QuantumAttention(embedding_dim, n_qubits)
        
        # 量子核函數
        self.quantum_kernel = QuantumKernel(embedding_dim, n_qubits)
        
        # 量子混合層
        self.quantum_mixer = nn.Linear(embedding_dim * 2, embedding_dim)
        
        # 輸出層
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
        
        # 層歸一化
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # 詞嵌入 + 位置編碼
        embeddings = self.embedding(input_ids)
        embeddings = embeddings + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # 量子注意力
        attention_output = self.quantum_attention(embeddings)
        attention_output = self.layer_norm1(attention_output + embeddings)
        
        # 量子核處理
        kernel_output = self.quantum_kernel(attention_output)
        
        # 填充kernel輸出以匹配原始序列長度
        if kernel_output.size(1) < seq_len:
            padding = torch.zeros(batch_size, seq_len - kernel_output.size(1), 
                                self.embedding_dim, device=kernel_output.device)
            kernel_output = torch.cat([kernel_output, padding], dim=1)
        
        # 混合量子輸出
        combined = torch.cat([attention_output, kernel_output], dim=-1)
        mixed_output = self.quantum_mixer(combined)
        mixed_output = self.layer_norm2(mixed_output + attention_output)
        
        # 輸出層
        logits = self.output_layer(mixed_output)
        
        return logits

def create_quantum_qksan_model(vocab_size, embedding_dim=64, n_qubits=4, device="cpu"):
    """創建真正的量子QKSAN模型"""
    return QuantumQKSANModel(vocab_size, embedding_dim, n_qubits, device)

# 測試函數
def test_quantum_model():
    """測試量子模型"""
    print("🧪 測試量子QKSAN模型...")
    
    # 創建模型
    model = create_quantum_qksan_model(
        vocab_size=1000,
        embedding_dim=64,
        n_qubits=4,
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
    
    return model

if __name__ == "__main__":
    test_quantum_model()
