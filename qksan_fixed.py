import torch
import torch.nn as nn
import pennylane as qml

def create_fixed_qksan_model(vocab_size, embedding_dim=64, n_qubits=4, device="cpu"):
    """創建QKSAN模型"""
    class QKSANModel(nn.Module):
        def __init__(self, vocab_size, embedding_dim, n_qubits, device):
            super().__init__()
            self.vocab_size = vocab_size
            self.embedding_dim = embedding_dim
            self.n_qubits = n_qubits
            self.device = device
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.quantum_layer = nn.Linear(embedding_dim, embedding_dim)
            self.output_layer = nn.Linear(embedding_dim, vocab_size)
            self.pos_encoding = nn.Parameter(torch.randn(32, embedding_dim))
        def forward(self, input_ids):
            batch_size, seq_len = input_ids.shape
            embeddings = self.embedding(input_ids)
            embeddings = embeddings + self.pos_encoding[:seq_len].unsqueeze(0)
            quantum_output = torch.tanh(self.quantum_layer(embeddings))
            logits = self.output_layer(quantum_output)
            return logits
    return QKSANModel(vocab_size, embedding_dim, n_qubits, device)
