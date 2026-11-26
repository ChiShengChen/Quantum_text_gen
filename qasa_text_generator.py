import torch
import torch.nn as nn
import torch.nn.functional as F
from qasa_model import QuantumLayer

class QASATextGenerator(nn.Module):
    """基於QASA量子模型的文字生成器"""
    def __init__(self, vocab_size, hidden_dim=128, seq_len=32, n_qubits=8, device='cpu'):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.device = device
        
        # 詞嵌入
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # 位置編碼
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, hidden_dim))
        
        # QASA量子層
        self.quantum_layer = QuantumLayer(hidden_dim, hidden_dim)
        
        # 輸出層
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
        # 層歸一化
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, timestep=None):
        batch_size, seq_len = input_ids.shape
        
        # 詞嵌入 + 位置編碼
        embeddings = self.embedding(input_ids)
        embeddings = embeddings + self.pos_encoding[:seq_len].unsqueeze(0)
        embeddings = self.dropout(embeddings)
        
        # 序列處理：對每個位置應用量子層
        outputs = []
        for i in range(seq_len):
            # 獲取當前位置的嵌入
            current_embedding = embeddings[:, i, :]  # [batch_size, hidden_dim]
            
            # 應用QASA量子層
            quantum_output = self.quantum_layer(current_embedding, timestep)
            
            # 層歸一化
            quantum_output = self.layer_norm(quantum_output)
            
            outputs.append(quantum_output)
        
        # 堆疊所有位置的輸出
        sequence_output = torch.stack(outputs, dim=1)  # [batch_size, seq_len, hidden_dim]
        
        # 輸出層
        logits = self.output_layer(sequence_output)
        
        return logits
    
    def generate(self, input_ids, max_length=50, temperature=1.0, top_k=50, top_p=0.9):
        """量子文本生成"""
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # 獲取最後的序列
                current_input = generated[:, -self.seq_len:] if generated.size(1) > self.seq_len else generated
                
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

def create_qasa_text_generator(vocab_size, hidden_dim=128, seq_len=32, n_qubits=8, device='cpu'):
    """創建QASA文字生成器"""
    return QASATextGenerator(vocab_size, hidden_dim, seq_len, n_qubits, device)

# 測試函數
def test_qasa_text_generator():
    """測試QASA文字生成器"""
    print("🧪 測試QASA文字生成器...")
    
    # 創建模型
    model = create_qasa_text_generator(
        vocab_size=1000,
        hidden_dim=128,
        seq_len=32,
        n_qubits=8,
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
    
    # 檢查量子特性
    print("\n🔬 QASA量子特性分析:")
    print(f"✅ 使用QASA量子層: {hasattr(model, 'quantum_layer')}")
    print(f"✅ 量子層類型: {type(model.quantum_layer)}")
    print(f"✅ 量子層有量子電路: {hasattr(model.quantum_layer, 'qlayer')}")
    
    # 測試生成
    print("\n🚀 測試QASA量子文本生成...")
    start_tokens = torch.tensor([[1, 2, 3, 4, 5]])  # 開始token
    generated = model.generate(start_tokens, max_length=15, temperature=0.8)
    print(f"✅ 生成序列: {generated[0].tolist()}")
    
    return model

if __name__ == "__main__":
    test_qasa_text_generator()
