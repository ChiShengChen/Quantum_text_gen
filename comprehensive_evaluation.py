#!/usr/bin/env python3
"""
Comprehensive Evaluation System
全面的評估系統，包含多種指標來衡量文本生成模型的性能
"""

import os
import sys
import torch
import json
import time
import numpy as np
from collections import Counter
import re
from datetime import datetime

sys.path.append('.')

from data.datasets import create_dataloader
from qksan_fixed import create_fixed_qksan_model
from qrwkv_model import QuantumRWKVModel, ModelConfig

def create_model(model_type, vocab_size, device, **kwargs):
    """創建模型"""
    if model_type == 'qksan':
        return create_fixed_qksan_model(
            vocab_size=vocab_size,
            embedding_dim=kwargs.get('embedding_dim', 64),
            n_qubits=kwargs.get('n_qubits', 4),
            device=device
        )
    elif model_type == 'qrwkv':
        config = ModelConfig(
            vocab_size=vocab_size,
            n_embd=kwargs.get('embedding_dim', 64),
            n_head=8,
            n_layer=kwargs.get('n_layers', 2),
            n_qubits=kwargs.get('n_qubits', 2),
            block_size=256
        )
        return QuantumRWKVModel(config)
    elif model_type == 'qasa':
        # 創建QASA文本生成模型
        class QASATextGenerator(torch.nn.Module):
            def __init__(self, vocab_size, hidden_dim=64, seq_len=32, n_qubits=4, device='cpu'):
                super().__init__()
                self.vocab_size = vocab_size
                self.hidden_dim = hidden_dim
                self.seq_len = seq_len
                self.device = device
                
                # 嵌入層
                self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)
                
                # 量子層
                self.quantum_layer = torch.nn.Linear(hidden_dim, hidden_dim)
                
                # 輸出層
                self.output_layer = torch.nn.Linear(hidden_dim, vocab_size)
                
                # 位置編碼
                self.pos_encoding = torch.nn.Parameter(torch.randn(seq_len, hidden_dim))
                
            def forward(self, input_ids):
                batch_size, seq_len = input_ids.shape
                
                # 嵌入
                embeddings = self.embedding(input_ids)  # [B, S, H]
                
                # 添加位置編碼
                embeddings = embeddings + self.pos_encoding[:seq_len].unsqueeze(0)
                
                # 量子處理（簡化版）
                quantum_output = torch.tanh(self.quantum_layer(embeddings))
                
                # 輸出
                logits = self.output_layer(quantum_output)  # [B, S, V]
                
                return logits
        
        return QASATextGenerator(
            vocab_size=vocab_size,
            hidden_dim=kwargs.get('embedding_dim', 64),
            seq_len=kwargs.get('max_length', 32),
            n_qubits=kwargs.get('n_qubits', 4),
            device=device
        )
    else:
        raise ValueError(f"不支持的模型類型: {model_type}")

def calculate_perplexity(model, data_loader, device):
    """計算困惑度"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            try:
                outputs = model(input_ids)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                logits = outputs.view(-1, outputs.size(-1))
                targets = target_ids.view(-1)
                
                loss = torch.nn.functional.cross_entropy(logits, targets, reduction='sum')
                total_loss += loss.item()
                total_tokens += targets.numel()
            except Exception as e:
                print(f"計算困惑度時跳過batch: {e}")
                continue
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity

def calculate_bleu_score(references, candidates, n_grams=[1, 2, 3, 4]):
    """計算BLEU分數"""
    def get_ngrams(text, n):
        words = text.split()
        return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    
    def calculate_precision(reference_ngrams, candidate_ngrams):
        if not candidate_ngrams:
            return 0.0
        matches = sum(1 for ngram in candidate_ngrams if ngram in reference_ngrams)
        return matches / len(candidate_ngrams)
    
    bleu_scores = {}
    
    for n in n_grams:
        total_precision = 0.0
        total_candidates = 0
        
        for ref, cand in zip(references, candidates):
            ref_ngrams = get_ngrams(ref, n)
            cand_ngrams = get_ngrams(cand, n)
            
            if cand_ngrams:
                precision = calculate_precision(ref_ngrams, cand_ngrams)
                total_precision += precision
                total_candidates += 1
        
        if total_candidates > 0:
            bleu_scores[f'BLEU-{n}'] = total_precision / total_candidates
        else:
            bleu_scores[f'BLEU-{n}'] = 0.0
    
    return bleu_scores

def calculate_diversity_metrics(texts):
    """計算多樣性指標"""
    def get_ngrams(text, n):
        words = text.split()
        return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    
    # Distinct-1 and Distinct-2
    unigrams = []
    bigrams = []
    
    for text in texts:
        unigrams.extend(get_ngrams(text, 1))
        bigrams.extend(get_ngrams(text, 2))
    
    distinct_1 = len(set(unigrams)) / len(unigrams) if unigrams else 0
    distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0
    
    # Entropy
    word_counts = Counter()
    total_words = 0
    
    for text in texts:
        words = text.split()
        word_counts.update(words)
        total_words += len(words)
    
    if total_words > 0:
        entropy = -sum((count/total_words) * np.log2(count/total_words) 
                      for count in word_counts.values())
    else:
        entropy = 0
    
    return {
        'distinct_1': distinct_1,
        'distinct_2': distinct_2,
        'entropy': entropy,
        'vocabulary_size': len(word_counts)
    }

def calculate_repetition_rate(texts):
    """計算重複率"""
    def get_repetition_score(text):
        words = text.split()
        if len(words) < 2:
            return 0.0
        
        consecutive_repeats = 0
        total_pairs = len(words) - 1
        
        for i in range(len(words) - 1):
            if words[i] == words[i + 1]:
                consecutive_repeats += 1
        
        return consecutive_repeats / total_pairs if total_pairs > 0 else 0.0
    
    repetition_scores = [get_repetition_score(text) for text in texts]
    return np.mean(repetition_scores)

def calculate_fluency_metrics(texts):
    """計算流暢度指標"""
    def get_sentence_lengths(text):
        # 簡單的句子分割
        sentences = re.split(r'[.!?]+', text)
        return [len(s.split()) for s in sentences if s.strip()]
    
    all_lengths = []
    for text in texts:
        all_lengths.extend(get_sentence_lengths(text))
    
    if not all_lengths:
        return {
            'avg_sentence_length': 0,
            'sentence_length_std': 0,
            'sentence_length_variation': 0
        }
    
    avg_length = np.mean(all_lengths)
    std_length = np.std(all_lengths)
    variation = std_length / avg_length if avg_length > 0 else 0
    
    return {
        'avg_sentence_length': avg_length,
        'sentence_length_std': std_length,
        'sentence_length_variation': variation
    }

def generate_texts_for_evaluation(model, tokenizer, prompts, device, max_length=20):
    """生成文本用於評估"""
    model.eval()
    generated_texts = []
    
    with torch.no_grad():
        for prompt in prompts:
            # 編碼提示詞
            input_ids = tokenizer.encode(prompt)
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
            
            # 使用更好的生成策略
            generated = []
            current_input = input_tensor.clone()
            
            for _ in range(max_length):
                # 獲取模型輸出
                outputs = model(current_input)
                logits = outputs[:, -1, :]  # 只取最後一個位置的logits
                
                # 使用溫度採樣而不是貪心解碼
                temperature = 0.8
                logits = logits / temperature
                
                # 應用top-k採樣
                top_k = 10
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                
                # 計算概率
                probs = torch.softmax(top_k_logits, dim=-1)
                
                # 採樣下一個詞
                next_token_idx = torch.multinomial(probs, 1)
                next_token_id = top_k_indices[0, next_token_idx.item()]
                
                # 檢查是否為結束標記
                if next_token_id.item() == tokenizer.token_to_id('<PAD>') or len(generated) >= max_length - 1:
                    break
                
                generated.append(next_token_id.item())
                current_input = torch.cat([current_input, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)
            
            # 解碼生成的文本
            generated_text = tokenizer.decode(generated)
            full_text = prompt + " " + generated_text if generated_text else prompt
            generated_texts.append(full_text)
    
    return generated_texts

def get_reference_texts(dataset_name, num_samples=5):
    """獲取參考文本"""
    # 簡單的參考文本，實際應用中可以從測試集中獲取
    reference_texts = {
        'simple_sentences': [
            "the cat sat on the mat",
            "birds fly in the sky",
            "water flows down the river",
            "spring brings new life",
            "stars shine in the night"
        ],
        'short_stories': [
            "the old man whispered secrets to the wind",
            "she danced through the moonlight",
            "time stood still in that moment",
            "memories faded like autumn leaves",
            "hope bloomed in the darkest night"
        ],
        'quantum_phrases': [
            "quantum superposition allows multiple states",
            "entanglement connects particles instantly",
            "wave function collapse upon measurement",
            "quantum coherence maintains phase relationships",
            "quantum tunneling through energy barriers"
        ],
        'haiku': [
            "cherry blossoms fall",
            "mountain mist clears slowly",
            "autumn leaves dance",
            "winter snow blankets",
            "spring rain awakens"
        ],
        'proverbs': [
            "actions speak louder than words",
            "time heals all wounds",
            "knowledge is power",
            "practice makes perfect",
            "honesty is the best policy"
        ]
    }
    
    return reference_texts.get(dataset_name, reference_texts['simple_sentences'])[:num_samples]

def evaluate_model_comprehensive(model, tokenizer, test_loader, device, dataset_name):
    """全面評估模型"""
    print("📊 開始全面評估...")
    
    # 1. 計算困惑度
    perplexity = calculate_perplexity(model, test_loader, device)
    print(f"  📈 困惑度: {perplexity:.4f}")
    
    # 2. 生成文本
    test_prompts = ["the", "birds", "water", "spring", "stars"]
    generated_texts = generate_texts_for_evaluation(model, tokenizer, test_prompts, device)
    
    print(f"  📝 生成文本示例:")
    for i, text in enumerate(generated_texts):
        print(f"    {i+1}. {text}")
    
    # 3. 獲取參考文本
    reference_texts = get_reference_texts(dataset_name, len(generated_texts))
    
    # 4. 計算BLEU分數
    bleu_scores = calculate_bleu_score(reference_texts, generated_texts)
    print(f"  🎯 BLEU分數:")
    for metric, score in bleu_scores.items():
        print(f"    {metric}: {score:.4f}")
    
    # 5. 計算多樣性指標
    diversity_metrics = calculate_diversity_metrics(generated_texts)
    print(f"  🌈 多樣性指標:")
    print(f"    Distinct-1: {diversity_metrics['distinct_1']:.4f}")
    print(f"    Distinct-2: {diversity_metrics['distinct_2']:.4f}")
    print(f"    Entropy: {diversity_metrics['entropy']:.4f}")
    print(f"    Vocabulary Size: {diversity_metrics['vocabulary_size']}")
    
    # 6. 計算重複率
    repetition_rate = calculate_repetition_rate(generated_texts)
    print(f"  🔄 重複率: {repetition_rate:.4f}")
    
    # 7. 計算流暢度指標
    fluency_metrics = calculate_fluency_metrics(generated_texts)
    print(f"  📖 流暢度指標:")
    print(f"    平均句子長度: {fluency_metrics['avg_sentence_length']:.2f}")
    print(f"    句子長度標準差: {fluency_metrics['sentence_length_std']:.2f}")
    print(f"    句子長度變異係數: {fluency_metrics['sentence_length_variation']:.4f}")
    
    return {
        'perplexity': perplexity,
        'generated_texts': generated_texts,
        'reference_texts': reference_texts,
        'bleu_scores': bleu_scores,
        'diversity_metrics': diversity_metrics,
        'repetition_rate': repetition_rate,
        'fluency_metrics': fluency_metrics
    }

def run_comprehensive_experiment(model_type, dataset_name, device, epochs=8, batch_size=16, max_length=32, vocab_size=500, max_samples=200, learning_rate=0.001):
    """運行全面評估實驗"""
    print(f"\n{'='*80}")
    print(f"🎯 全面評估實驗: {model_type.upper()} on {dataset_name}")
    print(f"{'='*80}")
    
    # 創建數據加載器
    print("📊 創建數據加載器...")
    train_loader, tokenizer = create_dataloader(
        dataset_name=dataset_name,
        split='train',
        batch_size=batch_size,
        max_length=max_length,
        vocab_size=vocab_size,
        max_samples=max_samples,
        shuffle=True
    )
    
    val_loader, _ = create_dataloader(
        dataset_name=dataset_name,
        split='val',
        batch_size=batch_size,
        max_length=max_length,
        vocab_size=vocab_size,
        max_samples=max_samples // 4,
        shuffle=False
    )
    
    test_loader, _ = create_dataloader(
        dataset_name=dataset_name,
        split='test',
        batch_size=batch_size,
        max_length=max_length,
        vocab_size=vocab_size,
        max_samples=max_samples // 4,
        shuffle=False
    )
    
    print(f"✅ 數據加載器創建成功")
    print(f"📝 詞彙大小: {tokenizer.get_vocab_size()}")
    
    # 創建模型
    print(f"\n🤖 創建{model_type.upper()}模型...")
    vocab_size = tokenizer.get_vocab_size()
    model = create_model(model_type, vocab_size, device)
    
    print(f"✅ 模型創建成功")
    print(f"📊 模型參數數量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 訓練模型（簡化版）
    print(f"\n🎯 開始訓練...")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            logits = outputs.view(-1, outputs.size(-1))
            targets = target_ids.view(-1)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                target_ids = batch['target_ids'].to(device)
                
                outputs = model(input_ids)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                logits = outputs.view(-1, outputs.size(-1))
                targets = target_ids.view(-1)
                loss = criterion(logits, targets)
                val_loss += loss.item()
                val_batches += 1
        
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    print(f"✅ 訓練完成! 最佳驗證損失: {best_val_loss:.4f}")
    
    # 全面評估
    evaluation_results = evaluate_model_comprehensive(model, tokenizer, test_loader, device, dataset_name)
    
    # 保存結果
    print(f"\n💾 保存評估結果...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"./experiments/comprehensive/{model_type}/{dataset_name}_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    
    result = {
        'model_type': model_type,
        'dataset_name': dataset_name,
        'training_results': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1]
        },
        'evaluation_results': evaluation_results,
        'vocab_size': vocab_size,
        'model_dir': result_dir
    }
    
    # 保存詳細結果
    with open(os.path.join(result_dir, 'comprehensive_results.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    # 保存生成文本對比
    with open(os.path.join(result_dir, 'text_comparison.txt'), 'w') as f:
        f.write(f"Text Generation Comparison for {model_type.upper()} on {dataset_name}\n")
        f.write("=" * 60 + "\n\n")
        
        for i, (ref, gen) in enumerate(zip(evaluation_results['reference_texts'], evaluation_results['generated_texts'])):
            f.write(f"Example {i+1}:\n")
            f.write(f"Reference: {ref}\n")
            f.write(f"Generated: {gen}\n\n")
    
    print(f"✅ 全面評估完成: {model_type.upper()} on {dataset_name}")
    print(f"📁 結果保存到: {result_dir}")
    
    return result

def main():
    """主函數"""
    print("🚀 量子文字生成模型全面評估系統")
    print("=" * 80)
    
    # 設置設備
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  使用設備: {device}")
    
    # 實驗配置
    datasets = ['simple_sentences', 'short_stories', 'quantum_phrases', 'haiku', 'proverbs']
    models = ['qksan', 'qrwkv', 'qasa']  # 現在包含QASA
    
    # 從環境變量讀取配置
    epochs = int(os.environ.get('EPOCHS', 8))
    batch_size = int(os.environ.get('BATCH_SIZE', 16))
    max_length = int(os.environ.get('MAX_LENGTH', 32))
    vocab_size = int(os.environ.get('VOCAB_SIZE', 500))
    max_samples = int(os.environ.get('MAX_SAMPLES', 1000))
    learning_rate = float(os.environ.get('LEARNING_RATE', 0.001))
    
    print(f"📊 數據集: {datasets}")
    print(f"🤖 模型: {models}")
    print(f"🔄 訓練輪數: {epochs} epochs")
    print(f"📈 評估指標: 困惑度、BLEU、多樣性、重複率、流暢度")
    
    # 運行實驗
    all_results = {}
    
    for model_type in models:
        for dataset_name in datasets:
            try:
                result = run_comprehensive_experiment(model_type, dataset_name, device, epochs, batch_size, max_length, vocab_size, max_samples, learning_rate)
                experiment_key = f"{model_type}_{dataset_name}"
                all_results[experiment_key] = result
            except Exception as e:
                print(f"❌ 實驗失敗 {model_type} on {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 生成綜合報告
    print("\n📊 生成綜合評估報告...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = f"./experiments/comprehensive/comprehensive_report_{timestamp}"
    os.makedirs(report_dir, exist_ok=True)
    
    # 保存所有結果
    with open(os.path.join(report_dir, 'all_comprehensive_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # 生成報告
    report = f"""
# 量子文字生成模型全面評估報告

## 實驗概覽
- **實驗時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **訓練輪數**: {epochs} epochs
- **數據集數量**: {len(datasets)}
- **模型數量**: {len(models)}個
- **評估指標**: 困惑度、BLEU-1/2/3/4、多樣性(Distinct-1/2)、重複率、流暢度

## 評估指標說明

### 1. 困惑度 (Perplexity)
- 衡量語言模型預測下一個詞的能力
- 越低越好

### 2. BLEU分數 (BLEU-1/2/3/4)
- 衡量生成文本與參考文本的n-gram重疊度
- 越高越好

### 3. 多樣性指標
- **Distinct-1/2**: 衡量詞彙使用的多樣性
- **Entropy**: 詞彙分布的熵
- **Vocabulary Size**: 使用的詞彙大小

### 4. 重複率 (Repetition Rate)
- 衡量文本中連續重複詞的比例
- 越低越好

### 5. 流暢度指標
- **平均句子長度**: 句子的平均詞數
- **句子長度標準差**: 句子長度的變化
- **句子長度變異係數**: 標準差/平均值

## 實驗結果

### 按模型類型比較
"""
    
    for model in models:
        model_data = [v for k, v in all_results.items() if v['model_type'] == model]
        if model_data:
            report += f"\n#### {model.upper()} 模型\n"
            report += "| 數據集 | 困惑度 | BLEU-1 | BLEU-2 | Distinct-1 | Distinct-2 | 重複率 |\n"
            report += "|--------|--------|--------|--------|------------|------------|--------|\n"
            
            for data in model_data:
                eval_results = data['evaluation_results']
                report += f"| {data['dataset_name']} | {eval_results['perplexity']:.2f} | {eval_results['bleu_scores']['BLEU-1']:.3f} | {eval_results['bleu_scores']['BLEU-2']:.3f} | {eval_results['diversity_metrics']['distinct_1']:.3f} | {eval_results['diversity_metrics']['distinct_2']:.3f} | {eval_results['repetition_rate']:.3f} |\n"
    
    # 性能總結
    report += "\n## 性能總結\n\n"
    
    for model in models:
        model_data = [v for k, v in all_results.items() if v['model_type'] == model]
        if model_data:
            avg_perplexity = np.mean([d['evaluation_results']['perplexity'] for d in model_data if d['evaluation_results']['perplexity'] != float('inf')])
            avg_bleu1 = np.mean([d['evaluation_results']['bleu_scores']['BLEU-1'] for d in model_data])
            avg_distinct1 = np.mean([d['evaluation_results']['diversity_metrics']['distinct_1'] for d in model_data])
            avg_repetition = np.mean([d['evaluation_results']['repetition_rate'] for d in model_data])
            
            report += f"- **{model.upper()}**: 平均困惑度={avg_perplexity:.2f}, 平均BLEU-1={avg_bleu1:.3f}, 平均Distinct-1={avg_distinct1:.3f}, 平均重複率={avg_repetition:.3f}\n"
    
    with open(os.path.join(report_dir, 'comprehensive_report.md'), 'w') as f:
        f.write(report)
    
    print(f"📄 綜合報告已保存到: {report_dir}")
    print(f"\n{'='*80}")
    print("🎉 全面評估完成!")
    print(f"📊 共完成了 {len(all_results)} 個實驗")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
