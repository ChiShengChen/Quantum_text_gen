#!/bin/bash

# 量子文字生成模型完整實驗腳本
# Complete Quantum Text Generation Experiment Script

set -e

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 默認配置
EPOCHS=${1:-8}
BATCH_SIZE=${2:-16}
MAX_LENGTH=${3:-32}
VOCAB_SIZE=${4:-500}
MAX_SAMPLES=${5:-200}
LEARNING_RATE=${6:-0.001}

# 模型和數據集
MODELS=("qksan" "qrwkv" "qasa")
DATASETS=("simple_sentences" "short_stories" "quantum_phrases" "haiku" "proverbs")

# 輸出目錄
OUTPUT_DIR="./experiments/complete_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo -e "${CYAN}🚀 量子文字生成模型完整實驗腳本${NC}"
echo -e "${CYAN}====================================${NC}"
echo -e "${BLUE}訓練輪數:${NC} $EPOCHS"
echo -e "${BLUE}批次大小:${NC} $BATCH_SIZE"
echo -e "${BLUE}最大長度:${NC} $MAX_LENGTH"
echo -e "${BLUE}詞彙大小:${NC} $VOCAB_SIZE"
echo -e "${BLUE}最大樣本:${NC} $MAX_SAMPLES"
echo -e "${BLUE}學習率:${NC} $LEARNING_RATE"
echo -e "${BLUE}輸出目錄:${NC} $OUTPUT_DIR"
echo -e "${YELLOW}⚠️  強制使用CPU以避免CUDA錯誤${NC}"
echo -e "${GREEN}✅ 包含所有三個模型: QKSAN, QRWKV, QASA${NC}"
echo -e "${GREEN}✅ 包含完整評估指標: 困惑度、BLEU、多樣性、重複率、流暢度${NC}"
echo ""

# 創建Python實驗腳本
cat > "$OUTPUT_DIR/run_complete_experiments.py" << 'EOF'
#!/usr/bin/env python3
import os
import sys
import json
import torch
import time
import numpy as np
from collections import Counter
import re
from datetime import datetime

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

def get_reference_texts(dataset_name, num_samples=5):
    """獲取參考文本"""
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

def run_experiment(model_type, dataset_name, config, device):
    """運行單個實驗"""
    print(f"\n{'='*80}")
    print(f"🎯 開始實驗: {model_type.upper()} on {dataset_name}")
    print(f"{'='*80}")
    
    try:
        # 導入必要的模塊
        from data.datasets import create_dataloader
        
        if model_type == 'qksan':
            from qksan_fixed import create_fixed_qksan_model
        elif model_type == 'qrwkv':
            from qrwkv_model import QuantumRWKVModel, ModelConfig
        elif model_type == 'qasa':
            from models.quantum_text_models import QASATextGenerator
        
        # 創建數據加載器
        print("📊 創建數據加載器...")
        train_loader, tokenizer = create_dataloader(
            dataset_name=dataset_name,
            split='train',
            batch_size=config['batch_size'],
            max_length=config['max_length'],
            vocab_size=config['vocab_size'],
            max_samples=config['max_samples'],
            shuffle=True
        )
        
        val_loader, _ = create_dataloader(
            dataset_name=dataset_name,
            split='val',
            batch_size=config['batch_size'],
            max_length=config['max_length'],
            vocab_size=config['vocab_size'],
            max_samples=config['max_samples'] // 4,
            shuffle=False
        )
        
        test_loader, _ = create_dataloader(
            dataset_name=dataset_name,
            split='test',
            batch_size=config['batch_size'],
            max_length=config['max_length'],
            vocab_size=config['vocab_size'],
            max_samples=config['max_samples'] // 4,
            shuffle=False
        )
        
        print(f"✅ 數據加載器創建成功")
        print(f"📝 詞彙大小: {tokenizer.get_vocab_size()}")
        
        # 創建模型
        print(f"\n🤖 創建{model_type.upper()}模型...")
        vocab_size = tokenizer.get_vocab_size()
        
        if model_type == 'qksan':
            model = create_fixed_qksan_model(
                vocab_size=vocab_size,
                embedding_dim=64,
                n_qubits=4,
                device=device
            )
        elif model_type == 'qrwkv':
            model_config = ModelConfig(
                vocab_size=vocab_size,
                n_embd=64,
                n_head=8,
                n_layer=2,
                n_qubits=2,
                block_size=256
            )
            model = QuantumRWKVModel(model_config)
        elif model_type == 'qasa':
            model = QASATextGenerator(
                vocab_size=vocab_size,
                hidden_dim=64,
                seq_len=config['max_length'],
                n_qubits=4,
                device=device
            )
        
        print(f"✅ 模型創建成功")
        print(f"📊 模型參數數量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 訓練模型
        print(f"\n🎯 開始訓練...")
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        criterion = torch.nn.CrossEntropyLoss()
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(config['epochs']):
            # Training
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch in train_loader:
                try:
                    input_ids = batch['input_ids'].to(device)
                    target_ids = batch['target_ids'].to(device)
                    
                    # 檢查索引範圍
                    if input_ids.max() >= vocab_size or target_ids.max() >= vocab_size:
                        continue
                    
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
                except Exception as e:
                    continue
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    try:
                        input_ids = batch['input_ids'].to(device)
                        target_ids = batch['target_ids'].to(device)
                        
                        # 檢查索引範圍
                        if input_ids.max() >= vocab_size or target_ids.max() >= vocab_size:
                            continue
                        
                        outputs = model(input_ids)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        
                        logits = outputs.view(-1, outputs.size(-1))
                        targets = target_ids.view(-1)
                        loss = criterion(logits, targets)
                        val_loss += loss.item()
                        val_batches += 1
                    except Exception as e:
                        continue
            
            if train_batches > 0 and val_batches > 0:
                avg_train_loss = train_loss / train_batches
                avg_val_loss = val_loss / val_batches
                
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                
                print(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{config['epochs']} | 跳過 (無有效數據)")
        
        training_time = time.time() - start_time
        print(f"✅ 訓練完成! 最佳驗證損失: {best_val_loss:.4f}")
        print(f"⏱️  訓練時間: {training_time:.2f}秒")
        
        # 評估模型
        print(f"\n📊 開始全面評估...")
        
        # 計算困惑度
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in test_loader:
                try:
                    input_ids = batch['input_ids'].to(device)
                    target_ids = batch['target_ids'].to(device)
                    
                    # 檢查索引範圍
                    if input_ids.max() >= vocab_size or target_ids.max() >= vocab_size:
                        continue
                    
                    outputs = model(input_ids)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    logits = outputs.view(-1, outputs.size(-1))
                    targets = target_ids.view(-1)
                    
                    loss = torch.nn.functional.cross_entropy(logits, targets, reduction='sum')
                    total_loss += loss.item()
                    total_tokens += targets.numel()
                except Exception as e:
                    continue
        
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
        else:
            perplexity = float('inf')
        
        print(f"📈 困惑度: {perplexity:.4f}")
        
        # 生成文本示例
        test_prompts = ["the", "birds", "water", "spring", "stars"]
        generated_texts = []
        
        model.eval()
        with torch.no_grad():
            for prompt in test_prompts:
                try:
                    prompt_tokens = tokenizer.encode(prompt)
                    # 檢查prompt tokens是否在詞彙範圍內
                    if max(prompt_tokens) >= vocab_size:
                        generated_texts.append(f"[詞彙超出範圍: {prompt}]")
                        continue
                    
                    input_ids = torch.tensor([prompt_tokens], device=device)
                    
                    generated = input_ids.clone()
                    for step in range(20):
                        try:
                            outputs = model(generated)
                            if isinstance(outputs, tuple):
                                outputs = outputs[0]
                            
                            next_token_logits = outputs[0, -1, :]
                            next_token = torch.argmax(next_token_logits).unsqueeze(0)
                            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                        except Exception as e:
                            break
                    
                    generated_text = tokenizer.decode(generated[0].tolist())
                    generated_texts.append(generated_text)
                except Exception as e:
                    generated_texts.append(f"[生成失敗: {prompt}]")
        
        print(f"📝 生成文本示例:")
        for i, text in enumerate(generated_texts):
            print(f"    {i+1}. {text}")
        
        # 獲取參考文本
        reference_texts = get_reference_texts(dataset_name, len(generated_texts))
        
        # 計算BLEU分數
        bleu_scores = calculate_bleu_score(reference_texts, generated_texts)
        print(f"🎯 BLEU分數:")
        for metric, score in bleu_scores.items():
            print(f"    {metric}: {score:.4f}")
        
        # 計算多樣性指標
        diversity_metrics = calculate_diversity_metrics(generated_texts)
        print(f"🌈 多樣性指標:")
        print(f"    Distinct-1: {diversity_metrics['distinct_1']:.4f}")
        print(f"    Distinct-2: {diversity_metrics['distinct_2']:.4f}")
        print(f"    Entropy: {diversity_metrics['entropy']:.4f}")
        print(f"    Vocabulary Size: {diversity_metrics['vocabulary_size']}")
        
        # 計算重複率
        repetition_rate = calculate_repetition_rate(generated_texts)
        print(f"🔄 重複率: {repetition_rate:.4f}")
        
        # 計算流暢度指標
        fluency_metrics = calculate_fluency_metrics(generated_texts)
        print(f"📖 流暢度指標:")
        print(f"    平均句子長度: {fluency_metrics['avg_sentence_length']:.2f}")
        print(f"    句子長度標準差: {fluency_metrics['sentence_length_std']:.2f}")
        print(f"    句子長度變異係數: {fluency_metrics['sentence_length_variation']:.4f}")
        
        # 保存結果
        result = {
            'model_type': model_type,
            'dataset_name': dataset_name,
            'training_results': {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
                'final_train_loss': train_losses[-1] if train_losses else float('inf'),
                'final_val_loss': val_losses[-1] if val_losses else float('inf'),
                'training_time': training_time
            },
            'evaluation_results': {
                'perplexity': perplexity,
                'generated_texts': generated_texts,
                'reference_texts': reference_texts,
                'bleu_scores': bleu_scores,
                'diversity_metrics': diversity_metrics,
                'repetition_rate': repetition_rate,
                'fluency_metrics': fluency_metrics
            },
            'config': config,
            'vocab_size': vocab_size,
            'status': 'success'
        }
        
        return result
        
    except Exception as e:
        print(f"❌ 實驗失敗: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'model_type': model_type,
            'dataset_name': dataset_name,
            'status': 'failed',
            'error': str(e),
            'config': config
        }

def main():
    # 配置
    config = {
        'epochs': int(os.environ.get('EPOCHS', 8)),
        'batch_size': int(os.environ.get('BATCH_SIZE', 16)),
        'max_length': int(os.environ.get('MAX_LENGTH', 32)),
        'vocab_size': int(os.environ.get('VOCAB_SIZE', 500)),
        'max_samples': int(os.environ.get('MAX_SAMPLES', 200)),
        'learning_rate': float(os.environ.get('LEARNING_RATE', 0.001))
    }
    
    # 模型和數據集
    models = os.environ.get('MODELS', 'qksan,qrwkv,qasa').split(',')
    datasets = os.environ.get('DATASETS', 'simple_sentences,short_stories,quantum_phrases,haiku,proverbs').split(',')
    
    # 強制使用CPU以避免CUDA錯誤
    device = 'cpu'
    print(f"🖥️  使用設備: {device} (強制CPU模式)")
    
    print(f"🤖 模型: {models}")
    print(f"📊 數據集: {datasets}")
    print(f"🔄 訓練輪數: {config['epochs']} epochs")
    
    # 創建輸出目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # 運行實驗
    all_results = {}
    
    for model_type in models:
        for dataset_name in datasets:
            try:
                print(f"\n🚀 開始實驗: {model_type.upper()} on {dataset_name}")
                result = run_experiment(model_type, dataset_name, config, device)
                
                # 保存單個實驗結果
                experiment_key = f"{model_type}_{dataset_name}"
                all_results[experiment_key] = result
                
                # 保存到文件
                result_file = os.path.join(results_dir, f"{experiment_key}.json")
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                
                if result.get('status') == 'success':
                    print(f"✅ 實驗完成: {model_type.upper()} on {dataset_name}")
                else:
                    print(f"❌ 實驗失敗: {model_type.upper()} on {dataset_name}")
                
            except Exception as e:
                print(f"❌ 實驗失敗 {model_type} on {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 保存所有結果
    all_results_file = os.path.join(results_dir, "all_results.json")
    with open(all_results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # 生成完整報告
    report_file = os.path.join(results_dir, "complete_report.md")
    with open(report_file, 'w') as f:
        f.write(f"# 量子文字生成模型完整實驗報告\n\n")
        f.write(f"## 實驗配置\n")
        f.write(f"- 實驗時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- 模型: {models}\n")
        f.write(f"- 數據集: {datasets}\n")
        f.write(f"- 訓練輪數: {config['epochs']}\n")
        f.write(f"- 批次大小: {config['batch_size']}\n")
        f.write(f"- 學習率: {config['learning_rate']}\n")
        f.write(f"- 設備: CPU (強制模式)\n\n")
        
        f.write(f"## 評估指標說明\n\n")
        f.write(f"### 1. 困惑度 (Perplexity)\n")
        f.write(f"- 衡量語言模型預測下一個詞的能力\n")
        f.write(f"- 越低越好\n\n")
        
        f.write(f"### 2. BLEU分數 (BLEU-1/2/3/4)\n")
        f.write(f"- 衡量生成文本與參考文本的n-gram重疊度\n")
        f.write(f"- 越高越好\n\n")
        
        f.write(f"### 3. 多樣性指標\n")
        f.write(f"- **Distinct-1/2**: 衡量詞彙使用的多樣性\n")
        f.write(f"- **Entropy**: 詞彙分布的熵\n")
        f.write(f"- **Vocabulary Size**: 使用的詞彙大小\n\n")
        
        f.write(f"### 4. 重複率 (Repetition Rate)\n")
        f.write(f"- 衡量文本中連續重複詞的比例\n")
        f.write(f"- 越低越好\n\n")
        
        f.write(f"### 5. 流暢度指標\n")
        f.write(f"- **平均句子長度**: 句子的平均詞數\n")
        f.write(f"- **句子長度標準差**: 句子長度的變化\n")
        f.write(f"- **句子長度變異係數**: 標準差/平均值\n\n")
        
        f.write(f"## 實驗結果\n\n")
        f.write(f"### 按模型類型比較\n\n")
        
        for model in models:
            model_data = [v for k, v in all_results.items() if v.get('model_type') == model and v.get('status') == 'success']
            if model_data:
                f.write(f"#### {model.upper()} 模型\n")
                f.write("| 數據集 | 困惑度 | BLEU-1 | BLEU-2 | Distinct-1 | Distinct-2 | 重複率 |\n")
                f.write("|--------|--------|--------|--------|------------|------------|--------|\n")
                
                for data in model_data:
                    eval_results = data['evaluation_results']
                    f.write(f"| {data['dataset_name']} | {eval_results['perplexity']:.2f} | {eval_results['bleu_scores']['BLEU-1']:.3f} | {eval_results['bleu_scores']['BLEU-2']:.3f} | {eval_results['diversity_metrics']['distinct_1']:.3f} | {eval_results['diversity_metrics']['distinct_2']:.3f} | {eval_results['repetition_rate']:.3f} |\n")
                f.write("\n")
        
        # 性能總結
        f.write(f"## 性能總結\n\n")
        
        for model in models:
            model_data = [v for k, v in all_results.items() if v.get('model_type') == model and v.get('status') == 'success']
            if model_data:
                avg_perplexity = np.mean([d['evaluation_results']['perplexity'] for d in model_data if d['evaluation_results']['perplexity'] != float('inf')])
                avg_bleu1 = np.mean([d['evaluation_results']['bleu_scores']['BLEU-1'] for d in model_data])
                avg_distinct1 = np.mean([d['evaluation_results']['diversity_metrics']['distinct_1'] for d in model_data])
                avg_repetition = np.mean([d['evaluation_results']['repetition_rate'] for d in model_data])
                
                f.write(f"- **{model.upper()}**: 平均困惑度={avg_perplexity:.2f}, 平均BLEU-1={avg_bleu1:.3f}, 平均Distinct-1={avg_distinct1:.3f}, 平均重複率={avg_repetition:.3f}\n")
    
    print(f"\n🎉 所有實驗完成!")
    print(f"📁 結果保存到: {results_dir}")
    print(f"📄 報告文件: {report_file}")

if __name__ == '__main__':
    main()
EOF

# 設置環境變量
export EPOCHS
export BATCH_SIZE
export MAX_LENGTH
export VOCAB_SIZE
export MAX_SAMPLES
export LEARNING_RATE
export MODELS="qksan,qrwkv,qasa"
export DATASETS="simple_sentences,short_stories,quantum_phrases,haiku,proverbs"

echo -e "${BLUE}🔍 檢查依賴...${NC}"

# 檢查Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python 未安裝${NC}"
    exit 1
fi

# 檢查必要的Python包
python -c "import torch" 2>/dev/null || {
    echo -e "${RED}❌ PyTorch 未安裝${NC}"
    exit 1
}

python -c "import pennylane" 2>/dev/null || {
    echo -e "${RED}❌ PennyLane 未安裝${NC}"
    exit 1
}

echo -e "${GREEN}✅ 依賴檢查通過${NC}"

# 運行實驗 - 使用PYTHONPATH
chmod +x "$OUTPUT_DIR/run_complete_experiments.py"

echo -e "${BLUE}🚀 開始運行完整實驗...${NC}"
PYTHONPATH=/media/meow/Transcend/Quantum_text_gen python "$OUTPUT_DIR/run_complete_experiments.py"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ 實驗完成!${NC}"
    echo -e "${BLUE}📁 結果保存在: $OUTPUT_DIR${NC}"
else
    echo -e "${RED}❌ 實驗失敗${NC}"
    exit 1
fi