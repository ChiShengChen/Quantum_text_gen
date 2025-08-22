#!/usr/bin/env python3
"""
Multi-Metric Evaluation System for Quantum Text Generation
全面的評估指標系統
"""

import torch
import torch.nn.functional as F
import numpy as np
import re
import math
from typing import List, Dict, Any, Tuple
from collections import Counter
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class MultiMetricEvaluator:
    """多指標評估器"""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1
    
    def calculate_perplexity(self, model, data_loader, device='cpu') -> float:
        """計算困惑度"""
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                target_ids = batch['target_ids'].to(device)
                
                outputs = model(input_ids)
                logits = outputs.view(-1, outputs.size(-1))
                targets = target_ids.view(-1)
                
                loss = F.cross_entropy(logits, targets, reduction='sum')
                total_loss += loss.item()
                total_tokens += targets.numel()
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        return perplexity
    
    def calculate_bleu_score(self, references: List[List[str]], 
                           candidates: List[str], 
                           weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)) -> float:
        """計算BLEU分數"""
        if len(references) != len(candidates):
            raise ValueError("References and candidates must have the same length")
        
        total_bleu = 0.0
        for ref, cand in zip(references, candidates):
            # Tokenize
            ref_tokens = [ref.split() if isinstance(ref, str) else ref]
            cand_tokens = cand.split() if isinstance(cand, str) else cand
            
            # Calculate BLEU
            bleu = sentence_bleu(ref_tokens, cand_tokens, weights=weights, smoothing_function=self.smoothing)
            total_bleu += bleu
        
        return total_bleu / len(candidates)
    
    def calculate_rouge_scores(self, references: List[str], candidates: List[str]) -> Dict[str, float]:
        """計算ROUGE分數"""
        if len(references) != len(candidates):
            raise ValueError("References and candidates must have the same length")
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for ref, cand in zip(references, candidates):
            scores = self.rouge_scorer.score(ref, cand)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        return {
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores)
        }
    
    def calculate_meteor_score(self, references: List[List[str]], candidates: List[str]) -> float:
        """計算METEOR分數"""
        if len(references) != len(candidates):
            raise ValueError("References and candidates must have the same length")
        
        # Tokenize
        ref_tokens = [[ref.split() if isinstance(ref, str) else ref] for ref in references]
        cand_tokens = [cand.split() if isinstance(cand, str) else cand for cand in candidates]
        
        try:
            meteor = meteor_score(ref_tokens, cand_tokens)
            return meteor
        except Exception as e:
            print(f"METEOR計算失敗: {e}")
            return 0.0
    
    def calculate_diversity_metrics(self, texts: List[str], n_grams: List[int] = [1, 2]) -> Dict[str, float]:
        """計算多樣性指標"""
        metrics = {}
        
        for n in n_grams:
            # Distinct-n
            all_ngrams = []
            for text in texts:
                tokens = text.split()
                ngrams = [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
                all_ngrams.extend(ngrams)
            
            if all_ngrams:
                distinct_ngrams = len(set(all_ngrams))
                total_ngrams = len(all_ngrams)
                distinct_n = distinct_ngrams / total_ngrams if total_ngrams > 0 else 0.0
                metrics[f'distinct_{n}'] = distinct_n
            else:
                metrics[f'distinct_{n}'] = 0.0
        
        # Entropy
        all_tokens = []
        for text in texts:
            tokens = text.split()
            all_tokens.extend(tokens)
        
        if all_tokens:
            token_counts = Counter(all_tokens)
            total_tokens = len(all_tokens)
            entropy = -sum((count/total_tokens) * math.log2(count/total_tokens) 
                          for count in token_counts.values())
            metrics['entropy'] = entropy
        else:
            metrics['entropy'] = 0.0
        
        return metrics
    
    def calculate_repetition_rate(self, texts: List[str]) -> float:
        """計算重複率"""
        total_repetitions = 0
        total_tokens = 0
        
        for text in texts:
            tokens = text.split()
            total_tokens += len(tokens)
            
            # Count consecutive repetitions
            repetitions = 0
            for i in range(1, len(tokens)):
                if tokens[i] == tokens[i-1]:
                    repetitions += 1
            
            total_repetitions += repetitions
        
        repetition_rate = total_repetitions / total_tokens if total_tokens > 0 else 0.0
        return repetition_rate
    
    def calculate_coherence_score(self, texts: List[str]) -> float:
        """計算連貫性分數 (基於詞彙重疊)"""
        coherence_scores = []
        
        for text in texts:
            sentences = text.split('.')
            if len(sentences) < 2:
                coherence_scores.append(1.0)
                continue
            
            # Calculate word overlap between consecutive sentences
            overlaps = []
            for i in range(len(sentences) - 1):
                sent1 = set(sentences[i].split())
                sent2 = set(sentences[i+1].split())
                
                if sent1 and sent2:
                    overlap = len(sent1.intersection(sent2)) / len(sent1.union(sent2))
                    overlaps.append(overlap)
            
            avg_overlap = np.mean(overlaps) if overlaps else 0.0
            coherence_scores.append(avg_overlap)
        
        return np.mean(coherence_scores)
    
    def calculate_fluency_score(self, texts: List[str]) -> float:
        """計算流暢性分數 (基於句子長度變化)"""
        fluency_scores = []
        
        for text in texts:
            sentences = text.split('.')
            if len(sentences) < 2:
                fluency_scores.append(1.0)
                continue
            
            # Calculate sentence length variation
            lengths = [len(sent.split()) for sent in sentences if sent.strip()]
            if len(lengths) < 2:
                fluency_scores.append(1.0)
                continue
            
            # Normalized standard deviation of sentence lengths
            mean_length = np.mean(lengths)
            std_length = np.std(lengths)
            cv = std_length / mean_length if mean_length > 0 else 0.0
            
            # Higher CV means more variation (potentially less fluent)
            fluency = 1.0 / (1.0 + cv)
            fluency_scores.append(fluency)
        
        return np.mean(fluency_scores)
    
    def evaluate_generated_texts(self, 
                               references: List[str],
                               candidates: List[str],
                               model=None,
                               data_loader=None,
                               device='cpu') -> Dict[str, float]:
        """全面評估生成的文本"""
        
        print("開始多指標評估...")
        
        results = {}
        
        # 1. BLEU Score
        print("計算BLEU分數...")
        bleu_1 = self.calculate_bleu_score(references, candidates, weights=(1.0, 0, 0, 0))
        bleu_2 = self.calculate_bleu_score(references, candidates, weights=(0.5, 0.5, 0, 0))
        bleu_4 = self.calculate_bleu_score(references, candidates, weights=(0.25, 0.25, 0.25, 0.25))
        
        results['bleu_1'] = bleu_1
        results['bleu_2'] = bleu_2
        results['bleu_4'] = bleu_4
        
        # 2. ROUGE Scores
        print("計算ROUGE分數...")
        rouge_scores = self.calculate_rouge_scores(references, candidates)
        results.update(rouge_scores)
        
        # 3. METEOR Score
        print("計算METEOR分數...")
        meteor = self.calculate_meteor_score(references, candidates)
        results['meteor'] = meteor
        
        # 4. Diversity Metrics
        print("計算多樣性指標...")
        diversity = self.calculate_diversity_metrics(candidates)
        results.update(diversity)
        
        # 5. Repetition Rate
        print("計算重複率...")
        repetition_rate = self.calculate_repetition_rate(candidates)
        results['repetition_rate'] = repetition_rate
        
        # 6. Coherence Score
        print("計算連貫性分數...")
        coherence = self.calculate_coherence_score(candidates)
        results['coherence'] = coherence
        
        # 7. Fluency Score
        print("計算流暢性分數...")
        fluency = self.calculate_fluency_score(candidates)
        results['fluency'] = fluency
        
        # 8. Perplexity (if model and data_loader provided)
        if model is not None and data_loader is not None:
            print("計算困惑度...")
            perplexity = self.calculate_perplexity(model, data_loader, device)
            results['perplexity'] = perplexity
        
        print("評估完成!")
        return results
    
    def generate_evaluation_report(self, results: Dict[str, float], model_name: str, dataset_name: str) -> str:
        """生成評估報告"""
        
        report = f"""
# 多指標評估報告

## 模型信息
- **模型**: {model_name}
- **數據集**: {dataset_name}
- **評估時間**: {torch.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 評估結果

### 1. 生成質量指標
- **BLEU-1**: {results.get('bleu_1', 0):.4f}
- **BLEU-2**: {results.get('bleu_2', 0):.4f}
- **BLEU-4**: {results.get('bleu_4', 0):.4f}
- **ROUGE-1**: {results.get('rouge1', 0):.4f}
- **ROUGE-2**: {results.get('rouge2', 0):.4f}
- **ROUGE-L**: {results.get('rougeL', 0):.4f}
- **METEOR**: {results.get('meteor', 0):.4f}

### 2. 多樣性指標
- **Distinct-1**: {results.get('distinct_1', 0):.4f}
- **Distinct-2**: {results.get('distinct_2', 0):.4f}
- **Entropy**: {results.get('entropy', 0):.4f}

### 3. 文本質量指標
- **重複率**: {results.get('repetition_rate', 0):.4f}
- **連貫性**: {results.get('coherence', 0):.4f}
- **流暢性**: {results.get('fluency', 0):.4f}

### 4. 模型性能指標
- **困惑度**: {results.get('perplexity', 0):.4f}

## 評估總結
- **整體質量**: {(results.get('bleu_4', 0) + results.get('rougeL', 0) + results.get('meteor', 0)) / 3:.4f}
- **多樣性**: {results.get('distinct_1', 0):.4f}
- **連貫性**: {results.get('coherence', 0):.4f}
"""
        
        return report

def test_multi_metrics():
    """測試多指標評估系統"""
    
    print("測試多指標評估系統")
    print("=" * 50)
    
    evaluator = MultiMetricEvaluator()
    
    # 測試數據
    references = [
        "The cat sits on the mat.",
        "Birds fly in the sky.",
        "Water flows down the river."
    ]
    
    candidates = [
        "The cat sits on the mat.",
        "Birds fly in the sky.",
        "Water flows down the river."
    ]
    
    # 評估
    results = evaluator.evaluate_generated_texts(references, candidates)
    
    # 生成報告
    report = evaluator.generate_evaluation_report(results, "Test Model", "Test Dataset")
    
    print("評估結果:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n評估報告:")
    print(report)
    
    return results

if __name__ == '__main__':
    test_multi_metrics()
