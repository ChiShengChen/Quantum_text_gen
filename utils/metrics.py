"""
Evaluation metrics for quantum text generation
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import re
from collections import Counter
import math

def calculate_perplexity(loss: float) -> float:
    """Calculate perplexity from cross-entropy loss"""
    return math.exp(min(loss, 100))  # Cap to prevent overflow

def calculate_bleu_score(predictions: List[str], 
                        references: List[str],
                        n_gram: int = 4) -> float:
    """
    Calculate BLEU score for text generation
    Simplified implementation for basic evaluation
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predictions must match number of references")
    
    total_score = 0.0
    
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        
        if len(pred_tokens) == 0:
            continue
        
        # Calculate n-gram precision for each n
        precisions = []
        
        for n in range(1, min(n_gram + 1, len(pred_tokens) + 1)):
            pred_ngrams = get_ngrams(pred_tokens, n)
            ref_ngrams = get_ngrams(ref_tokens, n)
            
            if len(pred_ngrams) == 0:
                precisions.append(0.0)
                continue
            
            # Count matches
            matches = 0
            for ngram in pred_ngrams:
                if ngram in ref_ngrams:
                    matches += min(pred_ngrams[ngram], ref_ngrams[ngram])
            
            precision = matches / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
            precisions.append(precision)
        
        # Calculate geometric mean of precisions
        if precisions:
            # Brevity penalty
            bp = min(1.0, len(pred_tokens) / len(ref_tokens)) if len(ref_tokens) > 0 else 0.0
            
            # Geometric mean
            log_sum = sum(math.log(max(p, 1e-10)) for p in precisions)
            geo_mean = math.exp(log_sum / len(precisions))
            
            bleu = bp * geo_mean
            total_score += bleu
    
    return total_score / len(predictions) if predictions else 0.0

def get_ngrams(tokens: List[str], n: int) -> Counter:
    """Get n-grams from token list"""
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngrams.append(ngram)
    return Counter(ngrams)

def calculate_rouge_l(predictions: List[str], 
                     references: List[str]) -> float:
    """
    Calculate ROUGE-L score (Longest Common Subsequence)
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predictions must match number of references")
    
    total_score = 0.0
    
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            continue
        
        # Calculate LCS
        lcs_length = longest_common_subsequence(pred_tokens, ref_tokens)
        
        # Calculate precision and recall
        precision = lcs_length / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
        recall = lcs_length / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
        
        # Calculate F1 score
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        total_score += f1
    
    return total_score / len(predictions) if predictions else 0.0

def longest_common_subsequence(seq1: List[str], seq2: List[str]) -> int:
    """Calculate length of longest common subsequence"""
    m, n = len(seq1), len(seq2)
    
    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]

def calculate_diversity_metrics(texts: List[str]) -> Dict[str, float]:
    """
    Calculate diversity metrics for generated texts
    """
    if not texts:
        return {'distinct_1': 0.0, 'distinct_2': 0.0, 'entropy': 0.0}
    
    all_tokens = []
    all_bigrams = []
    
    for text in texts:
        tokens = text.lower().split()
        all_tokens.extend(tokens)
        
        # Get bigrams
        for i in range(len(tokens) - 1):
            all_bigrams.append((tokens[i], tokens[i + 1]))
    
    # Calculate distinct-1 (unique unigrams / total unigrams)
    distinct_1 = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0.0
    
    # Calculate distinct-2 (unique bigrams / total bigrams)
    distinct_2 = len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0.0
    
    # Calculate entropy
    token_counts = Counter(all_tokens)
    total_tokens = len(all_tokens)
    entropy = 0.0
    
    if total_tokens > 0:
        for count in token_counts.values():
            prob = count / total_tokens
            entropy -= prob * math.log2(prob)
    
    return {
        'distinct_1': distinct_1,
        'distinct_2': distinct_2,
        'entropy': entropy,
        'vocab_size': len(set(all_tokens)),
        'total_tokens': len(all_tokens)
    }

def calculate_repetition_metrics(texts: List[str]) -> Dict[str, float]:
    """
    Calculate repetition metrics for generated texts
    """
    if not texts:
        return {'repetition_rate': 0.0, 'avg_repetitions': 0.0}
    
    total_repetition_rate = 0.0
    total_repetitions = 0.0
    
    for text in texts:
        tokens = text.lower().split()
        if len(tokens) <= 1:
            continue
        
        # Count token frequencies
        token_counts = Counter(tokens)
        
        # Calculate repetition rate (1 - unique_tokens/total_tokens)
        unique_tokens = len(set(tokens))
        repetition_rate = 1.0 - (unique_tokens / len(tokens))
        total_repetition_rate += repetition_rate
        
        # Calculate average repetitions per unique token
        avg_reps = sum(token_counts.values()) / len(token_counts) if token_counts else 0.0
        total_repetitions += avg_reps
    
    num_texts = len([t for t in texts if len(t.split()) > 1])
    
    return {
        'repetition_rate': total_repetition_rate / num_texts if num_texts > 0 else 0.0,
        'avg_repetitions': total_repetitions / num_texts if num_texts > 0 else 0.0
    }

def calculate_semantic_coherence(texts: List[str]) -> float:
    """
    Simple semantic coherence metric based on sentence structure
    This is a simplified version - in practice, you'd use more sophisticated methods
    """
    if not texts:
        return 0.0
    
    total_coherence = 0.0
    
    for text in texts:
        sentences = re.split(r'[.!?]+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            total_coherence += 1.0  # Single sentence is coherent
            continue
        
        # Simple coherence based on consistent sentence length and structure
        lengths = [len(s.split()) for s in sentences]
        if lengths:
            # Penalize extreme variations in sentence length
            avg_length = sum(lengths) / len(lengths)
            length_variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
            coherence = 1.0 / (1.0 + length_variance / avg_length) if avg_length > 0 else 0.0
        else:
            coherence = 0.0
        
        total_coherence += coherence
    
    return total_coherence / len(texts)

def evaluate_generation_quality(predictions: List[str],
                               references: Optional[List[str]] = None,
                               calculate_bleu: bool = True,
                               calculate_rouge: bool = True) -> Dict[str, float]:
    """
    Comprehensive evaluation of text generation quality
    """
    metrics = {}
    
    # Basic statistics
    lengths = [len(text.split()) for text in predictions]
    metrics['avg_length'] = np.mean(lengths) if lengths else 0.0
    metrics['std_length'] = np.std(lengths) if lengths else 0.0
    metrics['min_length'] = min(lengths) if lengths else 0.0
    metrics['max_length'] = max(lengths) if lengths else 0.0
    
    # Diversity metrics
    diversity = calculate_diversity_metrics(predictions)
    metrics.update(diversity)
    
    # Repetition metrics
    repetition = calculate_repetition_metrics(predictions)
    metrics.update(repetition)
    
    # Semantic coherence
    metrics['coherence'] = calculate_semantic_coherence(predictions)
    
    # Reference-based metrics (if references provided)
    if references and len(references) == len(predictions):
        if calculate_bleu:
            metrics['bleu_score'] = calculate_bleu_score(predictions, references)
        
        if calculate_rouge:
            metrics['rouge_l'] = calculate_rouge_l(predictions, references)
    
    return metrics

class GenerationEvaluator:
    """Evaluator for text generation models"""
    
    def __init__(self):
        self.metrics_history = []
    
    def evaluate(self, 
                predictions: List[str],
                references: Optional[List[str]] = None,
                save_results: bool = True) -> Dict[str, float]:
        """Evaluate predictions and optionally save results"""
        
        metrics = evaluate_generation_quality(
            predictions, references, 
            calculate_bleu=references is not None,
            calculate_rouge=references is not None
        )
        
        if save_results:
            self.metrics_history.append({
                'timestamp': np.datetime64('now'),
                'metrics': metrics,
                'num_predictions': len(predictions)
            })
        
        return metrics
    
    def get_metrics_summary(self) -> Dict[str, float]:
        """Get summary of all evaluations"""
        if not self.metrics_history:
            return {}
        
        # Average metrics across all evaluations
        all_metrics = {}
        for eval_result in self.metrics_history:
            for key, value in eval_result['metrics'].items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)
        
        summary = {}
        for key, values in all_metrics.items():
            summary[f'{key}_mean'] = np.mean(values)
            summary[f'{key}_std'] = np.std(values)
            summary[f'{key}_min'] = np.min(values)
            summary[f'{key}_max'] = np.max(values)
        
        return summary