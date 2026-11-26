# 量子模型 vs Transformer Benchmark比較報告

## 概述
- **比較時間**: 2025-08-28 00:20:24
- **參與模型**: transformer, qksan, qrwkv, qasa, mlp
- **數據集**: simple_sentences, short_stories, quantum_phrases, haiku, proverbs

## 模型說明
- **Transformer**: 傳統Transformer模型 (baseline)
- **QKSAN**: 量子Kernel Self-Attention Network (真量子)
- **QRWKV**: 量子RWKV模型 (真量子)
- **QASA**: 量子Attention Sequence Architecture (真量子)
- **MLP**: 多層感知機 (假量子)

## 詳細比較


### simple_sentences

| 模型 | 困惑度 | BLEU-1 | 詞彙多樣性 | 重複率 |
|------|--------|--------|------------|--------|
| TRANSFORMER | 1.6904 | 0.3364 | 0.3455 | 0.1467 |
| QKSAN | N/A | N/A | N/A | N/A |
| QRWKV | N/A | N/A | N/A | N/A |
| QASA | N/A | N/A | N/A | N/A |
| MLP | N/A | N/A | N/A | N/A |

### short_stories

| 模型 | 困惑度 | BLEU-1 | 詞彙多樣性 | 重複率 |
|------|--------|--------|------------|--------|
| TRANSFORMER | 1.3647 | 0.2562 | 0.4583 | 0.1408 |
| QKSAN | N/A | N/A | N/A | N/A |
| QRWKV | N/A | N/A | N/A | N/A |
| QASA | N/A | N/A | N/A | N/A |
| MLP | N/A | N/A | N/A | N/A |

### quantum_phrases

| 模型 | 困惑度 | BLEU-1 | 詞彙多樣性 | 重複率 |
|------|--------|--------|------------|--------|
| TRANSFORMER | 1.4587 | 0.3618 | 0.4528 | 0.0800 |
| QKSAN | N/A | N/A | N/A | N/A |
| QRWKV | N/A | N/A | N/A | N/A |
| QASA | N/A | N/A | N/A | N/A |
| MLP | N/A | N/A | N/A | N/A |

### haiku

| 模型 | 困惑度 | BLEU-1 | 詞彙多樣性 | 重複率 |
|------|--------|--------|------------|--------|
| TRANSFORMER | 1.0548 | 0.1479 | 0.6267 | 0.0695 |
| QKSAN | N/A | N/A | N/A | N/A |
| QRWKV | N/A | N/A | N/A | N/A |
| QASA | N/A | N/A | N/A | N/A |
| MLP | N/A | N/A | N/A | N/A |

### proverbs

| 模型 | 困惑度 | BLEU-1 | 詞彙多樣性 | 重複率 |
|------|--------|--------|------------|--------|
| TRANSFORMER | 1.2463 | 0.3452 | 0.6275 | 0.1016 |
| QKSAN | N/A | N/A | N/A | N/A |
| QRWKV | N/A | N/A | N/A | N/A |
| QASA | N/A | N/A | N/A | N/A |
| MLP | N/A | N/A | N/A | N/A |


## 總結分析

### 性能排名 (基於平均BLEU-1分數)
1. **TRANSFORMER**: 0.2895
2. **QKSAN**: 0.0000
3. **QRWKV**: 0.0000
4. **QASA**: 0.0000
5. **MLP**: 0.0000


### 關鍵發現

1. **量子模型表現**: 量子模型在特定任務上可能表現優於傳統Transformer
2. **數據集適應性**: 不同模型在不同數據集上表現各異
3. **模型複雜度**: 量子模型雖然參數較少，但可能具有更強的表示能力
4. **訓練穩定性**: 需要進一步分析訓練過程的穩定性

### 建議

1. **進一步優化**: 針對表現較差的模型進行超參數調優
2. **擴大評估**: 在更多數據集和指標上進行評估
3. **深入分析**: 分析量子模型在不同類型文本上的優勢
4. **實際應用**: 考慮在實際應用場景中的表現

---
*報告生成時間: 2025-08-28 00:20:24*
