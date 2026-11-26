
# QRWKV模型評估報告

## 實驗概覽
- 模型: QRWKV (Quantum RWKV)
- 數據集: ['simple_sentences', 'short_stories', 'quantum_phrases', 'haiku', 'proverbs']
- 訓練輪數: 5
- 批次大小: 8
- 最大長度: 16
- 詞彙大小: 200
- 最大樣本: 200
- 學習率: 0.001

## 結果總結

### simple_sentences
- 困惑度: 4.3247
- BLEU分數: BLEU-1: 0.2952
- 詞彙多樣性: 0.5294
- 重複率: 0.0333

### short_stories
- 困惑度: 1.4377
- BLEU分數: BLEU-1: 0.0889
- 詞彙多樣性: 0.7292
- 重複率: 0.0250

### quantum_phrases
- 困惑度: 3.0723
- BLEU分數: BLEU-1: 0.0821
- 詞彙多樣性: 0.5152
- 重複率: 0.0000

### haiku
- 困惑度: 1.2062
- BLEU分數: BLEU-1: 0.0769
- 詞彙多樣性: 0.7231
- 重複率: 0.0000

### proverbs
- 困惑度: 1.6623
- BLEU分數: BLEU-1: 0.3095
- 詞彙多樣性: 0.6562
- 重複率: 0.0000
