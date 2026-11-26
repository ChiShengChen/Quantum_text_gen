
# QRWKV模型評估報告

## 實驗概覽
- 模型: QRWKV (Quantum RWKV)
- 數據集: ['simple_sentences', 'short_stories', 'quantum_phrases', 'haiku', 'proverbs']
- 訓練輪數: 50
- 批次大小: 16
- 最大長度: 32
- 詞彙大小: 500
- 最大樣本: 1000
- 學習率: 0.001

## 結果總結

### simple_sentences
- 困惑度: 3.2545
- BLEU分數: BLEU-1: 0.2286
- 詞彙多樣性: 0.5429
- 重複率: 0.0333

### short_stories
- 困惑度: 1.3437
- BLEU分數: BLEU-1: 0.1111
- 詞彙多樣性: 0.4783
- 重複率: 0.0250

### quantum_phrases
- 困惑度: 2.1007
- BLEU分數: BLEU-1: 0.0857
- 詞彙多樣性: 0.4571
- 重複率: 0.0000

### haiku
- 困惑度: 1.0540
- BLEU分數: BLEU-1: 0.0154
- 詞彙多樣性: 0.7031
- 重複率: 0.0000

### proverbs
- 困惑度: 1.3543
- BLEU分數: BLEU-1: 0.0333
- 詞彙多樣性: 0.6129
- 重複率: 0.0000
