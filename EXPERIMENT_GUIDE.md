# 量子文字生成模型實驗指南

## 🚀 快速開始

### 1. 運行完整評估實驗
```bash
# 使用現有的comprehensive_evaluation.py（包含QKSAN和QRWKV）
./run_complete.sh

# 或者直接運行
PYTHONPATH=/media/meow/Transcend/Quantum_text_gen python comprehensive_evaluation.py
```

### 2. 運行自定義參數實驗
```bash
# 使用bash腳本（需要修復路徑問題）
./run_experiments.sh [epochs] [batch_size] [max_length] [vocab_size] [max_samples] [learning_rate]

# 示例：運行5個epoch，批次大小8，最大長度16
./run_experiments.sh 5 8 16 200 50 0.001
```

## 📊 實驗結果

### 最新實驗結果（2025-08-23）
- **QKSAN模型**: 在5個數據集上成功訓練和評估
- **QRWKV模型**: 在5個數據集上成功訓練和評估
- **QASA模型**: 需要進一步調試

### 性能總結
- **QKSAN**: 平均困惑度=88.53, 平均BLEU-1=0.042, 平均Distinct-1=0.030, 平均重複率=0.950
- **QRWKV**: 平均困惑度=1.50, 平均BLEU-1=0.080, 平均Distinct-1=0.440, 平均重複率=0.000

## 🔧 技術細節

### 支持的模型
1. **QKSAN** - Quantum-Kernel Self-Attention Network
2. **QRWKV** - Quantum RWKV
3. **QASA** - Quantum Attention Sequence Architecture

### 支持的數據集
1. **simple_sentences** - 簡單句子
2. **short_stories** - 短故事
3. **quantum_phrases** - 量子短語
4. **haiku** - 俳句
5. **proverbs** - 諺語

### 評估指標
1. **困惑度 (Perplexity)** - 語言模型預測能力
2. **BLEU分數** - 文本生成質量
3. **多樣性指標** - Distinct-1/2, Entropy, Vocabulary Size
4. **重複率** - 文本重複程度
5. **流暢度指標** - 句子長度統計

## 🛠️ 故障排除

### 常見問題

#### 1. ModuleNotFoundError: No module named 'data'
**解決方案**: 設置PYTHONPATH環境變量
```bash
export PYTHONPATH=/media/meow/Transcend/Quantum_text_gen
```

#### 2. CUDA錯誤
**解決方案**: 強制使用CPU
```bash
# 在Python腳本中設置
device = 'cpu'
```

#### 3. 內存不足
**解決方案**: 減少批次大小和最大樣本數
```bash
./run_experiments.sh 5 4 16 100 20 0.001
```

## 📁 輸出文件

實驗完成後，結果保存在以下目錄：
```
./experiments/comprehensive/
├── comprehensive_report_YYYYMMDD_HHMMSS/
│   ├── all_comprehensive_results.json
│   └── comprehensive_report.md
├── qksan/
│   └── [dataset_name]_YYYYMMDD_HHMMSS/
└── qrwkv/
    └── [dataset_name]_YYYYMMDD_HHMMSS/
```

## 🎯 最佳實踐

1. **小規模測試**: 先用較小的參數進行測試
2. **監控資源**: 注意CPU和內存使用情況
3. **保存結果**: 定期保存實驗結果
4. **錯誤處理**: 檢查錯誤日誌並調整參數

## 📈 性能優化建議

1. **減少量子比特數**: 降低n_qubits參數
2. **使用較小的嵌入維度**: 降低embedding_dim
3. **減少層數**: 降低n_layer參數
4. **使用較小的詞彙表**: 降低vocab_size

## 🔮 未來改進

1. **添加QASA支持**: 修復QASA模型的問題
2. **GPU支持**: 優化CUDA兼容性
3. **更多評估指標**: 添加ROUGE、METEOR等
4. **可視化**: 添加訓練過程可視化
5. **超參數調優**: 自動化超參數搜索
