# 量子文字生成模型專案

## 🚀 快速開始

### 運行完整評估 (包含所有三個模型)

#### 使用默認參數
```bash
./run_all_models.sh
```

#### 自定義參數
```bash
./run_all_models.sh [epochs] [batch_size] [max_length] [vocab_size] [max_samples] [learning_rate]
```

#### 示例
```bash
# 運行5個epoch，批次大小8
./run_all_models.sh 5 8

# 運行10個epoch，批次大小4，最大長度16
./run_all_models.sh 10 4 16

# 完整自定義參數
./run_all_models.sh 15 8 16 300 100 0.0005
```

#### 直接運行Python腳本
```bash
PYTHONPATH=/media/meow/Transcend/Quantum_text_gen python comprehensive_evaluation.py
```

## 📊 支持的模型

1. **QKSAN** - Quantum-Kernel Self-Attention Network
2. **QRWKV** - Quantum RWKV
3. **QASA** - Quantum Attention Sequence Architecture
## 📊 專案總結

### 為什麼之前沒有QASA?
- 之前的comprehensive_evaluation.py只包含QKSAN和QRWKV
- 現在已經修復，包含所有三個模型

### 快速測試
```bash
# 快速測試（2個epoch，小批次，1000樣本）
./run_all_models.sh 2 8

# 完整訓練（8個epoch，標準配置，1000樣本）
./run_all_models.sh 8 16

# 長時間訓練（15個epoch，大批次，1000樣本）
./run_all_models.sh 15 32

# 大數據集訓練（50個epoch，大批次，2000樣本）
./run_all_models.sh 50 16 32 500 2000 0.001
```

### 數據集說明
- **simple_sentences**: 簡單英文句子，可生成5000個樣本
- **short_stories**: 短故事片段，可生成3000個樣本  
- **quantum_phrases**: 量子計算相關短語，可生成2000個樣本
- **haiku**: 日本俳句，可生成1500個樣本
- **proverbs**: 諺語和格言，可生成2500個樣本

**注意**: 默認使用1000個樣本進行訓練（每個數據集800個訓練樣本），可以通過參數調整到更大數據集。

### 可用的bash腳本
- `./run_all_models.sh` - 運行完整評估 (包含QASA)

### 參數設定
腳本支持以下參數（按順序）：
1. **epochs** - 訓練輪數 (默認: 8)
2. **batch_size** - 批次大小 (默認: 16)
3. **max_length** - 最大序列長度 (默認: 32)
4. **vocab_size** - 詞彙表大小 (默認: 500)
5. **max_samples** - 最大樣本數 (默認: 200)
6. **learning_rate** - 學習率 (默認: 0.001)

### 核心文件
- `comprehensive_evaluation.py` - 完整評估腳本
- `qksan_fixed.py` - QKSAN模型
- `qrwkv_model.py` - QRWKV模型
- `qasa_model.py` - QASA模型
- `analyze_results.py` - 結果分析腳本
