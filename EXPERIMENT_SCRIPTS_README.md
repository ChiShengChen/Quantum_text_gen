# 實驗腳本使用指南

## 📋 概述

本專案現在包含了完整的實驗腳本系統，支持Transformer baseline與量子模型的比較評估。

## 🚀 主要腳本

### 1. 主實驗腳本 (`run_experiments.sh`)

**功能**: 統一的實驗入口，支持多種評估模式

**使用方法**:
```bash
./run_experiments.sh [模式] [參數...]
```

**可用模式**:
- `quick` - 快速比較 (包含Transformer baseline)
- `full` - 完整評估 (包含Transformer baseline)  
- `original` - 原始評估 (只有量子模型)
- `test` - 測試Transformer模型

**示例**:
```bash
# 快速比較
./run_experiments.sh quick

# 完整評估，自定義參數
./run_experiments.sh full 30 8 16 500 50 0.001

# 測試Transformer模型
./run_experiments.sh test

# 查看幫助
./run_experiments.sh --help
```

### 2. Transformer Baseline評估 (`run_transformer_baseline.sh`)

**功能**: 專門運行包含Transformer baseline的完整評估

**使用方法**:
```bash
./run_transformer_baseline.sh [EPOCHS] [BATCH_SIZE] [MAX_LENGTH] [VOCAB_SIZE] [MAX_SAMPLES] [LEARNING_RATE]
```

**默認參數**:
- EPOCHS: 50
- BATCH_SIZE: 16
- MAX_LENGTH: 32
- VOCAB_SIZE: 1000
- MAX_SAMPLES: 100
- LEARNING_RATE: 0.001

### 3. 快速比較腳本 (`run_quick_comparison.sh`)

**功能**: 快速比較所有模型，只測試simple_sentences數據集

**使用方法**:
```bash
./run_quick_comparison.sh [EPOCHS] [BATCH_SIZE] [MAX_LENGTH] [VOCAB_SIZE] [MAX_SAMPLES] [LEARNING_RATE]
```

**默認參數** (優化為快速測試):
- EPOCHS: 20
- BATCH_SIZE: 8
- MAX_LENGTH: 16
- VOCAB_SIZE: 500
- MAX_SAMPLES: 50
- LEARNING_RATE: 0.001

### 4. 原始評估腳本 (`run_all_models.sh`)

**功能**: 原始的三個量子模型評估 (已更新支持Transformer)

**使用方法**:
```bash
./run_all_models.sh [EPOCHS] [BATCH_SIZE] [MAX_LENGTH] [VOCAB_SIZE] [MAX_SAMPLES] [LEARNING_RATE]
```

## 📊 評估模型

所有腳本現在都支持以下四個模型:

1. **QKSAN** - 量子Kernel Self-Attention Network
2. **QRWKV** - Quantum RWKV模型
3. **QASA** - Quantum Attention with Self-Attention
4. **TRANSFORMER** - 傳統Transformer Decoder (baseline)

## 📁 結果輸出

### 快速比較結果
```
./experiments/quick_comparison_YYYYMMDD_HHMMSS/
├── quick_comparison_results.json
└── quick_comparison_report.md
```

### 完整評估結果
```
./experiments/comprehensive/comprehensive_report_with_transformer_YYYYMMDD_HHMMSS/
├── all_comprehensive_results.json
└── comprehensive_report.md
```

### 原始評估結果
```
./experiments/comprehensive/comprehensive_report_YYYYMMDD_HHMMSS/
├── all_comprehensive_results.json
└── comprehensive_report.md
```

## 🎯 推薦使用流程

### 1. 首次使用 - 測試模型
```bash
./run_experiments.sh test
```

### 2. 快速驗證 - 快速比較
```bash
./run_experiments.sh quick
```

### 3. 完整評估 - 包含Transformer baseline
```bash
./run_experiments.sh full
```

### 4. 原始評估 - 只有量子模型
```bash
./run_experiments.sh original
```

## ⚙️ 環境要求

- Python 3.7+
- PyTorch
- PennyLane
- 其他依賴見requirements.txt

## 🔧 故障排除

### 常見問題

1. **權限錯誤**
   ```bash
   chmod +x run_experiments.sh run_transformer_baseline.sh run_quick_comparison.sh
   ```

2. **依賴缺失**
   ```bash
   pip install torch pennylane
   ```

3. **CUDA錯誤**
   - 腳本已配置為使用CPU
   - 如需GPU，請修改環境變量

### 日誌查看

所有腳本都會輸出詳細的執行日誌，包括:
- 依賴檢查
- 模型創建
- 訓練進度
- 評估結果
- 錯誤信息

## 📈 性能比較

使用這些腳本，你可以輕鬆比較:

- **困惑度** - 語言建模準確性
- **BLEU分數** - 文本生成質量
- **多樣性指標** - 詞彙使用多樣性
- **重複率** - 文本重複程度
- **流暢度指標** - 句子結構質量

## 🎉 總結

現在你可以通過簡單的命令行操作，輕鬆進行:

1. **快速驗證** - 驗證Transformer baseline的效果
2. **完整比較** - 全面評估所有模型的性能
3. **基準測試** - 建立傳統深度學習的baseline
4. **性能分析** - 深入分析量子模型的優勢和劣勢

這為量子文字生成研究提供了完整的實驗框架！
