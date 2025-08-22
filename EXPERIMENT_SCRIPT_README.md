# 量子文字生成模型實驗腳本使用說明

## 概述

`run_experiments.sh` 是一個自動化的bash腳本，可以一次性運行QKSAN、QRWKV、QASA三個量子文字生成模型在所有數據集上的訓練和評估。

## 功能特點

- ✅ **自動化運行**: 一次性運行所有模型和數據集
- ✅ **參數可調**: 支持自定義訓練參數
- ✅ **完整評估**: 包含困惑度計算和文本生成
- ✅ **結果保存**: 自動保存所有實驗結果和報告
- ✅ **錯誤處理**: 單個實驗失敗不影響其他實驗
- ✅ **依賴檢查**: 自動檢查必要的Python包

## 支持的模型

1. **QKSAN** - Quantum-Kernel Self-Attention Network
2. **QRWKV** - Quantum RWKV
3. **QASA** - Quantum Attention Sequence Architecture

## 支持的數據集

1. **simple_sentences** - 簡單句子
2. **short_stories** - 短故事
3. **quantum_phrases** - 量子短語
4. **haiku** - 俳句
5. **proverbs** - 諺語

## 使用方法

### 基本用法

```bash
# 使用默認參數運行所有實驗
./run_experiments.sh

# 指定訓練輪數
./run_experiments.sh 10

# 指定多個參數
./run_experiments.sh 15 32 64 1000 300 0.0005
```

### 參數說明

腳本接受最多6個參數（按順序）：

```bash
./run_experiments.sh [epochs] [batch_size] [max_length] [vocab_size] [max_samples] [learning_rate]
```

| 參數 | 默認值 | 說明 |
|------|--------|------|
| epochs | 8 | 訓練輪數 |
| batch_size | 16 | 批次大小 |
| max_length | 32 | 最大序列長度 |
| vocab_size | 500 | 詞彙大小 |
| max_samples | 200 | 最大樣本數 |
| learning_rate | 0.001 | 學習率 |

### 使用示例

```bash
# 快速測試（較少epoch）
./run_experiments.sh 5 8 16 200 100 0.001

# 完整訓練（較多epoch）
./run_experiments.sh 20 32 64 1000 500 0.0005

# 大規模實驗
./run_experiments.sh 50 64 128 2000 1000 0.0001
```

## 輸出結果

腳本會在 `./experiments/bash_run_YYYYMMDD_HHMMSS/` 目錄下生成以下文件：

### 目錄結構
```
experiments/bash_run_20250822_143000/
├── run_experiments.py          # 生成的Python腳本
└── results_20250822_143000/    # 實驗結果目錄
    ├── qksan_simple_sentences.json
    ├── qksan_short_stories.json
    ├── qrwkv_simple_sentences.json
    ├── qrwkv_short_stories.json
    ├── qasa_simple_sentences.json
    ├── qasa_short_stories.json
    ├── all_results.json        # 所有結果匯總
    └── experiment_report.md    # 實驗報告
```

### 結果文件內容

每個實驗的JSON文件包含：

```json
{
  "model_type": "qksan",
  "dataset_name": "simple_sentences",
  "training_results": {
    "train_losses": [3.5, 3.2, 2.9, ...],
    "val_losses": [3.4, 3.1, 2.8, ...],
    "best_val_loss": 2.8,
    "final_train_loss": 2.7,
    "final_val_loss": 2.8,
    "training_time": 925.6
  },
  "evaluation_results": {
    "perplexity": 16.5,
    "generated_texts": [
      "the cat sat on the mat",
      "birds fly in the sky",
      ...
    ]
  },
  "config": {
    "epochs": 8,
    "batch_size": 16,
    "max_length": 32,
    "vocab_size": 500,
    "max_samples": 200,
    "learning_rate": 0.001
  },
  "vocab_size": 42
}
```

### 實驗報告

`experiment_report.md` 包含：

- 實驗配置摘要
- 所有模型的性能比較表格
- 訓練時間統計
- 困惑度比較

## 系統要求

### 必需軟件
- Python 3.7+
- PyTorch
- PennyLane
- NumPy

### 檢查依賴
腳本會自動檢查以下依賴：
- Python 是否安裝
- PyTorch 是否可用
- PennyLane 是否可用

## 故障排除

### 常見問題

1. **權限錯誤**
   ```bash
   chmod +x run_experiments.sh
   ```

2. **Python包缺失**
   ```bash
   pip install torch pennylane numpy
   ```

3. **內存不足**
   - 減少 `batch_size` 或 `max_samples`
   - 使用較小的 `vocab_size`

4. **訓練時間過長**
   - 減少 `epochs`
   - 減少 `max_samples`
   - 使用較小的模型配置

### 錯誤日誌

如果實驗失敗，檢查：
- 控制台輸出的錯誤信息
- 生成的Python腳本是否正確
- 數據集文件是否存在

## 自定義配置

### 修改模型配置

編輯生成的 `run_experiments.py` 文件中的模型配置：

```python
# QKSAN配置
model = create_fixed_qksan_model(
    vocab_size=vocab_size,
    embedding_dim=64,  # 修改隱藏層維度
    n_qubits=4,        # 修改量子比特數
    device=device
)

# QRWKV配置
model_config = ModelConfig(
    vocab_size=vocab_size,
    n_embd=64,         # 修改嵌入維度
    n_head=8,          # 修改注意力頭數
    n_layer=2,         # 修改層數
    n_qubits=2,        # 修改量子比特數
    block_size=256
)
```

### 修改評估指標

在 `run_experiments.py` 中添加更多評估指標：

```python
# 添加BLEU分數計算
from nltk.translate.bleu_score import sentence_bleu

# 添加多樣性指標
def calculate_diversity(texts):
    # 實現多樣性計算
    pass
```

## 性能優化建議

1. **GPU加速**: 確保CUDA可用以加速訓練
2. **批次大小**: 根據GPU內存調整batch_size
3. **數據預處理**: 預先處理數據集以提高加載速度
4. **模型簡化**: 對於快速測試，使用較小的模型配置

## 擴展功能

### 添加新模型

1. 在 `run_experiment` 函數中添加新的模型類型
2. 導入相應的模型模塊
3. 添加模型創建邏輯

### 添加新數據集

1. 在 `data/datasets.py` 中添加新數據集
2. 更新腳本中的數據集列表

### 添加新評估指標

1. 在評估部分添加新的指標計算
2. 更新結果保存格式
3. 修改報告生成邏輯

## 聯繫信息

如有問題或建議，請查看項目文檔或提交issue。
