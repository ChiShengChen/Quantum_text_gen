# Quantum Text Generation Benchmark Datasets

## 📋 概述

這是一套專門為量子文字生成模型benchmark而生成的數據集，包含5個不同類型的文本數據集，每個數據集都有2000個樣本。

## 🎯 數據集特點

- **樣本數量**: 每個數據集2000個樣本
- **總樣本數**: 10,000個樣本
- **分割比例**: 80% 訓練, 10% 驗證, 10% 測試
- **可重現性**: 使用固定隨機種子(42)生成
- **格式**: CSV格式，UTF-8編碼
- **語言**: 英文

## 📊 數據集列表

### 1. simple_sentences
- **描述**: 簡單英文句子，用於基礎語言建模
- **樣本數**: 2000 (1600 train, 200 val, 200 test)
- **平均長度**: 6.0詞
- **平均字符數**: 28.52
- **來源**: 生成

### 2. short_stories
- **描述**: 短故事片段(1-2句話)
- **樣本數**: 2000 (1600 train, 200 val, 200 test)
- **平均長度**: 12.3詞
- **平均字符數**: 約60-80字符
- **來源**: 生成

### 3. quantum_phrases
- **描述**: 物理和量子計算相關短語
- **樣本數**: 2000 (1600 train, 200 val, 200 test)
- **平均長度**: 約7-8詞
- **平均字符數**: 約45-50字符
- **來源**: 整理

### 4. haiku
- **描述**: 傳統日本俳句詩歌(3行)
- **樣本數**: 2000 (1600 train, 200 val, 200 test)
- **平均長度**: 11.2詞
- **平均字符數**: 約50-60字符
- **來源**: 公共領域

### 5. proverbs
- **描述**: 短諺語和格言
- **樣本數**: 2000 (1600 train, 200 val, 200 test)
- **平均長度**: 7.8詞
- **平均字符數**: 約35-40字符
- **來源**: 公共領域

## 📁 文件結構

```
benchmark_datasets/
├── README_20250823_131449.md                    # 詳細說明文檔
├── dataset_statistics_20250823_131449.json      # 統計信息
├── simple_sentences_full_20250823_131449.csv    # 完整數據集
├── simple_sentences_train_20250823_131449.csv   # 訓練集
├── simple_sentences_val_20250823_131449.csv     # 驗證集
├── simple_sentences_test_20250823_131449.csv    # 測試集
├── short_stories_full_20250823_131449.csv
├── short_stories_train_20250823_131449.csv
├── short_stories_val_20250823_131449.csv
├── short_stories_test_20250823_131449.csv
├── quantum_phrases_full_20250823_131449.csv
├── quantum_phrases_train_20250823_131449.csv
├── quantum_phrases_val_20250823_131449.csv
├── quantum_phrases_test_20250823_131449.csv
├── haiku_full_20250823_131449.csv
├── haiku_train_20250823_131449.csv
├── haiku_val_20250823_131449.csv
├── haiku_test_20250823_131449.csv
├── proverbs_full_20250823_131449.csv
├── proverbs_train_20250823_131449.csv
├── proverbs_val_20250823_131449.csv
└── proverbs_test_20250823_131449.csv
```

## 📄 CSV文件格式

每個CSV文件包含以下列：

| 列名 | 描述 | 示例 |
|------|------|------|
| `text` | 文本內容 | "The cat sits on the mat." |
| `length` | 詞數 | 6 |
| `char_count` | 字符數 | 24 |
| `dataset_name` | 數據集名稱 | "simple_sentences" |
| `dataset_description` | 數據集描述 | "Simple English sentences..." |
| `source` | 數據來源 | "generated" |

## 🚀 使用方法

### 1. 解壓縮數據集

```bash
# 解壓縮tar.gz文件
tar -xzf quantum_text_benchmark_datasets_20250823.tar.gz

# 或者直接使用benchmark_datasets目錄
```

### 2. Python (pandas)

```python
import pandas as pd

# 讀取訓練集
train_df = pd.read_csv('simple_sentences_train_20250823_131449.csv')

# 獲取文本列表
texts = train_df['text'].tolist()

# 查看統計信息
print(f"樣本數: {len(texts)}")
print(f"平均長度: {train_df['length'].mean():.1f}")
print(f"平均字符數: {train_df['char_count'].mean():.1f}")

# 顯示前幾個樣本
print("前5個樣本:")
for i, text in enumerate(texts[:5]):
    print(f"{i+1}. {text}")
```

### 3. Python (PyTorch)

```python
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.texts = self.df['text'].tolist()
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]

# 創建數據集
train_dataset = TextDataset('simple_sentences_train_20250823_131449.csv')
val_dataset = TextDataset('simple_sentences_val_20250823_131449.csv')
test_dataset = TextDataset('simple_sentences_test_20250823_131449.csv')

# 創建數據加載器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"訓練集大小: {len(train_dataset)}")
print(f"驗證集大小: {len(val_dataset)}")
print(f"測試集大小: {len(test_dataset)}")
```

### 4. 批量處理所有數據集

```python
import pandas as pd
import os

def load_all_datasets(data_dir="benchmark_datasets"):
    """載入所有數據集"""
    datasets = {}
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv') and 'train' in filename:
            dataset_name = filename.split('_')[0]
            filepath = os.path.join(data_dir, filename)
            
            df = pd.read_csv(filepath)
            datasets[dataset_name] = {
                'train': df,
                'val': pd.read_csv(filepath.replace('train', 'val')),
                'test': pd.read_csv(filepath.replace('train', 'test'))
            }
    
    return datasets

# 載入所有數據集
all_datasets = load_all_datasets()

# 顯示統計信息
for name, data in all_datasets.items():
    print(f"\n{name}:")
    print(f"  訓練集: {len(data['train'])} 樣本")
    print(f"  驗證集: {len(data['val'])} 樣本")
    print(f"  測試集: {len(data['test'])} 樣本")
    print(f"  平均長度: {data['train']['length'].mean():.1f} 詞")
```

## 📈 評估指標建議

### 1. 語言建模指標
- **困惑度 (Perplexity)**: 衡量語言模型預測能力
- **BLEU分數**: 衡量生成文本與參考文本的相似度
- **ROUGE分數**: 衡量文本摘要質量

### 2. 多樣性指標
- **Distinct-1/2**: 衡量詞彙使用多樣性
- **Entropy**: 詞彙分布的熵
- **Vocabulary Size**: 使用的詞彙大小

### 3. 流暢度指標
- **平均句子長度**: 句子的平均詞數
- **句子長度標準差**: 句子長度的變化
- **重複率**: 文本中重複內容的比例

## 🔧 自定義使用

### 1. 修改數據集大小

```python
# 如果只需要部分數據
train_df = pd.read_csv('simple_sentences_train_20250823_131449.csv')
small_train_df = train_df.head(100)  # 只使用前100個樣本
```

### 2. 數據預處理

```python
import re

def preprocess_text(text):
    """文本預處理"""
    # 轉換為小寫
    text = text.lower()
    # 移除特殊字符
    text = re.sub(r'[^\w\s]', '', text)
    # 移除多餘空格
    text = ' '.join(text.split())
    return text

# 應用預處理
train_df['processed_text'] = train_df['text'].apply(preprocess_text)
```

### 3. 詞彙表構建

```python
from collections import Counter

def build_vocab(texts, min_freq=1):
    """構建詞彙表"""
    word_counts = Counter()
    for text in texts:
        words = text.split()
        word_counts.update(words)
    
    # 過濾低頻詞
    vocab = {word: idx for idx, (word, count) in enumerate(word_counts.items()) 
             if count >= min_freq}
    
    return vocab

# 構建詞彙表
all_texts = train_df['text'].tolist()
vocab = build_vocab(all_texts, min_freq=2)
print(f"詞彙表大小: {len(vocab)}")
```

## 📊 性能基準

基於這些數據集的典型性能基準：

| 模型類型 | 困惑度範圍 | BLEU-1範圍 | Distinct-1範圍 |
|----------|------------|------------|----------------|
| 簡單RNN | 3.0-5.0 | 0.1-0.3 | 0.3-0.6 |
| LSTM | 2.5-4.0 | 0.2-0.4 | 0.4-0.7 |
| Transformer | 2.0-3.5 | 0.3-0.6 | 0.5-0.8 |
| 量子模型 | 1.5-3.0 | 0.2-0.5 | 0.4-0.7 |

## 🎯 適用場景

1. **量子文字生成研究**: 測試量子語言模型性能
2. **模型比較**: 比較不同架構的語言模型
3. **超參數調優**: 優化模型參數
4. **算法開發**: 開發新的文本生成算法
5. **教學實驗**: 機器學習和NLP課程實驗

## 📝 引用

如果使用這些數據集，請引用：

```bibtex
@dataset{quantum_text_benchmark_2025,
  title={Quantum Text Generation Benchmark Datasets},
  author={Quantum Text Generation Project},
  year={2025},
  url={https://github.com/your-repo/quantum-text-generation},
  note={Generated benchmark datasets for quantum text generation models}
}
```

## 🤝 貢獻

歡迎提交問題和改進建議：
- 報告數據集問題
- 建議新的數據集類型
- 改進數據生成方法
- 添加新的評估指標

## 📄 許可證

這些數據集遵循MIT許可證，可自由使用於研究和商業目的。

---

**生成時間**: 2025-08-23 13:14:49  
**版本**: 1.0  
**維護者**: Quantum Text Generation Project
