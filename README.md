# Quantum Text Generation Project

基於量子計算的文字生成專案，整合了三種不同的量子模型架構進行自然語言處理和文字生成。

## 專案架構

```
Quantum_text_gen/
├── config/                 # 配置文件
│   ├── model_configs.py    # 模型配置
│   └── data_configs.py     # 數據配置
├── data/                   # 數據處理模組
│   └── datasets.py         # 數據集載入和處理
├── models/                 # 量子模型
│   ├── qasa_model.py      # QASA模型
│   ├── qksan_improved.py  # QKSAN模型
│   ├── qrwkv_model.py     # QRWKV模型
│   └── quantum_text_models.py  # 文字生成適配器
├── training/              # 訓練模組
│   └── trainer.py         # 訓練器
├── generation/            # 文字生成模組
│   └── generator.py       # 生成器
├── utils/                 # 工具模組
│   ├── logger.py          # 日誌工具
│   └── metrics.py         # 評估指標
└── quantum_text_gen.py    # 主程式入口
```

## 量子模型介紹

### 1. QASA (Quantum Attention Sequence Architecture)
- **特點**: 結合量子電路和Transformer架構的序列處理模型
- **適用**: 短序列文字生成，時間步相關的任務
- **量子特性**: 使用量子電路處理注意力機制

### 2. QKSAN (Quantum-Kernel Self-Attention Network)
- **特點**: 基於量子核方法的增強分類器，支援多種編碼和ansatz
- **適用**: 分類任務適配的文字生成
- **量子特性**: 振幅編碼、角度編碼，Hardware-Efficient和QAOA ansatz

### 3. QRWKV (Quantum RWKV)
- **特點**: 量子增強的RWKV架構，結合時間混合和量子通道混合
- **適用**: 長序列建模和高效推理
- **量子特性**: 量子通道混合層，保持RWKV的線性複雜度

## 數據集

專案提供5個適合量子計算的小型數據集：

1. **Simple Sentences** (5000樣本)
   - 簡單英文句子，適合基礎語言建模
   - 詞彙量: 500，平均長度: 8.5詞

2. **Short Stories** (3000樣本)
   - 短篇故事片段，1-2句話
   - 詞彙量: 800，平均長度: 12.3詞

3. **Quantum Phrases** (2000樣本)
   - 物理和量子計算相關短語
   - 詞彙量: 600，平均長度: 9.8詞

4. **Haiku** (1500樣本)
   - 傳統日式俳句（3行）
   - 詞彙量: 400，平均長度: 11.2詞

5. **Proverbs** (2500樣本)
   - 短諺語和格言
   - 詞彙量: 600，平均長度: 7.8詞

## 安裝要求

```bash
pip install -r requirements.txt
```

主要依賴：
- PyTorch >= 1.9.0
- PennyLane >= 0.28.0
- NumPy >= 1.20.0
- Scikit-learn >= 1.0.0

## 使用方法

### 1. 訓練模型

```bash
# 訓練QASA模型
python quantum_text_gen.py --mode train --model qasa --dataset simple_sentences --epochs 50 --batch_size 16

# 訓練QKSAN模型
python quantum_text_gen.py --mode train --model qksan --dataset quantum_phrases --qksan_variant PCAAmHE --n_qubits 6

# 訓練QRWKV模型
python quantum_text_gen.py --mode train --model qrwkv --dataset short_stories --n_layers 4 --n_heads 8
```

### 2. 文字生成

```bash
# 使用訓練好的模型生成文字
python quantum_text_gen.py --mode generate --checkpoint ./checkpoints/best_model.pth --prompt "The cat" --max_length 50 --temperature 1.0 --do_sample

# 生成多個序列
python quantum_text_gen.py --mode generate --checkpoint ./checkpoints/best_model.pth --prompt "Quantum computing" --num_sequences 3
```

### 3. 互動模式

```bash
# 啟動互動式文字生成
python quantum_text_gen.py --mode interactive --model qasa

# 使用已訓練的模型進行互動
python quantum_text_gen.py --mode interactive --checkpoint ./checkpoints/best_model.pth
```

### 4. 模型評估

```bash
# 評估模型生成質量
python quantum_text_gen.py --mode evaluate --checkpoint ./checkpoints/best_model.pth --dataset simple_sentences
```

## 參數說明

### 模型參數
- `--model`: 量子模型類型 (qasa, qksan, qrwkv)
- `--hidden_dim`: 隱藏層維度 (default: 128)
- `--n_qubits`: 量子比特數量 (default: 6)
- `--n_layers`: 網路層數 (default: 4)

### 數據參數
- `--dataset`: 數據集名稱
- `--vocab_size`: 詞彙表大小 (default: 1000)
- `--max_length`: 最大序列長度 (default: 64)
- `--batch_size`: 批次大小 (default: 16)

### 訓練參數
- `--epochs`: 訓練輪數 (default: 50)
- `--learning_rate`: 學習率 (default: 1e-3)
- `--optimizer`: 優化器 (adam, adamw, sgd)
- `--scheduler`: 學習率調度器 (cosine, step, plateau)

### 生成參數
- `--temperature`: 採樣溫度 (default: 1.0)
- `--top_k`: Top-k採樣 (default: 50)
- `--top_p`: Nucleus採樣 (default: 0.9)
- `--do_sample`: 是否使用採樣

## 實驗結果示例

### QASA模型在Simple Sentences上的結果：
```
Epoch 45: Train Loss: 2.1234, Train PPL: 8.35
Epoch 45: Val Loss: 2.3456, Val PPL: 10.44
Best validation loss: 2.2890
```

### 生成示例：
```
Input: "The cat"
Generated: "The cat sits on the warm sunny windowsill watching birds."

Input: "Quantum computing"
Generated: "Quantum computing uses superposition and entanglement to process information."
```

## 評估指標

專案提供多種文字生成評估指標：

- **Perplexity**: 語言模型困惑度
- **BLEU Score**: 與參考文本的相似度
- **ROUGE-L**: 最長公共子序列F1分數
- **Diversity Metrics**: 
  - Distinct-1/2: 唯一n-gram比例
  - Entropy: 詞彙熵
- **Repetition Rate**: 重複率
- **Coherence**: 語義連貫性

## 量子優勢分析

1. **表達能力**: 量子態的疊加性提供更豐富的特徵表示
2. **並行計算**: 量子並行性在某些操作上可能提供優勢
3. **記憶效率**: 量子態可以用指數級更少的參數表示某些函數
4. **探索能力**: 量子隨機性有助於避免局部最優解

## 注意事項

1. **量子模擬限制**: 在經典計算機上模擬量子電路存在性能限制
2. **量子比特數量**: 受限於可用的量子比特數量，通常為4-8個
3. **訓練穩定性**: 量子梯度可能存在消失問題，需要小心調參
4. **數據集大小**: 適合小型數據集，大規模數據需要更多優化

## 未來改進

1. **更大規模實驗**: 在真實量子硬體上運行
2. **混合算法**: 開發更好的經典-量子混合策略
3. **新的量子編碼**: 探索更有效的量子特徵編碼方法
4. **應用擴展**: 擴展到其他NLP任務如翻譯、摘要等

## 貢獻指南

歡迎提交Issue和Pull Request來改進專案：

1. Fork專案
2. 創建特性分支
3. 提交更改
4. 發起Pull Request

## 授權

本專案採用MIT授權協議。

## 聯繫方式

如有問題或建議，請通過Issue或郵件聯繫。

---

*本專案旨在探索量子計算在自然語言處理領域的應用可能性，為量子機器學習研究提供實用工具。*