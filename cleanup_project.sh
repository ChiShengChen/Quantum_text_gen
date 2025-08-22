#!/bin/bash

# 量子文字生成專案清理腳本
# 只保留最新版本的文件

set -e

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}🧹 量子文字生成專案清理腳本${NC}"
echo -e "${CYAN}==============================${NC}"
echo ""

# 創建備份目錄
BACKUP_DIR="./backup_$(date +%Y%m%d_%H%M%S)"
echo -e "${YELLOW}📁 創建備份目錄: $BACKUP_DIR${NC}"
mkdir -p "$BACKUP_DIR"

# 備份舊文件
echo -e "${BLUE}💾 備份舊文件...${NC}"

# 備份舊的實驗腳本
mkdir -p "$BACKUP_DIR/old_scripts"
for file in run_*.py run_*.sh; do
    if [ -f "$file" ] && [ "$file" != "run_all_models.sh" ] && [ "$file" != "run_complete_evaluation.py" ]; then
        echo "備份: $file"
        mv "$file" "$BACKUP_DIR/old_scripts/"
    fi
done

# 備份舊的實驗腳本
for file in *experiment*.py *training*.py *monitor*.py; do
    if [ -f "$file" ] && [ "$file" != "run_complete_evaluation.py" ]; then
        echo "備份: $file"
        mv "$file" "$BACKUP_DIR/old_scripts/"
    fi
done

# 備份舊的測試腳本
mkdir -p "$BACKUP_DIR/test_scripts"
for file in test_*.py simple_*.py quick_*.py; do
    if [ -f "$file" ]; then
        echo "備份: $file"
        mv "$file" "$BACKUP_DIR/test_scripts/"
    fi
done

# 備份舊的修復腳本
mkdir -p "$BACKUP_DIR/fix_scripts"
for file in *fix*.py *enhanced*.py; do
    if [ -f "$file" ]; then
        echo "備份: $file"
        mv "$file" "$BACKUP_DIR/fix_scripts/"
    fi
done

# 備份舊的報告文件
mkdir -p "$BACKUP_DIR/reports"
for file in *REPORT*.md *SUMMARY*.md *GUIDE*.md; do
    if [ -f "$file" ]; then
        echo "備份: $file"
        mv "$file" "$BACKUP_DIR/reports/"
    fi
done

# 保留核心文件
echo -e "${GREEN}✅ 保留核心文件:${NC}"
echo "  - qksan_fixed.py (QKSAN模型)"
echo "  - qrwkv_model.py (QRWKV模型)"
echo "  - qasa_model.py (QASA模型)"
echo "  - data/ (數據集)"
echo "  - models/ (模型定義)"
echo "  - experiments/ (實驗結果)"

# 創建新的README
echo -e "${BLUE}📝 創建新的README...${NC}"
cat > README.md << 'EOF'
# 量子文字生成模型專案

## 🚀 快速開始

### 運行完整評估 (包含所有三個模型)
```bash
./run_all_models.sh
```

### 運行Python腳本
```bash
PYTHONPATH=/media/meow/Transcend/Quantum_text_gen python run_complete_evaluation.py
```

## 📊 支持的模型

1. **QKSAN** - Quantum-Kernel Self-Attention Network
2. **QRWKV** - Quantum RWKV
3. **QASA** - Quantum Attention Sequence Architecture

## 📈 評估指標

- **困惑度 (Perplexity)** - 語言模型預測能力
- **BLEU分數** - 文本生成質量
- **多樣性指標** - Distinct-1/2, Entropy, Vocabulary Size
- **重複率** - 文本重複程度
- **流暢度指標** - 句子長度統計

## 📁 專案結構

```
├── qksan_fixed.py          # QKSAN模型實現
├── qrwkv_model.py          # QRWKV模型實現
├── qasa_model.py           # QASA模型實現
├── run_complete_evaluation.py  # 完整評估腳本
├── run_all_models.sh       # 運行腳本
├── analyze_results.py      # 結果分析腳本
├── data/                   # 數據集
├── models/                 # 模型定義
└── experiments/            # 實驗結果
```

## 🔧 使用方法

1. **安裝依賴**:
   ```bash
   pip install torch pennylane numpy pandas matplotlib
   ```

2. **運行實驗**:
   ```bash
   ./run_all_models.sh
   ```

3. **分析結果**:
   ```bash
   python analyze_results.py
   ```

## 📊 最新實驗結果

查看 `./experiments/comprehensive/` 目錄中的最新實驗結果。

## 🛠️ 故障排除

### 常見問題

1. **ModuleNotFoundError**: 設置PYTHONPATH
   ```bash
   export PYTHONPATH=/media/meow/Transcend/Quantum_text_gen
   ```

2. **CUDA錯誤**: 腳本已強制使用CPU

3. **內存不足**: 減少批次大小和最大樣本數

## 📈 性能總結

基於最新實驗結果:
- **QRWKV**: 平均困惑度=1.55, 平均BLEU-1=0.080, 平均Distinct-1=0.440
- **QKSAN**: 平均困惑度=103.05, 平均BLEU-1=0.004, 平均Distinct-1=0.125
- **QASA**: 待評估

## 🎯 改進建議

1. 增加詞彙大小
2. 調整學習率
3. 使用更多訓練數據
4. 優化量子電路參數

---

*最後更新: 2025-08-23*
EOF

echo -e "${GREEN}✅ 專案清理完成!${NC}"
echo -e "${BLUE}📁 備份目錄: $BACKUP_DIR${NC}"
echo -e "${BLUE}📝 新的README已創建${NC}"
echo ""
echo -e "${CYAN}🎯 現在你可以使用以下命令運行完整評估:${NC}"
echo -e "${GREEN}./run_all_models.sh${NC}"
echo ""
echo -e "${YELLOW}⚠️  如果需要恢復舊文件，請查看備份目錄${NC}"
