#!/bin/bash

# 快速比較腳本 - 包含Transformer baseline
set -e

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
NC='\033[0m'

# 快速測試配置
EPOCHS=${1:-20}
BATCH_SIZE=${2:-8}
MAX_LENGTH=${3:-16}
VOCAB_SIZE=${4:-500}
MAX_SAMPLES=${5:-50}
LEARNING_RATE=${6:-0.001}

echo -e "${PURPLE}🚀 快速比較: Transformer Baseline vs 量子模型${NC}"
echo -e "${PURPLE}==============================================${NC}"
echo -e "${GREEN}✅ 包含所有四個模型: QKSAN, QRWKV, QASA, TRANSFORMER${NC}"
echo -e "${CYAN}🎯 快速測試配置，只評估simple_sentences數據集${NC}"
echo -e "${YELLOW}⚡ 快速驗證Transformer baseline效果${NC}"

# 檢查依賴
echo -e "${BLUE}🔍 檢查依賴...${NC}"
python -c "import torch" 2>/dev/null || { echo -e "${RED}❌ PyTorch 未安裝${NC}"; exit 1; }
python -c "import pennylane" 2>/dev/null || { echo -e "${RED}❌ PennyLane 未安裝${NC}"; exit 1; }
echo -e "${GREEN}✅ 依賴檢查通過${NC}"

# 設置環境變量
export EPOCHS BATCH_SIZE MAX_LENGTH VOCAB_SIZE MAX_SAMPLES LEARNING_RATE

# 運行快速比較
echo -e "${BLUE}🚀 開始運行快速比較...${NC}"
PYTHONPATH=/media/meow/Transcend/Quantum_text_gen python quick_transformer_comparison.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ 快速比較完成!${NC}"
    echo -e "${BLUE}📁 結果保存在: ./experiments/quick_comparison_*/${NC}"
else
    echo -e "${RED}❌ 快速比較失敗${NC}"
    exit 1
fi
