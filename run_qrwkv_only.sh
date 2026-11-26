#!/bin/bash

# QRWKV模型專用運行腳本

set -e

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 默認配置
EPOCHS=${1:-50}
BATCH_SIZE=${2:-16}
MAX_LENGTH=${3:-32}
VOCAB_SIZE=${4:-500}
MAX_SAMPLES=${5:-1000}
LEARNING_RATE=${6:-0.001}

echo -e "${CYAN}🚀 QRWKV模型專用評估系統${NC}"
echo -e "${CYAN}====================================${NC}"
echo -e "${GREEN}✅ 只運行QRWKV模型${NC}"
echo -e "${GREEN}✅ 包含完整評估指標: 困惑度、BLEU、多樣性、重複率、流暢度${NC}"
echo -e "${YELLOW}⚠️  強制使用CPU以避免CUDA錯誤${NC}"
echo ""
echo -e "${BLUE}📊 訓練配置:${NC}"
echo -e "${BLUE}  訓練輪數:${NC} $EPOCHS epochs"
echo -e "${BLUE}  批次大小:${NC} $BATCH_SIZE"
echo -e "${BLUE}  最大長度:${NC} $MAX_LENGTH"
echo -e "${BLUE}  詞彙大小:${NC} $VOCAB_SIZE"
echo -e "${BLUE}  最大樣本:${NC} $MAX_SAMPLES"
echo -e "${BLUE}  學習率:${NC} $LEARNING_RATE"
echo ""

# 檢查依賴
echo -e "${BLUE}🔍 檢查依賴...${NC}"

# 檢查Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python 未安裝${NC}"
    exit 1
fi

# 檢查必要的Python包
python -c "import torch" 2>/dev/null || {
    echo -e "${RED}❌ PyTorch 未安裝${NC}"
    exit 1
}

python -c "import pennylane" 2>/dev/null || {
    echo -e "${RED}❌ PennyLane 未安裝${NC}"
    exit 1
}

echo -e "${GREEN}✅ 依賴檢查通過${NC}"

# 設置環境變量
export EPOCHS
export BATCH_SIZE
export MAX_LENGTH
export VOCAB_SIZE
export MAX_SAMPLES
export LEARNING_RATE

# 運行QRWKV評估
echo -e "${BLUE}🚀 開始運行QRWKV評估...${NC}"
PYTHONPATH=/media/meow/Transcend/Quantum_text_gen python run_qrwkv_only.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ QRWKV評估完成!${NC}"
    echo -e "${BLUE}📁 結果保存在: ./experiments/comprehensive/qrwkv/${NC}"
else
    echo -e "${RED}❌ QRWKV評估失敗${NC}"
    exit 1
fi
