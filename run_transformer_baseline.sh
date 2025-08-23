#!/bin/bash

# Transformer Baseline 評估腳本
# 專門用於運行包含Transformer baseline的完整評估

set -e

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
NC='\033[0m'

# 默認配置
EPOCHS=${1:-50}
BATCH_SIZE=${2:-16}
MAX_LENGTH=${3:-32}
VOCAB_SIZE=${4:-1000}
MAX_SAMPLES=${5:-100}
LEARNING_RATE=${6:-0.001}

echo -e "${PURPLE}🚀 Transformer Baseline 評估系統${NC}"
echo -e "${PURPLE}====================================${NC}"
echo -e "${GREEN}✅ 包含所有四個模型: QKSAN, QRWKV, QASA, TRANSFORMER${NC}"
echo -e "${GREEN}✅ 包含完整評估指標: 困惑度、BLEU、多樣性、重複率、流暢度${NC}"
echo -e "${CYAN}🎯 重點: Transformer作為傳統深度學習baseline${NC}"
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

# 檢查Transformer baseline文件
if [ ! -f "transformer_baseline.py" ]; then
    echo -e "${RED}❌ transformer_baseline.py 文件不存在${NC}"
    exit 1
fi

if [ ! -f "run_transformer_baseline_evaluation.py" ]; then
    echo -e "${RED}❌ run_transformer_baseline_evaluation.py 文件不存在${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Transformer baseline文件檢查通過${NC}"

# 設置環境變量
export EPOCHS BATCH_SIZE MAX_LENGTH VOCAB_SIZE MAX_SAMPLES LEARNING_RATE

# 運行Transformer baseline評估
echo -e "${BLUE}🚀 開始運行Transformer Baseline評估...${NC}"
echo -e "${CYAN}📊 這將包含所有四個模型的完整比較${NC}"
echo ""

PYTHONPATH=/media/meow/Transcend/Quantum_text_gen python run_transformer_baseline_evaluation.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Transformer Baseline評估完成!${NC}"
    echo -e "${BLUE}📁 結果保存在: ./experiments/comprehensive/comprehensive_report_with_transformer_*/${NC}"
    echo ""
    echo -e "${PURPLE}🎯 評估完成，現在可以查看詳細報告${NC}"
    echo -e "${PURPLE}📄 報告包含量子模型與Transformer baseline的詳細比較${NC}"
else
    echo -e "${RED}❌ 評估失敗${NC}"
    exit 1
fi

echo ""
echo -e "${CYAN}====================================${NC}"
echo -e "${GREEN}🎉 Transformer Baseline評估流程完成!${NC}"
echo -e "${CYAN}====================================${NC}"
