#!/bin/bash

# 量子文字生成模型完整評估腳本
# 包含所有三個模型: QKSAN, QRWKV, QASA

set -e

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}🚀 量子文字生成模型完整評估系統${NC}"
echo -e "${CYAN}====================================${NC}"
echo -e "${GREEN}✅ 包含所有三個模型: QKSAN, QRWKV, QASA${NC}"
echo -e "${GREEN}✅ 包含完整評估指標: 困惑度、BLEU、多樣性、重複率、流暢度${NC}"
echo -e "${YELLOW}⚠️  強制使用CPU以避免CUDA錯誤${NC}"
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

# 運行完整評估
echo -e "${BLUE}🚀 開始運行完整評估...${NC}"
PYTHONPATH=/media/meow/Transcend/Quantum_text_gen python run_complete_evaluation.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ 完整評估完成!${NC}"
    echo -e "${BLUE}📁 結果保存在: ./experiments/comprehensive/${NC}"
else
    echo -e "${RED}❌ 評估失敗${NC}"
    exit 1
fi
