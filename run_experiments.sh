#!/bin/bash

# 主實驗腳本 - 支持多種評估模式
set -e

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
NC='\033[0m'

# 顯示幫助信息
show_help() {
    echo -e "${CYAN}🚀 量子文字生成模型實驗系統${NC}"
    echo -e "${CYAN}================================${NC}"
    echo -e "${GREEN}使用方法:${NC}"
    echo -e "  $0 [模式] [參數...]"
    echo ""
    echo -e "${BLUE}可用模式:${NC}"
    echo -e "  ${GREEN}quick${NC}     - 快速比較 (包含Transformer baseline)"
    echo -e "  ${GREEN}full${NC}      - 完整評估 (包含Transformer baseline)"
    echo -e "  ${GREEN}original${NC}  - 原始評估 (只有量子模型)"
    echo -e "  ${GREEN}test${NC}      - 測試Transformer模型"
    echo ""
    echo -e "${BLUE}參數說明:${NC}"
    echo -e "  EPOCHS      - 訓練輪數 (默認: 50)"
    echo -e "  BATCH_SIZE  - 批次大小 (默認: 16)"
    echo -e "  MAX_LENGTH  - 最大長度 (默認: 32)"
    echo -e "  VOCAB_SIZE  - 詞彙大小 (默認: 1000)"
    echo -e "  MAX_SAMPLES - 最大樣本 (默認: 100)"
    echo -e "  LEARNING_RATE - 學習率 (默認: 0.001)"
    echo ""
    echo -e "${YELLOW}示例:${NC}"
    echo -e "  $0 quick                    # 快速比較"
    echo -e "  $0 full 30 8 16 500 50 0.001  # 完整評估，自定義參數"
    echo -e "  $0 test                     # 測試Transformer模型"
}

# 檢查參數
if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
    exit 0
fi

MODE=$1
shift

# 根據模式設置默認參數
case $MODE in
    "quick")
        EPOCHS=${1:-20}
        BATCH_SIZE=${2:-8}
        MAX_LENGTH=${3:-16}
        VOCAB_SIZE=${4:-500}
        MAX_SAMPLES=${5:-50}
        LEARNING_RATE=${6:-0.001}
        ;;
    "full"|"original")
        EPOCHS=${1:-50}
        BATCH_SIZE=${2:-16}
        MAX_LENGTH=${3:-32}
        VOCAB_SIZE=${4:-1000}
        MAX_SAMPLES=${5:-100}
        LEARNING_RATE=${6:-0.001}
        ;;
    "test")
        echo -e "${BLUE}🧪 測試Transformer模型...${NC}"
        python test_transformer_baseline.py
        exit $?
        ;;
    *)
        echo -e "${RED}❌ 未知模式: $MODE${NC}"
        show_help
        exit 1
        ;;
esac

# 設置環境變量
export EPOCHS BATCH_SIZE MAX_LENGTH VOCAB_SIZE MAX_SAMPLES LEARNING_RATE

echo -e "${PURPLE}🚀 開始運行實驗: $MODE 模式${NC}"
echo -e "${BLUE}📊 配置: EPOCHS=$EPOCHS, BATCH_SIZE=$BATCH_SIZE, MAX_LENGTH=$MAX_LENGTH${NC}"
echo -e "${BLUE}📊 配置: VOCAB_SIZE=$VOCAB_SIZE, MAX_SAMPLES=$MAX_SAMPLES, LEARNING_RATE=$LEARNING_RATE${NC}"

# 根據模式運行不同的腳本
case $MODE in
    "quick")
        echo -e "${CYAN}🎯 運行快速比較...${NC}"
        ./run_quick_comparison.sh $EPOCHS $BATCH_SIZE $MAX_LENGTH $VOCAB_SIZE $MAX_SAMPLES $LEARNING_RATE
        ;;
    "full")
        echo -e "${CYAN}🎯 運行完整評估 (包含Transformer baseline)...${NC}"
        ./run_transformer_baseline.sh $EPOCHS $BATCH_SIZE $MAX_LENGTH $VOCAB_SIZE $MAX_SAMPLES $LEARNING_RATE
        ;;
    "original")
        echo -e "${CYAN}🎯 運行原始評估 (只有量子模型)...${NC}"
        ./run_all_models.sh $EPOCHS $BATCH_SIZE $MAX_LENGTH $VOCAB_SIZE $MAX_SAMPLES $LEARNING_RATE
        ;;
esac

echo -e "${GREEN}✅ 實驗完成!${NC}"
