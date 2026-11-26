#!/bin/bash

# Transformer Benchmark Script
# 運行傳統Transformer模型在相同數據集上的benchmark

echo "🚀 開始Transformer Benchmark評估"
echo "=================================="

# 設置環境變量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 檢查CUDA可用性
if command -v nvidia-smi &> /dev/null; then
    echo "✅ CUDA可用"
    DEVICE="cuda"
else
    echo "⚠️  CUDA不可用，使用CPU"
    DEVICE="cpu"
fi

# 檢查依賴
echo "🔍 檢查依賴..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"

# 創建輸出目錄
mkdir -p experiments/comprehensive/transformer

# 運行Transformer benchmark
echo "🎯 運行Transformer benchmark..."
python run_transformer_benchmark.py

# 檢查結果
echo "📊 檢查結果..."
if [ -d "experiments/comprehensive/transformer" ]; then
    echo "✅ Transformer結果目錄已創建"
    ls -la experiments/comprehensive/transformer/
else
    echo "❌ Transformer結果目錄未找到"
fi

echo "🎉 Transformer benchmark完成！"
