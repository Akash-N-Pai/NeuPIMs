#!/bin/bash

# Run MoE simulation with increased HBM buffer and smaller batch size
# This avoids running out of activation memory when processing many experts

echo "========================================="
echo "Running GPT-3 7B with MoE (64 experts)"
echo "Configuration:"
echo "  - Batch size: 64 (reduced from 512)"
echo "  - HBM activation buffer: 4 GB (increased from 512 MB)"
echo "  - Experts: 64, top-2 routing"
echo "  - Load imbalance: enabled (80% skew)"
echo "========================================="
echo ""

./build/bin/Simulator \
    --npu_config=./configs/npu_configs/systolic_ws_128x128_dev.json \
    --memory_config=./configs/memory_configs/neupims_moe.json \
    --client_config=./request-traces/clb/share-gpt2-bs64-ms7B-tp4-clb-0.csv \
    --model_config=./configs/model_configs/gpt3-7B-moe.json \
    --system_config=./configs/system_configs/sub-batch-off-moe.json \
    --log_dir=./experiment_logs/MoE_$(date +%Y-%m-%d_%H:%M:%S)

echo ""
echo "========================================="
echo "MoE simulation complete!"
echo "Check logs in ./experiment_logs/"
echo "========================================="

