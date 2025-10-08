#!/bin/bash

echo "=========================================="
echo "  NeuPIMs: Dense vs MoE Comparison"
echo "=========================================="
echo ""

# Build first
echo "Building..."
cd build; make -j > /dev/null 2>&1; cd ..;

# Configuration files
config=./configs/systolic_ws_128x128_dev.json
mem_config=./configs/memory_configs/neupims.json
sys_config=./configs/system_configs/sub-batch-off.json
cli_config=./request-traces/clb/share-gpt2-bs512-ms7B-tp4-clb-0.csv

DATE=$(date "+%F_%H:%M:%S")

# Run Dense FFN
echo "Running Dense FFN baseline..."
model_config_dense=./configs/model_configs/gpt3-7B.json
LOG_DIR_DENSE=experiment_logs/Dense_${DATE}
mkdir -p $LOG_DIR_DENSE

./build/bin/Simulator \
    --config $config \
    --mem_config $mem_config \
    --cli_config $cli_config \
    --model_config $model_config_dense \
    --sys_config $sys_config \
    --log_dir $LOG_DIR_DENSE > /dev/null 2>&1

echo "  ✓ Dense FFN completed"

# Run MoE FFN
echo "Running MoE FFN (2 of 8 experts)..."
model_config_moe=./configs/model_configs/gpt3-7B-moe.json
LOG_DIR_MOE=experiment_logs/MoE_${DATE}
mkdir -p $LOG_DIR_MOE

./build/bin/Simulator \
    --config $config \
    --mem_config $mem_config \
    --cli_config $cli_config \
    --model_config $model_config_moe \
    --sys_config $sys_config \
    --log_dir $LOG_DIR_MOE > /dev/null 2>&1

echo "  ✓ MoE FFN completed"
echo ""

# Extract and compare results
echo "=========================================="
echo "  Performance Comparison"
echo "=========================================="
echo ""

echo "Stage Cycles:"
echo "-------------"
echo "Dense FFN:"
grep "Stage" $LOG_DIR_DENSE/_summary.tsv 2>/dev/null | tail -n +2 || echo "  No data"
echo ""
echo "MoE FFN:"
grep "Stage" $LOG_DIR_MOE/_summary.tsv 2>/dev/null | tail -n +2 || echo "  No data"
echo ""

echo "Memory Bandwidth:"
echo "-----------------"
echo "Dense FFN:"
grep "DRAM: AVG BW" $LOG_DIR_DENSE/*.txt 2>/dev/null | tail -1 || echo "  No data"
echo ""
echo "MoE FFN:"
grep "DRAM: AVG BW" $LOG_DIR_MOE/*.txt 2>/dev/null | tail -1 || echo "  No data"
echo ""

echo "Total Cycles:"
echo "-------------"
echo "Dense FFN:"
grep "DRAM total cycles" $LOG_DIR_DENSE/*.txt 2>/dev/null | tail -1 || echo "  No data"
echo ""
echo "MoE FFN:"
grep "DRAM total cycles" $LOG_DIR_MOE/*.txt 2>/dev/null | tail -1 || echo "  No data"
echo ""

echo "=========================================="
echo "  Logs saved to:"
echo "  Dense: $LOG_DIR_DENSE"
echo "  MoE:   $LOG_DIR_MOE"
echo "=========================================="

