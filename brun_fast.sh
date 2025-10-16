#!/bin/bash

# FAST simulation mode - minimal logging
echo "=== FAST MODE: Building optimized binary ==="
cd build; make -j$(nproc); cd ..;

# config file
config=./configs/systolic_ws_128x128_dev.json
mem_config=./configs/memory_configs/neupims.json
model_config=./configs/model_configs/gpt3-7B-moe.json
sys_config=./configs/system_configs/sub-batch-off.json
cli_config=./request-traces/clb/share-gpt2-bs512-ms7B-tp4-clb-0.csv

# log file
LOG_LEVEL=info  # Use 'warn' or 'error' for even less logging
DATE=$(date "+%F_%H:%M:%S")
LOG_DIR=experiment_logs/${DATE}

mkdir -p $LOG_DIR;
CONFIG_FILE=${LOG_DIR}/config.log
TERMINAL_LOG=${LOG_DIR}/terminal_output.log

echo "log directory: $LOG_DIR"
echo "Log level: $LOG_LEVEL (use 'warn' for faster execution)"

# Run with minimal output
time ./build/bin/Simulator \
    --config $config \
    --mem_config $mem_config \
    --cli_config $cli_config \
    --model_config $model_config \
    --sys_config $sys_config \
    --log_dir $LOG_DIR \
    --log_level $LOG_LEVEL \
    2>&1 | tee >(sed -n '/----------Stage E----------/,$p' > ${TERMINAL_LOG})

echo "memory config: $mem_config" > ${CONFIG_FILE}
echo "client config: $cli_config" >> ${CONFIG_FILE}
echo "model config: $model_config" >> ${CONFIG_FILE}
echo "system config: $sys_config" >> ${CONFIG_FILE}
cat ${CONFIG_FILE}

echo ""
echo "=========================================="
echo "Terminal output saved to: ${TERMINAL_LOG}"
echo "Results in: ${LOG_DIR}/"
echo "=========================================="
echo ""

# Quick stats summary
if [ -f "${LOG_DIR}/SA_stage_E.tsv" ]; then
    echo "MoE Expert Operations:"
    echo "  FC1:  $(grep -c "moe_expert.*\.fc1" ${LOG_DIR}/SA_stage_E.tsv 2>/dev/null || echo 0)"
    echo "  FC2:  $(grep -c "moe_expert.*\.fc2" ${LOG_DIR}/SA_stage_E.tsv 2>/dev/null || echo 0)"
    echo "  GELU: $(grep -c "moe_expert.*\.gelu" ${LOG_DIR}/SA_stage_E.tsv 2>/dev/null || echo 0)"
    echo ""
    echo "First expert FC1 timing:"
    grep "moe_expert\.0\.fc1" ${LOG_DIR}/SA_stage_E.tsv | awk -F'\t' '{print "  Cycles: "$4}' 2>/dev/null || echo "  (not found)"
fi

