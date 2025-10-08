#!/bin/bash

cd build; make; cd ..;

# MoE configuration
config=./configs/systolic_ws_128x128_dev.json
mem_config=./configs/memory_configs/neupims.json
model_config=./configs/model_configs/gpt3-7B-moe.json
sys_config=./configs/system_configs/sub-batch-off.json
cli_config=./request-traces/clb/share-gpt2-bs512-ms7B-tp4-clb-0.csv

# log file
LOG_LEVEL=info
DATE=$(date "+%F_%H:%M:%S")

LOG_DIR=experiment_logs/MoE_${DATE}

mkdir -p $LOG_DIR;
LOG_NAME=simulator.log
CONFIG_FILE=${LOG_DIR}/config.log

echo "Testing MoE implementation..."
echo "log directory: $LOG_DIR"

./build/bin/Simulator \
    --config $config \
    --mem_config $mem_config \
    --cli_config $cli_config \
    --model_config $model_config \
    --sys_config $sys_config \
    --log_dir $LOG_DIR

echo "memory config: $mem_config" > ${CONFIG_FILE}
echo "client config: $cli_config" >> ${CONFIG_FILE}
echo "model config: $model_config" >> ${CONFIG_FILE}
echo "system config: $sys_config" >> ${CONFIG_FILE}
cat ${CONFIG_FILE}

echo ""
echo "MoE Test Results:"
grep "MoE enabled" ${LOG_DIR}/* 2>/dev/null || echo "MoE not enabled"
grep "Total weight size" ${LOG_DIR}/* 2>/dev/null
grep "Stage E" ${LOG_DIR}/* 2>/dev/null

