cd build; make; cd ..;

# config file
config=./configs/systolic_ws_128x128_dev.json
mem_config=./configs/memory_configs/neupims.json
model_config=./configs/model_configs/gpt3-7B-moe.json
sys_config=./configs/system_configs/sub-batch-off.json
cli_config=./request-traces/clb/share-gpt2-bs512-ms7B-tp4-clb-0.csv

# log file
LOG_LEVEL=info
DATE=$(date "+%F_%H:%M:%S")

LOG_DIR=experiment_logs/${DATE}

mkdir -p $LOG_DIR;
LOG_NAME=simulator.log
CONFIG_FILE=${LOG_DIR}/config.log
TERMINAL_LOG=${LOG_DIR}/terminal_output.log

echo "log directory: $LOG_DIR"



# Capture all terminal output (stdout + stderr) to file
# Only save content from "Stage E" onwards
./build/bin/Simulator \
    --config $config \
    --mem_config $mem_config \
    --cli_config $cli_config \
    --model_config $model_config \
    --sys_config $sys_config \
    --log_dir $LOG_DIR \
    2>&1 | tee >(sed -n '/----------Stage E----------/,$p' > ${TERMINAL_LOG})


echo "memory config: $mem_config" > ${CONFIG_FILE}
echo "client config: $cli_config" >> ${CONFIG_FILE}
echo "model config: $model_config" >> ${CONFIG_FILE}
echo "system config: $sys_config" >> ${CONFIG_FILE}
cat ${CONFIG_FILE}

# Generate execution summary
echo ""
echo "=========================================="
echo "Terminal output saved to: ${TERMINAL_LOG}"
echo "=========================================="
echo ""

# Analyze MoE expert execution if MoE is enabled
if grep -q "moe_enabled.*true" $model_config; then
    echo "MoE Execution Analysis:"
    echo "----------------------"
    
    # Count expert operations
    expert_fc1_count=$(grep -c "moe_expert.*\.fc1" ${LOG_DIR}/SA_stage_E.tsv 2>/dev/null || echo "0")
    expert_fc2_count=$(grep -c "moe_expert.*\.fc2" ${LOG_DIR}/SA_stage_E.tsv 2>/dev/null || echo "0")
    expert_gelu_count=$(grep -c "moe_expert.*\.gelu" ${LOG_DIR}/SA_stage_E.tsv 2>/dev/null || echo "0")
    
    echo "  Expert FC1 operations:  $expert_fc1_count"
    echo "  Expert FC2 operations:  $expert_fc2_count"
    echo "  Expert GELU operations: $expert_gelu_count"
    
    # Check for overlapping execution (parallel) vs sequential
    if [ -f "${LOG_DIR}/SA_stage_E.tsv" ]; then
        echo ""
        echo "  First 5 experts FC1 timing:"
        grep "moe_expert\.[0-4]\.fc1" ${LOG_DIR}/SA_stage_E.tsv | awk -F'\t' '{print "    "$1": Start="$2" End="$3}' 2>/dev/null || echo "    (data not available)"
    fi
    
    echo ""
fi