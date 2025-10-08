#pragma once

#include "Common.h"

/**
 * MoE-specific statistics tracking
 */
namespace MoEStats {

struct ExpertUtilization {
    uint32_t expert_id;
    uint32_t tokens_processed;
    uint64_t compute_cycles;
    uint64_t memory_accesses;
    double load_percentage;  // tokens_processed / total_tokens
};

struct MoELayerStat {
    std::string layer_name;
    uint32_t num_experts;
    uint32_t experts_per_token;
    uint32_t total_tokens;
    
    // Router stats
    uint64_t router_cycles;
    uint64_t router_memory_reads;
    uint64_t router_memory_writes;
    
    // Expert stats
    std::vector<ExpertUtilization> expert_stats;
    double load_balance_variance;  // Variance in expert utilization
    
    // Combine stats
    uint64_t combine_cycles;
    
    // Total
    uint64_t total_moe_cycles;
    double avg_expert_utilization;
};

// Global MoE stats collector
extern std::vector<MoELayerStat> layer_stats;

void init();
void record_router_completion(std::string layer_name, uint64_t cycles);
void record_expert_completion(std::string layer_name, uint32_t expert_id, uint64_t cycles);
void record_combine_completion(std::string layer_name, uint64_t cycles);
void print_stats();
void log_stats(std::string log_dir);

}  // namespace MoEStats

