#pragma once

#include "Common.h"
#include "MoEExpertCache.h"
#include "MoETokenDispatcher.h"
#include <vector>

/**
 * MoEExecution: Manages parallel expert execution with optimizations
 * 
 * Features:
 * 1. Skip inactive experts (0 tokens)
 * 2. Parallel expert execution (wall-time = max, not sum)
 * 3. Expert caching (avoid redundant param loads)
 * 4. Per-expert batched GEMM (only process assigned tokens)
 * 5. Double buffering (overlap param load + compute)
 */
class MoEExecution {
   public:
    struct ExpertTask {
        uint32_t expert_id;
        uint32_t num_tokens;
        bool needs_param_load;  // False if cached
        uint64_t param_load_cycles;
        uint64_t compute_cycles;
        uint64_t total_cycles;  // param_load + compute (or just compute if cached)
    };
    
    MoEExecution(uint32_t num_experts, uint32_t expert_cache_size);
    
    // Plan expert execution for a batch
    std::vector<ExpertTask> plan_execution(
        const std::vector<uint32_t>& expert_token_counts,
        uint64_t param_load_cycles_per_expert,
        uint64_t compute_cycles_per_token);
    
    // Calculate total stage latency with parallelism
    uint64_t calculate_parallel_latency(const std::vector<ExpertTask>& tasks);
    
    // Calculate with double buffering optimization
    uint64_t calculate_double_buffered_latency(const std::vector<ExpertTask>& tasks);
    
    // Statistics
    void print_execution_plan(const std::vector<ExpertTask>& tasks);
    
   private:
    uint32_t _num_experts;
    MoEExpertCache _expert_cache;
    
    uint64_t _total_param_loads;
    uint64_t _total_compute_cycles;
    uint64_t _cache_saved_cycles;
};

