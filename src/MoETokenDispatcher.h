#pragma once

#include "Common.h"
#include <random>
#include <algorithm>

/**
 * MoETokenDispatcher: Simulates realistic token-to-expert assignment
 * 
 * Models real MoE behavior where:
 * - Top 5% of experts handle 80% of tokens (load imbalance)
 * - Remaining experts handle remaining 20% of tokens
 * - Load skew is configurable
 */
class MoETokenDispatcher {
   public:
    MoETokenDispatcher(uint32_t num_experts, uint32_t experts_per_token, 
                       uint32_t batch_size, bool enable_imbalance, double skew_factor);
    
    // Get number of tokens assigned to each expert
    std::vector<uint32_t> get_expert_token_counts();
    
    // Get which tokens are assigned to which expert
    std::vector<std::vector<uint32_t>> get_expert_token_assignments();
    
    // Statistics
    void print_distribution();
    double get_load_imbalance_ratio();  // max_load / avg_load
    
   private:
    uint32_t _num_experts;
    uint32_t _experts_per_token;
    uint32_t _batch_size;
    bool _enable_imbalance;
    double _skew_factor;  // 0.8 = 80% of tokens go to top 5% experts
    
    std::vector<uint32_t> _expert_token_counts;
    std::vector<std::vector<uint32_t>> _expert_token_assignments;
    
    void generate_assignments();
    void generate_skewed_distribution();
    void generate_uniform_distribution();
};

