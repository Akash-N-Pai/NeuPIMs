#include "MoETokenDispatcher.h"

MoETokenDispatcher::MoETokenDispatcher(uint32_t num_experts, uint32_t experts_per_token,
                                       uint32_t batch_size, bool enable_imbalance, 
                                       double skew_factor)
    : _num_experts(num_experts),
      _experts_per_token(experts_per_token),
      _batch_size(batch_size),
      _enable_imbalance(enable_imbalance),
      _skew_factor(skew_factor) {
    
    _expert_token_counts.resize(num_experts, 0);
    _expert_token_assignments.resize(num_experts);
    
    generate_assignments();
}

void MoETokenDispatcher::generate_assignments() {
    if (_enable_imbalance) {
        generate_skewed_distribution();
    } else {
        generate_uniform_distribution();
    }
}

void MoETokenDispatcher::generate_uniform_distribution() {
    // Simple uniform distribution: round-robin assignment
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    
    for (uint32_t token_id = 0; token_id < _batch_size; ++token_id) {
        // Select top-k experts randomly (uniform)
        std::vector<uint32_t> available_experts(_num_experts);
        std::iota(available_experts.begin(), available_experts.end(), 0);
        std::shuffle(available_experts.begin(), available_experts.end(), gen);
        
        for (uint32_t k = 0; k < _experts_per_token; ++k) {
            uint32_t expert_id = available_experts[k];
            _expert_token_counts[expert_id]++;
            _expert_token_assignments[expert_id].push_back(token_id);
        }
    }
}

void MoETokenDispatcher::generate_skewed_distribution() {
    // Realistic skewed distribution (Pareto principle)
    // Top 5% of experts handle skew_factor% of tokens
    
    uint32_t top_experts_count = std::max(1u, _num_experts * 5 / 100);  // Top 5%
    uint32_t total_token_assignments = _batch_size * _experts_per_token;
    uint32_t tokens_to_top = total_token_assignments * _skew_factor;  // 80% to top 5%
    uint32_t tokens_to_rest = total_token_assignments - tokens_to_top;
    
    std::random_device rd;
    std::mt19937 gen(42);
    
    spdlog::info("MoE Load Distribution:");
    spdlog::info("  Top {}% experts ({}): handle {}% of assignments", 
                 5, top_experts_count, (int)(_skew_factor * 100));
    spdlog::info("  Remaining experts ({}): handle {}% of assignments",
                 _num_experts - top_experts_count, (int)((1.0 - _skew_factor) * 100));
    
    // Create probability distribution (Zipf-like)
    std::vector<double> expert_probs(_num_experts);
    
    // Top experts get higher probability
    for (uint32_t i = 0; i < top_experts_count; ++i) {
        expert_probs[i] = 1.0 / (i + 1);  // Zipf: 1, 1/2, 1/3, ...
    }
    
    // Remaining experts get lower probability
    for (uint32_t i = top_experts_count; i < _num_experts; ++i) {
        expert_probs[i] = 1.0 / (i + 1) * 0.2;  // Much lower
    }
    
    // Normalize probabilities
    double sum = std::accumulate(expert_probs.begin(), expert_probs.end(), 0.0);
    for (auto &prob : expert_probs) {
        prob /= sum;
    }
    
    // Assign tokens to experts based on probabilities
    std::discrete_distribution<> dist(expert_probs.begin(), expert_probs.end());
    
    for (uint32_t token_id = 0; token_id < _batch_size; ++token_id) {
        // Select top-k experts based on learned routing (simulated with distribution)
        std::set<uint32_t> selected_experts;
        
        while (selected_experts.size() < _experts_per_token) {
            uint32_t expert_id = dist(gen);
            selected_experts.insert(expert_id);
        }
        
        for (uint32_t expert_id : selected_experts) {
            _expert_token_counts[expert_id]++;
            _expert_token_assignments[expert_id].push_back(token_id);
        }
    }
}

std::vector<uint32_t> MoETokenDispatcher::get_expert_token_counts() {
    return _expert_token_counts;
}

std::vector<std::vector<uint32_t>> MoETokenDispatcher::get_expert_token_assignments() {
    return _expert_token_assignments;
}

void MoETokenDispatcher::print_distribution() {
    spdlog::info("========== MoE Token Distribution ==========");
    
    uint32_t total_assignments = 0;
    uint32_t min_tokens = _batch_size;
    uint32_t max_tokens = 0;
    
    for (uint32_t i = 0; i < _num_experts; ++i) {
        total_assignments += _expert_token_counts[i];
        min_tokens = std::min(min_tokens, _expert_token_counts[i]);
        max_tokens = std::max(max_tokens, _expert_token_counts[i]);
    }
    
    double avg_tokens = (double)total_assignments / _num_experts;
    
    spdlog::info("Total token-expert assignments: {}", total_assignments);
    spdlog::info("Average tokens per expert: {:.1f}", avg_tokens);
    spdlog::info("Min tokens: {} | Max tokens: {}", min_tokens, max_tokens);
    spdlog::info("Load imbalance ratio: {:.2f}x", max_tokens / avg_tokens);
    
    // Show distribution histogram
    spdlog::info("Expert token counts:");
    for (uint32_t i = 0; i < std::min(10u, _num_experts); ++i) {
        spdlog::info("  Expert {:2d}: {:3d} tokens ({:5.2f}%)", 
                     i, _expert_token_counts[i], 
                     100.0 * _expert_token_counts[i] / total_assignments);
    }
    if (_num_experts > 10) {
        spdlog::info("  ... ({} more experts)", _num_experts - 10);
        // Show last few (least loaded)
        for (uint32_t i = _num_experts - 3; i < _num_experts; ++i) {
            spdlog::info("  Expert {:2d}: {:3d} tokens ({:5.2f}%)", 
                         i, _expert_token_counts[i], 
                         100.0 * _expert_token_counts[i] / total_assignments);
        }
    }
    
    spdlog::info("============================================");
}

double MoETokenDispatcher::get_load_imbalance_ratio() {
    if (_expert_token_counts.empty()) return 1.0;
    
    uint32_t max_tokens = *std::max_element(_expert_token_counts.begin(), _expert_token_counts.end());
    uint32_t total = std::accumulate(_expert_token_counts.begin(), _expert_token_counts.end(), 0u);
    double avg_tokens = (double)total / _num_experts;
    
    return max_tokens / avg_tokens;
}

