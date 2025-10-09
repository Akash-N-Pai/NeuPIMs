#include "MoERoutingTraceReader.h"

MoERoutingTraceReader::MoERoutingTraceReader(std::string trace_path, uint32_t num_experts,
                                             uint32_t experts_per_token, uint32_t batch_size)
    : _trace_path(trace_path),
      _num_experts(num_experts),
      _experts_per_token(experts_per_token),
      _batch_size(batch_size),
      _has_trace(false) {
    
    _has_trace = load_trace();
    
    if (_has_trace) {
        spdlog::info("✓ Loaded MoE routing trace from: {}", trace_path);
    } else {
        spdlog::info("✗ No routing trace found at: {}, using simulated distribution", trace_path);
    }
}

bool MoERoutingTraceReader::load_trace() {
    std::ifstream trace_file(_trace_path);
    if (!trace_file.is_open()) {
        return false;
    }
    
    std::string line;
    // Skip header
    if (!std::getline(trace_file, line)) {
        return false;
    }
    
    // Parse CSV rows
    while (std::getline(trace_file, line)) {
        std::istringstream iss(line);
        std::string cell;
        std::vector<std::string> row;
        
        while (std::getline(iss, cell, ',')) {
            row.push_back(cell);
        }
        
        if (row.size() < 2 + _num_experts) {
            spdlog::warn("Invalid routing trace row (expected {} columns, got {})", 
                         2 + _num_experts, row.size());
            continue;
        }
        
        uint32_t layer_id = std::stoul(row[0]);
        uint32_t token_id = std::stoul(row[1]);
        
        // Parse expert probabilities
        std::vector<double> expert_probs(_num_experts);
        for (uint32_t i = 0; i < _num_experts; ++i) {
            expert_probs[i] = std::stod(row[2 + i]);
        }
        
        // Initialize layer if needed
        if (_routing_probs.find(layer_id) == _routing_probs.end()) {
            _routing_probs[layer_id].resize(_batch_size);
        }
        
        if (token_id < _batch_size) {
            _routing_probs[layer_id][token_id] = expert_probs;
        }
    }
    
    trace_file.close();
    
    spdlog::info("Loaded routing probabilities for {} layers", _routing_probs.size());
    return !_routing_probs.empty();
}

void MoERoutingTraceReader::compute_assignments(uint32_t layer_id) {
    if (_expert_token_counts.find(layer_id) != _expert_token_counts.end()) {
        return;  // Already computed
    }
    
    // Initialize counters
    std::vector<uint32_t> token_counts(_num_experts, 0);
    std::vector<std::vector<uint32_t>> token_assignments(_num_experts);
    
    if (!_has_trace || _routing_probs.find(layer_id) == _routing_probs.end()) {
        spdlog::warn("No routing data for layer {}", layer_id);
        _expert_token_counts[layer_id] = token_counts;
        _expert_token_assignments[layer_id] = token_assignments;
        return;
    }
    
    // For each token, select top-k experts
    for (uint32_t token_id = 0; token_id < _batch_size; ++token_id) {
        if (token_id >= _routing_probs[layer_id].size()) {
            spdlog::warn("Missing routing data for layer {} token {}", layer_id, token_id);
            continue;
        }
        
        auto& probs = _routing_probs[layer_id][token_id];
        
        // Find top-k experts by probability
        std::vector<std::pair<double, uint32_t>> prob_expert_pairs;
        for (uint32_t i = 0; i < _num_experts; ++i) {
            prob_expert_pairs.push_back({probs[i], i});
        }
        
        // Sort by probability (descending)
        std::sort(prob_expert_pairs.begin(), prob_expert_pairs.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // Select top-k
        for (uint32_t k = 0; k < _experts_per_token && k < _num_experts; ++k) {
            uint32_t expert_id = prob_expert_pairs[k].second;
            token_counts[expert_id]++;
            token_assignments[expert_id].push_back(token_id);
        }
    }
    
    _expert_token_counts[layer_id] = token_counts;
    _expert_token_assignments[layer_id] = token_assignments;
}

std::vector<uint32_t> MoERoutingTraceReader::get_expert_token_counts(uint32_t layer_id) {
    compute_assignments(layer_id);
    return _expert_token_counts[layer_id];
}

std::vector<std::vector<uint32_t>> MoERoutingTraceReader::get_expert_token_assignments(uint32_t layer_id) {
    compute_assignments(layer_id);
    return _expert_token_assignments[layer_id];
}

void MoERoutingTraceReader::print_distribution(uint32_t layer_id) {
    compute_assignments(layer_id);
    
    auto& token_counts = _expert_token_counts[layer_id];
    
    spdlog::info("========== MoE Token Distribution (Layer {}) ==========", layer_id);
    spdlog::info("Source: {}", _has_trace ? "Routing Trace File" : "Simulated");
    
    uint32_t total_assignments = 0;
    uint32_t min_tokens = _batch_size;
    uint32_t max_tokens = 0;
    
    for (uint32_t i = 0; i < _num_experts; ++i) {
        total_assignments += token_counts[i];
        min_tokens = std::min(min_tokens, token_counts[i]);
        max_tokens = std::max(max_tokens, token_counts[i]);
    }
    
    double avg_tokens = (double)total_assignments / _num_experts;
    
    spdlog::info("Total token-expert assignments: {}", total_assignments);
    spdlog::info("Average tokens per expert: {:.1f}", avg_tokens);
    spdlog::info("Min tokens: {} | Max tokens: {}", min_tokens, max_tokens);
    spdlog::info("Load imbalance ratio: {:.2f}x", max_tokens / avg_tokens);
    
    spdlog::info("Expert token counts:");
    for (uint32_t i = 0; i < std::min(10u, _num_experts); ++i) {
        spdlog::info("  Expert {:2d}: {:3d} tokens ({:5.2f}%)", 
                     i, token_counts[i], 
                     100.0 * token_counts[i] / total_assignments);
    }
    
    spdlog::info("========================================================");
}

