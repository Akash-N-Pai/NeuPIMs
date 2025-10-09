#pragma once

#include "Common.h"
#include <fstream>
#include <sstream>

/**
 * MoERoutingTraceReader: Reads expert routing assignments from trace file
 * 
 * Trace file format (CSV):
 *   layer_id,token_id,expert_0,expert_1,...,expert_N
 *   0,0,0.45,0.32,0.10,...
 *   0,1,0.12,0.55,0.15,...
 * 
 * If trace file doesn't exist, falls back to MoETokenDispatcher simulation.
 */
class MoERoutingTraceReader {
   public:
    MoERoutingTraceReader(std::string trace_path, uint32_t num_experts, 
                          uint32_t experts_per_token, uint32_t batch_size);
    
    // Check if trace file was successfully loaded
    bool has_trace() const { return _has_trace; }
    
    // Get token assignments for a specific layer
    std::vector<uint32_t> get_expert_token_counts(uint32_t layer_id);
    std::vector<std::vector<uint32_t>> get_expert_token_assignments(uint32_t layer_id);
    
    void print_distribution(uint32_t layer_id);
    
   private:
    bool _has_trace;
    uint32_t _num_experts;
    uint32_t _experts_per_token;
    uint32_t _batch_size;
    std::string _trace_path;
    
    // Map: layer_id -> token_id -> expert probabilities
    std::map<uint32_t, std::vector<std::vector<double>>> _routing_probs;
    
    // Cached assignments per layer
    std::map<uint32_t, std::vector<uint32_t>> _expert_token_counts;
    std::map<uint32_t, std::vector<std::vector<uint32_t>>> _expert_token_assignments;
    
    bool load_trace();
    void compute_assignments(uint32_t layer_id);
};

