#pragma once
#include "../tensor/NPUTensor.h"
#include "Operation.h"

/**
 * MoECombine: Combines outputs from multiple experts using routing weights
 * 
 * Inputs:
 *   - routing_weights: [batch_size, experts_per_token] - normalized weights
 *   - expert_indices: [batch_size, experts_per_token] - expert assignments
 *   - expert_outputs: [num_experts] tensors, each [num_tokens_assigned, E]
 * 
 * Output: [batch_size, E] - weighted combination of expert outputs
 * 
 * Operation:
 *   For each token:
 *     output[token] = sum(weight[i] * expert_output[expert_idx[i]][token] 
 *                         for i in experts_per_token)
 */
class MoECombine : public Operation {
   public:
    MoECombine(std::string name, uint32_t num_experts, uint32_t experts_per_token);

    std::vector<Ptr<BTensor>> get_outputs(std::vector<Ptr<BTensor>> inputs) override;

   private:
    uint32_t _batch_size;
    uint32_t _num_experts;
    uint32_t _experts_per_token;
    
    std::vector<uint32_t> _input_dim;
    std::vector<uint32_t> _inner_loop;
    std::vector<uint32_t> _outer_loop;

    void calculate_loops();
    void initialize_tiles();
    Tile initialize_instructions(uint32_t batch_idx);
    uint32_t sram_size_needed();
};

