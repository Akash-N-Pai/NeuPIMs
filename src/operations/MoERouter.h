#pragma once
#include "../tensor/NPUTensor.h"
#include "Operation.h"

/**
 * MoERouter: Routes input tokens to experts using gating network
 * 
 * Input: [batch_size, E] - input tokens
 * Output: 
 *   - routing_weights: [batch_size, experts_per_token] - normalized weights for selected experts
 *   - expert_indices: [batch_size, experts_per_token] - indices of selected experts
 *   - token_to_expert_map: routing information for expert assignment
 * 
 * Operation:
 *   1. Compute logits: input × router_weight = [batch_size, num_experts]
 *   2. Apply softmax to get probabilities
 *   3. Select top-k experts per token
 *   4. Normalize selected expert weights
 */
class MoERouter : public Operation {
   public:
    MoERouter(std::string name, std::vector<Ptr<NPUTensor>> weights);

    std::vector<Ptr<BTensor>> get_outputs(std::vector<Ptr<BTensor>> inputs) override;

   private:
    uint32_t _batch_size;
    uint32_t _num_experts;
    uint32_t _experts_per_token;  // top-k
    
    std::vector<uint32_t> _input_dim;
    std::vector<uint32_t> _inner_loop;
    std::vector<uint32_t> _outer_loop;

    void calculate_loops();
    void initialize_tiles();
    Tile initialize_instructions(uint32_t batch_idx);
    uint32_t sram_size_needed();
};

