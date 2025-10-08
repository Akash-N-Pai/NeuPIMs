#pragma once
#include "../tensor/NPUTensor.h"
#include "Operation.h"

/**
 * MoEExpert: Single expert FFN computation
 * 
 * Each expert is a standard FFN: FC1 -> GELU -> FC2
 * 
 * Input: [num_tokens_assigned, E] - tokens routed to this expert
 * Output: [num_tokens_assigned, E] - expert output
 * 
 * Weights:
 *   - fc1_weight: [E, 4E]
 *   - fc1_bias: [4E]
 *   - fc2_weight: [4E, E]
 *   - fc2_bias: [E]
 */
class MoEExpert : public Operation {
   public:
    MoEExpert(std::string name, uint32_t expert_id, std::vector<Ptr<NPUTensor>> weights);

    std::vector<Ptr<BTensor>> get_outputs(std::vector<Ptr<BTensor>> inputs) override;

   private:
    uint32_t _expert_id;
    uint32_t _batch_size;  // number of tokens assigned to this expert
    
    std::vector<uint32_t> _input_dim;
    std::vector<uint32_t> _inner_loop;
    std::vector<uint32_t> _outer_loop;

    void calculate_loops();
    void initialize_tiles();
    Tile initialize_instructions(uint32_t batch_idx);
    uint32_t sram_size_needed();
};

