#pragma once
#include "../tensor/NPUTensor.h"
#include "Operation.h"

/**
 * ExpertParamLoad: Models parameter transfer from HBM to NPU SRAM
 * 
 * For MoE with off-chip experts:
 * - Experts stored in HBM (too large for on-chip storage)
 * - Each expert loaded one at a time via interconnect
 * - Parameter movement overhead = 2 × E × d_model × d_f
 *   where E = num_experts, d_model = model_n_embd, d_f = ffn_hidden_dim
 * 
 * For single expert:
 * - FC1 weights: [d_model, d_f] = [4096, 4096] = 16M parameters
 * - FC2 weights: [d_f, d_model] = [4096, 4096] = 16M parameters
 * - Total: 32M parameters × 2 bytes = 64MB per expert
 * 
 * Transfer time depends on interconnect bandwidth and latency
 */
class ExpertParamLoad : public Operation {
   public:
    ExpertParamLoad(std::string name, uint32_t expert_id, 
                    std::vector<Ptr<NPUTensor>> expert_weights,
                    Ptr<BTensor> data_tensor);  // The actual data to pass through

    std::vector<Ptr<BTensor>> get_outputs(std::vector<Ptr<BTensor>> inputs) override;

   private:
    uint32_t _expert_id;
    uint64_t _param_size_bytes;  // Total parameter size to load
    uint32_t _load_cycles;       // Cycles needed for transfer
    Ptr<BTensor> _data_tensor;   // The data tensor to pass through (normalized_input)
    std::vector<Ptr<NPUTensor>> _expert_weights;  // Store expert weights
    
    void calculate_load_cycles();
    void initialize_tiles();
    Tile initialize_instructions();
};

