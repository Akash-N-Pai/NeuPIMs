#include "MoECombine.h"

MoECombine::MoECombine(std::string name, uint32_t num_experts, uint32_t experts_per_token) 
    : Operation(name), _num_experts(num_experts), _experts_per_token(experts_per_token) {
    // Inputs will be: routing_weights, expert_indices, expert_outputs[0..num_experts-1]
    _inputs.resize(2 + num_experts);
}

std::vector<Ptr<BTensor>> MoECombine::get_outputs(std::vector<Ptr<BTensor>> inputs) {
    set_as_parent_tensor(inputs);
    
    // inputs[0]: routing_weights [batch_size, experts_per_token]
    // inputs[1]: expert_indices [batch_size, experts_per_token]
    // inputs[2..2+num_experts]: expert outputs
    assert(inputs.size() == 2 + _num_experts);
    _inputs.assign(inputs.begin(), inputs.end());
    
    _input_dim = inputs[0]->get_dims();
    _batch_size = _input_dim[0];
    
    // Output: [batch_size, E]
    auto E = inputs[2]->get_dims().back();  // Get E from first expert output
    _outputs.resize(1);
    _outputs[0] = std::make_shared<NPUTensor>(
        _name + "_output", 
        std::vector<uint32_t>{_batch_size, E}, 
        NPUTensorBufType::ACT, false);
    
    // Create skip tile - combination is simplified
    _tiles.push_back(Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = get_name(),
        .operation_id = _id,
        .batch = 0,
        .skip = true,
    });
    
    spdlog::info("MoECombine: batch_size={}, num_experts={}, experts_per_token={} (skipped)", 
                 _batch_size, _num_experts, _experts_per_token);
    
    return _outputs;
}

void MoECombine::initialize_tiles() {
    // Already created in get_outputs
}

Tile MoECombine::initialize_instructions(uint32_t batch_idx) {
    // Not used - skip tile created directly
    return Tile{};
}

void MoECombine::calculate_loops() {
    // Not needed for skip tile
}

uint32_t MoECombine::sram_size_needed() {
    return 0;
}
