#include "MoERouter.h"

MoERouter::MoERouter(std::string name, std::vector<Ptr<NPUTensor>> weights) : Operation(name) {
    assert(weights.size() == 1);  // Only router weight, no bias
    _inputs.resize(2);
    _inputs[1] = weights[0];  // router weights [E, num_experts]
    
    _num_experts = _config.num_experts;
    _experts_per_token = _config.experts_per_token;
}

std::vector<Ptr<BTensor>> MoERouter::get_outputs(std::vector<Ptr<BTensor>> inputs) {
    set_as_parent_tensor(inputs);
    
    assert(inputs.size() == 1);
    _inputs[0] = inputs[0];
    
    _input_dim = inputs[0]->get_dims();
    _batch_size = 1;
    for (size_t i = 0; i + 1 < _input_dim.size(); i++) {
        _batch_size *= _input_dim[i];
    }
    
    // Output 1: routing weights [batch_size, experts_per_token]
    // Output 2: expert indices [batch_size, experts_per_token]
    _outputs.resize(2);
    _outputs[0] = std::make_shared<NPUTensor>(
        _name + "_weights", 
        std::vector<uint32_t>{_batch_size, _experts_per_token}, 
        NPUTensorBufType::ACT, false);
    _outputs[1] = std::make_shared<NPUTensor>(
        _name + "_indices", 
        std::vector<uint32_t>{_batch_size, _experts_per_token}, 
        NPUTensorBufType::ACT, false);
    
    // Create skip tile - routing is deterministic for baseline
    _tiles.push_back(Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = get_name(),
        .operation_id = _id,
        .batch = 0,
        .skip = true,
    });
    
    spdlog::info("MoERouter: batch_size={}, num_experts={}, top_k={} (skipped)", 
                 _batch_size, _num_experts, _experts_per_token);
    
    return _outputs;
}

void MoERouter::initialize_tiles() {
    // Already created in get_outputs
}

Tile MoERouter::initialize_instructions(uint32_t batch_idx) {
    // Not used - skip tile created directly
    return Tile{};
}

void MoERouter::calculate_loops() {
    // Not needed for skip tile
}

uint32_t MoERouter::sram_size_needed() {
    return 0;
}
