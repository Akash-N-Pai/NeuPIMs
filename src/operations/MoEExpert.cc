#include "MoEExpert.h"

MoEExpert::MoEExpert(std::string name, uint32_t expert_id, std::vector<Ptr<NPUTensor>> weights) 
    : Operation(name), _expert_id(expert_id) {
    assert(weights.size() == 4);  // fc1_weight, fc1_bias, fc2_weight, fc2_bias
    _inputs.resize(5);
    _inputs[1] = weights[0];  // fc1_weight
    _inputs[2] = weights[1];  // fc1_bias
    _inputs[3] = weights[2];  // fc2_weight
    _inputs[4] = weights[3];  // fc2_bias
}

std::vector<Ptr<BTensor>> MoEExpert::get_outputs(std::vector<Ptr<BTensor>> inputs) {
    set_as_parent_tensor(inputs);
    
    assert(inputs.size() == 1);
    _inputs[0] = inputs[0];
    
    _input_dim = inputs[0]->get_dims();
    _batch_size = 1;
    for (size_t i = 0; i + 1 < _input_dim.size(); i++) {
        _batch_size *= _input_dim[i];
    }
    
    // Output: same shape as input [batch_size, E]
    _outputs.resize(1);
    _outputs[0] = std::make_shared<NPUTensor>(
        _name + "_output", 
        _input_dim, 
        NPUTensorBufType::ACT, false);
    
    // Create skip tile - expert computation is skipped for baseline
    // In a full implementation, this would execute FC1->GELU->FC2
    _tiles.push_back(Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = get_name(),
        .operation_id = _id,
        .batch = 0,
        .skip = true,
    });
    
    spdlog::info("MoEExpert {}: batch_size={} (skipped)", _expert_id, _batch_size);
    
    return _outputs;
}

void MoEExpert::initialize_tiles() {
    // Already created in get_outputs
}

Tile MoEExpert::initialize_instructions(uint32_t batch_idx) {
    // Not used - skip tile created directly
    return Tile{};
}

void MoEExpert::calculate_loops() {
    // Not needed for skip tile
}

uint32_t MoEExpert::sram_size_needed() {
    return 0;
}
