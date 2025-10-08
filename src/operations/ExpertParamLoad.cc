#include "ExpertParamLoad.h"

ExpertParamLoad::ExpertParamLoad(std::string name, uint32_t expert_id,
                                 std::vector<Ptr<NPUTensor>> expert_weights)
    : Operation(name), _expert_id(expert_id) {
    
    // Calculate total parameter size for this expert
    // FC1 + FC2 weights (bias is small, ignored)
    _param_size_bytes = 0;
    for (auto weight : expert_weights) {
        _param_size_bytes += weight->_inners[0]->_size;
    }
    
    _inputs.resize(expert_weights.size() + 1);
    for (size_t i = 0; i < expert_weights.size(); ++i) {
        _inputs[i + 1] = expert_weights[i];
    }
    
    calculate_load_cycles();
}

void ExpertParamLoad::calculate_load_cycles() {
    // Parameter Movement Overhead
    // Based on: 2 × E × d_model × d_f (for one expert, E=1)
    // 
    // FC1: [d_model, d_f] parameters
    // FC2: [d_f, d_model] parameters
    // Total: 2 × d_model × d_f parameters
    
    uint32_t d_model = _config.model_n_embd;
    uint32_t d_f = 4 * d_model / _config.n_tp;  // FFN hidden dim with tensor parallelism
    
    uint64_t total_params = 2 * d_model * d_f;  // FC1 + FC2
    uint64_t param_bytes = total_params * _config.precision;
    
    // Transfer latency calculation
    // Assuming interconnect bandwidth and latency
    uint32_t icnt_bandwidth_gbps = 32;  // Example: PCIe Gen4 x16 = 32 GB/s per direction
    uint32_t icnt_freq_mhz = _config.icnt_freq;  // From config
    
    // Bytes per cycle at interconnect frequency
    double bytes_per_cycle = (double)icnt_bandwidth_gbps * 1e9 / (icnt_freq_mhz * 1e6);
    
    // Cycles for transfer at icnt frequency
    uint64_t icnt_cycles = (uint64_t)(param_bytes / bytes_per_cycle);
    
    // Convert to core frequency cycles
    double freq_ratio = (double)_config.core_freq / (double)icnt_freq_mhz;
    _load_cycles = (uint64_t)(icnt_cycles * freq_ratio);
    
    // Add base latency
    _load_cycles += _config.expert_load_latency;
    
    spdlog::info("Expert {} param load: {} bytes, {} cycles at core freq", 
                 _expert_id, param_bytes, _load_cycles);
}

std::vector<Ptr<BTensor>> ExpertParamLoad::get_outputs(std::vector<Ptr<BTensor>> inputs) {
    set_as_parent_tensor(inputs);
    
    assert(inputs.size() == 1);
    _inputs[0] = inputs[0];
    
    // Output is same as input (just adds latency for parameter loading)
    _outputs.resize(1);
    _outputs[0] = std::make_shared<NPUTensor>(
        _name + "_output",
        inputs[0]->get_dims(),
        NPUTensorBufType::ACT, false);
    
    initialize_tiles();
    
    return _outputs;
}

void ExpertParamLoad::initialize_tiles() {
    auto tile = Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = get_name(),
        .operation_id = _id,
        .batch = 0,
        .K = 0,
        .accum = false,
    };
    
    // Create DUMMY instruction that takes _load_cycles to complete
    // This models the parameter transfer time
    tile.instructions.push_back(Instruction{
        .opcode = Opcode::DUMMY,
        .start_cycle = 0,
        .finish_cycle = _load_cycles,  // Will be set by core
        .dest_addr = ACCUM_SPAD_BASE,
        .size = (uint32_t)_param_size_bytes,
        .src_addrs = std::vector<addr_type>{0},  // Dummy address
    });
    
    _tiles.push_back(tile);
    
    spdlog::info("ExpertParamLoad {}: {} bytes, {} cycles overhead", 
                 _expert_id, _param_size_bytes, _load_cycles);
}

Tile ExpertParamLoad::initialize_instructions() {
    return Tile{};  // Created in initialize_tiles
}

