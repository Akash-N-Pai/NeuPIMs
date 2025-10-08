#include "ExpertParamLoad.h"

ExpertParamLoad::ExpertParamLoad(std::string name, uint32_t expert_id,
                                 std::vector<Ptr<NPUTensor>> expert_weights,
                                 Ptr<BTensor> data_tensor)
    : Operation(name), _expert_id(expert_id), _data_tensor(data_tensor) {
    
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
    // Based on actual weight size passed to this operation
    // The weights are already provided in the constructor, so we use _param_size_bytes
    // which was calculated from the actual expert weight tensors
    
    uint64_t param_bytes = _param_size_bytes;
    
    // Transfer latency calculation using interconnect bandwidth
    // Interconnect connects NPU cores to DRAM (NOT HBM2 bandwidth - that's for PIM)
    uint32_t icnt_bandwidth_gbps = 64;  // 256-bit wide @ 2000 MHz
    uint32_t core_freq_mhz = _config.core_freq;  // 1000 MHz
    
    // Bytes per cycle at core frequency
    double bytes_per_cycle = (double)icnt_bandwidth_gbps * 1e9 / (core_freq_mhz * 1e6);
    
    // Cycles for transfer at core frequency
    uint64_t transfer_cycles = (uint64_t)(param_bytes / bytes_per_cycle);
    
    // Add base latency (interconnect protocol overhead)
    _load_cycles = transfer_cycles + _config.expert_load_latency;
    
    spdlog::info("Expert {} param load: {} bytes, {} cycles at core freq", 
                 _expert_id, param_bytes, _load_cycles);
}

std::vector<Ptr<BTensor>> ExpertParamLoad::get_outputs(std::vector<Ptr<BTensor>> inputs) {
    set_as_parent_tensor(inputs);
    
    assert(inputs.size() == 1);
    _inputs[0] = inputs[0];  // This is the dependency trigger (normalized_input or prev signal)
    
    // TWO OUTPUTS for true double buffering:
    // Output 0: Data passthrough for FC1 (ALWAYS the data_tensor, not the input)
    // Output 1: Completion signal for chaining to next expert's param_load
    _outputs.resize(2);
    _outputs[0] = std::make_shared<NPUTensor>(
        _name + "_data_output",
        _data_tensor->get_dims(),  // Use stored data_tensor dimensions
        NPUTensorBufType::ACT, false);
    _outputs[1] = std::make_shared<NPUTensor>(
        _name + "_completion_signal",
        std::vector<uint32_t>{1},  // Tiny tensor - just a signal
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
    
    // Use DUMMY instruction with size = _load_cycles
    // The modified DUMMY opcode will return this as the cycle count
    // This properly injects the parameter load latency into the timeline
    tile.instructions.push_back(Instruction{
        .opcode = Opcode::DUMMY,
        .dest_addr = ACCUM_SPAD_BASE,
        .size = _load_cycles,  // DUMMY will use this as cycle count
        .src_addrs = std::vector<addr_type>{},  // No memory access needed
    });
    
    _tiles.push_back(tile);
    
    spdlog::info("ExpertParamLoad {}: {} bytes, {} cycles overhead", 
                 _expert_id, _param_size_bytes, _load_cycles);
}

Tile ExpertParamLoad::initialize_instructions() {
    return Tile{};  // Created in initialize_tiles
}

