#include "Model.h"

namespace BlockType {
std::string Attention = "attn";
std::string FeedForward = "ffn";
}  // namespace BlockType

namespace OperationType {
std::string LayerNorm = "ln";
std::string QKVGen = "QKVgen";
std::string Projection = "proj";
std::string FullyConnected1 = "fc1";
std::string FullyConnected2 = "fc2";
std::string LmHead = "lmhead";

std::string QKVSplit = "QKVsplit";
std::string QKMatMul = "QKmm";
std::string SoftMax = "softmax";
std::string LsVMatMul = "LsVmm";
std::string AReshape = "Areshape";
std::string Residual = "res";
std::string Gelu = "gelu";
std::string BatchSplit = "BSplit";
std::string BatchConcat = "BConcat";
std::string KCacheConcat = "Kccat";
std::string VCacheConcat = "Vccat";
std::string VConcat = "Vcat";
std::string PIMGEMVSoftmax = "PIMGEMVSoftmax";
std::string PIMGEMVAdd = "PIMGEMVAdd";
std::string NeuPIMSLogitSoftmax = "NeuPIMSLogitSoftmax";
std::string Attention = "Attention";
std::string Microbench = "Microbench";
std::string NeuPIMSAttend = "NeuPIMSAttend";
std::string FusedMHA = "FusedMHA";
std::string PIMGEMV = "PIMGEMV";

// MoE operations
std::string MoERouter = "moe_router";
std::string MoEExpert = "moe_expert";
std::string MoECombine = "moe_combine";
}  // namespace OperationType

namespace ParameterType {
std::string Weight = "weight";
std::string Bias = "bias";
}  // namespace ParameterType

Model::Model(SimulationConfig config, std::string name) {
    _name = name;
    _root_node_id = generate_id();
    _config = config;

    _num_batch = 3;
    _num_token = 13;
    _target_token = 14;

    init_params();

    // initialize_gpt2_params();

    // _is_decode = true;
    // if (_is_decode) {
    //     gpt2_decode();
    // } else {
    //     gpt2_encode();
    // }
}

/*
apply parallelism
pipeline parallelism in attention is applied to column(QKVgen, fc1) and row(proj, fc2)
need for layernorm variable to be at all chip
*/
void Model::init_params() {
    for (int i = 0; i < _config.model_n_layer; ++i) {
        auto attn = name_gen(LAYER(i), BlockType::Attention);
        create_weight(name_gen(attn, OperationType::LayerNorm, ParameterType::Weight),
                      {_config.model_n_embd});
        create_weight(name_gen(attn, OperationType::LayerNorm, ParameterType::Bias),
                      {_config.model_n_embd});
        create_weight(name_gen(attn, OperationType::QKVGen, ParameterType::Weight),
                      {_config.model_n_embd, 3 * _config.model_n_embd / _config.n_tp});
        create_weight(name_gen(attn, OperationType::QKVGen, ParameterType::Bias),
                      {3 * _config.model_n_embd / _config.n_tp});
        create_weight(name_gen(attn, OperationType::Projection, ParameterType::Weight),
                      {_config.model_n_embd / _config.n_tp, _config.model_n_embd});
        create_weight(name_gen(attn, OperationType::Projection, ParameterType::Bias),
                      {_config.model_n_embd});

        auto ffn = name_gen(LAYER(i), BlockType::FeedForward);
        create_weight(name_gen(ffn, OperationType::LayerNorm, ParameterType::Weight),
                      {_config.model_n_embd});
        create_weight(name_gen(ffn, OperationType::LayerNorm, ParameterType::Bias),
                      {_config.model_n_embd});
        
        if (_config.moe_enabled) {
            // MoE: Create router weights (no bias - routers typically don't use bias)
            create_weight(name_gen(ffn, OperationType::MoERouter, ParameterType::Weight),
                          {_config.model_n_embd, _config.num_experts});
            
            // Calculate scaled expert FFN dimension based on scaling mode
            uint32_t d_ff_expert = _config.get_expert_ffn_dim();
            
            spdlog::info("MoE FFN Scaling: mode='{}', d_ff_expert={} (dense d_ff={})",
                         _config.moe_ffn_scaling, d_ff_expert, 
                         4 * _config.model_n_embd / _config.n_tp);
            
            // MoE: Create per-expert weights with scaled FFN dimension
            for (uint32_t expert_id = 0; expert_id < _config.num_experts; ++expert_id) {
                auto expert = name_gen(ffn, OperationType::MoEExpert, std::to_string(expert_id));
                // FC1: [d_model, d_ff_expert]
                create_weight(name_gen(expert, OperationType::FullyConnected1, ParameterType::Weight),
                              {_config.model_n_embd, d_ff_expert});
                create_weight(name_gen(expert, OperationType::FullyConnected1, ParameterType::Bias),
                              {d_ff_expert});
                // FC2: [d_ff_expert, d_model]
                create_weight(name_gen(expert, OperationType::FullyConnected2, ParameterType::Weight),
                              {d_ff_expert, _config.model_n_embd});
                create_weight(name_gen(expert, OperationType::FullyConnected2, ParameterType::Bias),
                              {_config.model_n_embd});
            }
        } else {
            // Dense FFN: Create standard fc1/fc2 weights
            create_weight(name_gen(ffn, OperationType::FullyConnected1, ParameterType::Weight),
                          {_config.model_n_embd, 4 * _config.model_n_embd / _config.n_tp});
            create_weight(name_gen(ffn, OperationType::FullyConnected1, ParameterType::Bias),
                          {4 * _config.model_n_embd / _config.n_tp});
            create_weight(name_gen(ffn, OperationType::FullyConnected2, ParameterType::Weight),
                          {4 * _config.model_n_embd / _config.n_tp, _config.model_n_embd});
            create_weight(name_gen(ffn, OperationType::FullyConnected2, ParameterType::Bias),
                          {_config.model_n_embd});
        }
    }
    // LM head: both encoder, decoder are GEMV
    create_weight(name_gen(OperationType::LmHead, ParameterType::Weight),
                  {_config.model_n_embd, _config.model_vocab_size});

    // in advance, caculate weight size to decide base addr of buffer
    _wgt_size = 0;
    for (auto const &[tensor_name, tensor] : _wgt_map) {
        _wgt_size += tensor->_inners[0]->_size;
        // spdlog::info("{}: {}", tensor->get_name(), tensor->get_size());
    }
    spdlog::info("Total weight size: {} bytes ({:.2f} MB)", _wgt_size, _wgt_size / (1024.0 * 1024.0));
    
    if (_config.moe_enabled) {
        spdlog::info("========== MoE Parameter Count ==========");
        spdlog::info("Num experts: {}, top-{} routing", _config.num_experts, _config.experts_per_token);
        spdlog::info("FFN scaling mode: '{}'", _config.moe_ffn_scaling);
        
        uint32_t d_model = _config.model_n_embd;
        uint32_t d_ff_expert = _config.get_expert_ffn_dim();
        uint32_t d_ff_dense = 4 * d_model / _config.n_tp;
        
        // Per-layer parameter counts
        uint64_t attn_params = (d_model * 3 * d_model / _config.n_tp) + (d_model / _config.n_tp * d_model);
        uint64_t router_params = d_model * _config.num_experts;
        uint64_t expert_params = 2 * d_model * d_ff_expert;  // FC1 + FC2
        uint64_t all_experts_params = expert_params * _config.num_experts;
        uint64_t moe_ffn_params = router_params + all_experts_params;
        uint64_t layer_params = attn_params + moe_ffn_params;
        
        // For comparison: dense FFN
        uint64_t dense_ffn_params = 2 * d_model * d_ff_dense;
        
        spdlog::info("Per-layer breakdown:");
        spdlog::info("  Attention: {:.2f}M params", attn_params / 1e6);
        spdlog::info("  MoE Router: {:.2f}M params", router_params / 1e6);
        spdlog::info("  Per expert: {:.2f}M params (d_ff_expert={})", expert_params / 1e6, d_ff_expert);
        spdlog::info("  All {} experts: {:.2f}M params", _config.num_experts, all_experts_params / 1e6);
        spdlog::info("  Total MoE FFN: {:.2f}M params", moe_ffn_params / 1e6);
        spdlog::info("  Total layer: {:.2f}M params", layer_params / 1e6);
        spdlog::info("");
        spdlog::info("Comparison with dense FFN:");
        spdlog::info("  Dense FFN: {:.2f}M params (d_ff={})", dense_ffn_params / 1e6, d_ff_dense);
        spdlog::info("  MoE FFN ratio: {:.2f}×", (double)moe_ffn_params / dense_ffn_params);
        spdlog::info("");
        spdlog::info("Active parameters per token (top-{}):", _config.experts_per_token);
        uint64_t active_expert_params = expert_params * _config.experts_per_token;
        uint64_t active_params = attn_params + router_params + active_expert_params;
        spdlog::info("  {} experts: {:.2f}M params", _config.experts_per_token, active_expert_params / 1e6);
        spdlog::info("  Total active: {:.2f}M params", active_params / 1e6);
        spdlog::info("  Sparse activation: {:.1f}%", 100.0 * active_params / layer_params);
        spdlog::info("=========================================");
    }
}

Ptr<NPUTensor> Model::find_tensor(std::string name) { return _wgt_map[name]; }

std::vector<Ptr<NPUTensor>> Model::get_params(int layer_idx, std::string block_type,
                                              std::string operation_type) {
    std::string prefix = name_gen(LAYER(layer_idx), block_type, operation_type);
    Ptr<NPUTensor> wgt = find_tensor(name_gen(prefix, ParameterType::Weight));
    Ptr<NPUTensor> bias = find_tensor(name_gen(prefix, ParameterType::Bias));
    return {wgt, bias};
}

// todo: load from real address (requests)
std::shared_ptr<Tensor> Model::load_cache(uint32_t layer, std::string type) {
    std::vector<uint32_t> shape;
    if (type == "key") {
        shape.assign(
            {_config.model_n_head, _config.model_n_embd / _config.model_n_head, _num_token});
    } else if (type == "value") {
        shape.assign(
            {_config.model_n_head, _num_token, _config.model_n_embd / _config.model_n_head});
    }

    return create_tensor("layer" + std::to_string(layer) + "." + type, shape);
}

std::shared_ptr<Operation> Model::register_operation(std::shared_ptr<Operation> op) {
    // auto id = op->get_id();
    // _operation_map[id] = op;
    return _operation_map[op->get_id()] = op;
}

void Model::find_executable_node(std::shared_ptr<Tensor> tensor) {
    // spdlog::info("intializing from tensor {}", tensor_id);
    for (auto op : tensor->get_child_nodes()) {
        // spdlog::info("initializing operation {} ...", op->get_name());
        if (op->check_executable()) {
            // spdlog::info("success initializing operation {}!",
            // op->get_name());
            _executable_operations.push_back(op);
        }
    }
}

Model::Model(const Model &model) {
    _name = model._name;
    _root_node_id = _root_node_id;
    _input_tensor = _input_tensor;

    _input_name = model._input_name;
    _input_dim = model._input_dim;
    _wgt_size = model._wgt_size;

    for (auto const &[key, val] : model._tensor_map) {
        _tensor_map[key] = val;
    }
    for (auto const &[key, val] : model._operation_map) {
        _operation_map[key] = val;
    }
    for (auto operation : model._executable_operations) {
        _executable_operations.push_back(_operation_map[operation->get_id()]);
        spdlog::trace("add op {0:x}", fmt::ptr(_executable_operations.front()));
    }
}

// std::shared_ptr<Tensor> Model::get_tensor(uint32_t id) { return _tensor_map[id]; }

// std::shared_ptr<Tensor> Model::find_tensor(std::string name) {
//     for (auto const &[key, val] : _tensor_map) {
//         if (val->_name == name) {
//             return val;
//         }
//     }
//     assert(0);
//     return nullptr;
// }

// tensor initialization allocating to memory.
// for weight tensors and input tensors
std::shared_ptr<Tensor> Model::create_tensor(std::string name, std::vector<uint32_t> dims) {
    // auto tensor = std::make_shared<Tensor>(name, dims, true);
    // _tensor_map[tensor->get_id()] = tensor;

    // // spdlog::info("create tensor name {}", name);

    // return tensor;
}

// xxx is it necessary to return value?
std::shared_ptr<NPUTensor> Model::create_weight(std::string name, std::vector<uint32_t> dims) {
    // create_tensor(name, dims, TensorBufType::WGT);
    // auto tensor = std::make_shared<BatchedTensor>(name, dims, TensorBufType::WGT, true);
    auto tensor = std::make_shared<NPUTensor>(name, dims, NPUTensorBufType::WGT, true);
    _wgt_map[name] = tensor;
    return tensor;
}

// input: operation id
// erase target operation and insert readied operation.
void Model::finish_operation(uint32_t id) {
    _operation_map[id]->set_finish();
    for (auto iter = _executable_operations.begin(); iter != _executable_operations.end(); iter++) {
        if (id == (*iter)->get_id()) {
            _executable_operations.erase(iter);
            break;
        }
    }
    // log_operations();
    for (auto op : _operation_map[id]->get_child_nodes()) {
        if (op->check_executable() && !check_exist_in_executable(op->get_id())) {
            _executable_operations.push_back(op);
        }
    }
}

void Model::finish_operation_tile(uint32_t id, Tile &tile) {
    _operation_map[id]->reduce_tile(tile);
}

std::vector<std::shared_ptr<Operation>> Model::get_executable_operations() {
    return _executable_operations;
}

bool Model::check_finish() {
    bool finish = true;
    for (auto const &[key, val] : _operation_map) {
        finish = finish && val->check_finish();
    }
    if (_is_decode && finish && _num_token != _target_token) {
        finish = false;
        // gpt2_decode();
    }

    return finish;
}

bool Model::check_exist_in_executable(uint32_t op_id) {
    for (auto iter = _executable_operations.begin(); iter != _executable_operations.end(); iter++) {
        if (op_id == (*iter)->get_id()) {
            return true;
        }
    }
    return false;
}

// void Model::log_model() {
//     std::ofstream log_file(Config::global_config.operation_log_output_path);

//     if (!log_file.is_open()) {
//         assert(0);
//     }
//     log_file << "op_name\tstart_cycle\tend_cycle\tcompute_cycles\tnum_calculation\tmemory_"
//                 "stalls\tmemory_reads\tmemory_writes\tnpu_utlization"
//              << std::endl;
//     for (auto &[key, val] : _operation_map) {
//         log_file << val->repr();
//     }
//     log_file.close();
// }

uint64_t Model::get_weight_size() { return _wgt_size; }

addr_type Model::get_weight_top_addr() {
    // after aligning wgt_size, add alignment
    return AddressConfig::align(_wgt_size) + AddressConfig::alignment;
}