#include "MoEStats.h"

namespace MoEStats {

std::vector<MoELayerStat> layer_stats;

void init() {
    layer_stats.clear();
}

void record_router_completion(std::string layer_name, uint64_t cycles) {
    // Find or create layer stat
    for (auto &stat : layer_stats) {
        if (stat.layer_name == layer_name) {
            stat.router_cycles = cycles;
            return;
        }
    }
    
    // Create new layer stat
    MoELayerStat new_stat;
    new_stat.layer_name = layer_name;
    new_stat.num_experts = Config::global_config.num_experts;
    new_stat.experts_per_token = Config::global_config.experts_per_token;
    new_stat.router_cycles = cycles;
    new_stat.expert_stats.resize(Config::global_config.num_experts);
    for (uint32_t i = 0; i < Config::global_config.num_experts; ++i) {
        new_stat.expert_stats[i].expert_id = i;
    }
    layer_stats.push_back(new_stat);
}

void record_expert_completion(std::string layer_name, uint32_t expert_id, uint64_t cycles) {
    for (auto &stat : layer_stats) {
        if (stat.layer_name == layer_name) {
            if (expert_id < stat.expert_stats.size()) {
                stat.expert_stats[expert_id].compute_cycles = cycles;
            }
            return;
        }
    }
}

void record_combine_completion(std::string layer_name, uint64_t cycles) {
    for (auto &stat : layer_stats) {
        if (stat.layer_name == layer_name) {
            stat.combine_cycles = cycles;
            return;
        }
    }
}

void print_stats() {
    if (!Config::global_config.moe_enabled) return;
    
    spdlog::info("========== MoE Statistics ==========");
    for (auto &stat : layer_stats) {
        spdlog::info("Layer: {}", stat.layer_name);
        spdlog::info("  Router cycles: {}", stat.router_cycles);
        spdlog::info("  Experts ({} of {} active):", stat.experts_per_token, stat.num_experts);
        for (uint32_t i = 0; i < stat.experts_per_token; ++i) {
            spdlog::info("    Expert {}: {} cycles", i, stat.expert_stats[i].compute_cycles);
        }
        spdlog::info("  Combine cycles: {}", stat.combine_cycles);
        
        uint64_t total_expert_cycles = 0;
        for (uint32_t i = 0; i < stat.experts_per_token; ++i) {
            total_expert_cycles += stat.expert_stats[i].compute_cycles;
        }
        stat.total_moe_cycles = stat.router_cycles + total_expert_cycles + stat.combine_cycles;
        spdlog::info("  Total MoE cycles: {}", stat.total_moe_cycles);
    }
    spdlog::info("====================================");
}

void log_stats(std::string log_dir) {
    if (!Config::global_config.moe_enabled) return;
    
    std::string fname = log_dir + "/moe_stats.tsv";
    std::ofstream ofile(fname);
    if (!ofile.is_open()) {
        spdlog::warn("Could not open MoE stats file: {}", fname);
        return;
    }
    
    // Header
    ofile << "Layer\tRouterCycles\tExpert0Cycles\tExpert1Cycles\tCombineCycles\tTotalMoECycles\n";
    
    // Data
    for (auto &stat : layer_stats) {
        ofile << stat.layer_name << "\t";
        ofile << stat.router_cycles << "\t";
        for (uint32_t i = 0; i < stat.experts_per_token && i < 2; ++i) {
            ofile << stat.expert_stats[i].compute_cycles << "\t";
        }
        ofile << stat.combine_cycles << "\t";
        ofile << stat.total_moe_cycles << "\n";
    }
    
    ofile.close();
    spdlog::info("MoE stats logged to: {}", fname);
}

}  // namespace MoEStats

