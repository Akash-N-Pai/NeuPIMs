#include "MoEExecution.h"

MoEExecution::MoEExecution(uint32_t num_experts, uint32_t expert_cache_size)
    : _num_experts(num_experts), _expert_cache(expert_cache_size),
      _total_param_loads(0), _total_compute_cycles(0), _cache_saved_cycles(0) {
}

std::vector<MoEExecution::ExpertTask> MoEExecution::plan_execution(
    const std::vector<uint32_t>& expert_token_counts,
    uint64_t param_load_cycles_per_expert,
    uint64_t compute_cycles_per_token) {
    
    std::vector<ExpertTask> tasks;
    tasks.reserve(_num_experts);
    
    for (uint32_t expert_id = 0; expert_id < _num_experts; ++expert_id) {
        uint32_t num_tokens = expert_token_counts[expert_id];
        
        // OPTIMIZATION 1: Skip inactive experts (0 tokens)
        if (num_tokens == 0) {
            spdlog::debug("Expert {} inactive (0 tokens), skipping", expert_id);
            continue;
        }
        
        ExpertTask task;
        task.expert_id = expert_id;
        task.num_tokens = num_tokens;
        
        // OPTIMIZATION 3: Check cache - avoid param load if cached
        bool is_cached = _expert_cache.is_cached(expert_id);
        task.needs_param_load = !is_cached;
        
        if (is_cached) {
            _expert_cache.access_expert(expert_id);  // Update LRU
            task.param_load_cycles = 0;
            _cache_saved_cycles += param_load_cycles_per_expert;
            spdlog::debug("Expert {} cache HIT, saved {} cycles", 
                         expert_id, param_load_cycles_per_expert);
        } else {
            _expert_cache.access_expert(expert_id);  // Cache it
            task.param_load_cycles = param_load_cycles_per_expert;
            _total_param_loads++;
            spdlog::debug("Expert {} cache MISS, loading params ({} cycles)", 
                         expert_id, param_load_cycles_per_expert);
        }
        
        // OPTIMIZATION 4: Per-expert batched GEMM
        // Compute cycles proportional to actual tokens assigned
        task.compute_cycles = num_tokens * compute_cycles_per_token;
        
        task.total_cycles = task.param_load_cycles + task.compute_cycles;
        
        tasks.push_back(task);
        
        spdlog::debug("Expert {}: {} tokens, {} total cycles (load={}, compute={})",
                     expert_id, num_tokens, task.total_cycles, 
                     task.param_load_cycles, task.compute_cycles);
    }
    
    _total_compute_cycles = 0;
    for (const auto& task : tasks) {
        _total_compute_cycles += task.compute_cycles;
    }
    
    return tasks;
}

uint64_t MoEExecution::calculate_parallel_latency(const std::vector<ExpertTask>& tasks) {
    if (tasks.empty()) return 0;
    
    // OPTIMIZATION 2: Parallel expert execution
    // Wall-time = max(expert latencies), not sum
    // Assumes enough compute resources to run experts in parallel
    
    uint64_t max_latency = 0;
    for (const auto& task : tasks) {
        max_latency = std::max(max_latency, task.total_cycles);
    }
    
    spdlog::info("Parallel execution: {} active experts, max latency = {} cycles",
                 tasks.size(), max_latency);
    
    return max_latency;
}

uint64_t MoEExecution::calculate_double_buffered_latency(const std::vector<ExpertTask>& tasks) {
    if (tasks.empty()) return 0;
    
    // OPTIMIZATION 5: Double buffering - overlap param load and compute
    // While Expert i computes, load Expert i+1 parameters
    
    uint64_t total_latency = 0;
    
    for (size_t i = 0; i < tasks.size(); ++i) {
        const auto& task = tasks[i];
        
        if (i == 0) {
            // First expert: must load params, then compute (no overlap)
            total_latency += task.param_load_cycles + task.compute_cycles;
        } else {
            // Subsequent experts: overlap param load with previous compute
            const auto& prev_task = tasks[i - 1];
            
            if (task.needs_param_load) {
                // Can we hide param load in previous compute?
                if (task.param_load_cycles <= prev_task.compute_cycles) {
                    // Fully hidden! Only add compute time
                    total_latency += task.compute_cycles;
                    spdlog::debug("Expert {} param load fully hidden in Expert {} compute",
                                 task.expert_id, prev_task.expert_id);
                } else {
                    // Partially hidden
                    uint64_t exposed_load = task.param_load_cycles - prev_task.compute_cycles;
                    total_latency += exposed_load + task.compute_cycles;
                    spdlog::debug("Expert {} param load partially hidden ({} cycles exposed)",
                                 task.expert_id, exposed_load);
                }
            } else {
                // No param load needed (cached), just add compute
                total_latency += task.compute_cycles;
            }
        }
    }
    
    spdlog::info("Double-buffered execution: {} cycles (vs {} serial)",
                 total_latency, 
                 std::accumulate(tasks.begin(), tasks.end(), 0ULL,
                                [](uint64_t sum, const ExpertTask& t) { 
                                    return sum + t.total_cycles; 
                                }));
    
    return total_latency;
}

void MoEExecution::print_execution_plan(const std::vector<ExpertTask>& tasks) {
    spdlog::info("========== MoE Execution Plan ==========");
    spdlog::info("Total experts: {}", _num_experts);
    spdlog::info("Active experts: {}", tasks.size());
    spdlog::info("Inactive experts (skipped): {}", _num_experts - tasks.size());
    
    uint64_t total_serial = 0;
    uint64_t total_param_loads_cycles = 0;
    uint64_t total_compute = 0;
    
    spdlog::info("\nActive Expert Details:");
    spdlog::info("  ID | Tokens | Cached? | ParamLoad | Compute  | Total");
    spdlog::info("-----|--------|---------|-----------|----------|----------");
    
    for (const auto& task : tasks) {
        spdlog::info("  {:2d} | {:6d} |   {}   | {:8d} | {:8d} | {:8d}",
                     task.expert_id, 
                     task.num_tokens,
                     task.needs_param_load ? "NO " : "YES",
                     task.param_load_cycles,
                     task.compute_cycles,
                     task.total_cycles);
        
        total_serial += task.total_cycles;
        total_param_loads_cycles += task.param_load_cycles;
        total_compute += task.compute_cycles;
    }
    
    spdlog::info("\nExecution Summary:");
    spdlog::info("  Serial execution:         {} cycles", total_serial);
    spdlog::info("  Parallel execution:       {} cycles", calculate_parallel_latency(tasks));
    spdlog::info("  Double-buffered:          {} cycles", calculate_double_buffered_latency(tasks));
    spdlog::info("  Parameter load overhead:  {} cycles ({:.1f}%)", 
                 total_param_loads_cycles, 
                 100.0 * total_param_loads_cycles / total_serial);
    spdlog::info("  Compute cycles:           {} cycles ({:.1f}%)", 
                 total_compute,
                 100.0 * total_compute / total_serial);
    spdlog::info("  Cache saved:              {} cycles", _cache_saved_cycles);
    
    _expert_cache.print_stats();
    
    spdlog::info("========================================");
}

