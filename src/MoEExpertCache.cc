#include "MoEExpertCache.h"

MoEExpertCache::MoEExpertCache(uint32_t cache_capacity)
    : _cache_capacity(cache_capacity), _cache_hits(0), _cache_misses(0), _evictions(0) {
    spdlog::info("MoE Expert Cache initialized: capacity = {} experts", cache_capacity);
}

bool MoEExpertCache::is_cached(uint32_t expert_id) {
    return _cached_experts.find(expert_id) != _cached_experts.end();
}

void MoEExpertCache::cache_expert(uint32_t expert_id) {
    if (is_cached(expert_id)) {
        // Already cached, just update LRU
        access_expert(expert_id);
        return;
    }
    
    // Check if cache is full
    if (_cached_experts.size() >= _cache_capacity) {
        evict_lru();
    }
    
    // Add to cache
    _cached_experts.insert(expert_id);
    _lru_queue.push_back(expert_id);
    
    spdlog::debug("Cached expert {}, cache size: {}/{}", 
                  expert_id, _cached_experts.size(), _cache_capacity);
}

void MoEExpertCache::access_expert(uint32_t expert_id) {
    if (is_cached(expert_id)) {
        _cache_hits++;
        
        // Update LRU: move to back (most recently used)
        auto it = std::find(_lru_queue.begin(), _lru_queue.end(), expert_id);
        if (it != _lru_queue.end()) {
            _lru_queue.erase(it);
            _lru_queue.push_back(expert_id);
        }
    } else {
        _cache_misses++;
        cache_expert(expert_id);
    }
}

void MoEExpertCache::evict_lru() {
    if (_lru_queue.empty()) return;
    
    uint32_t lru_expert = _lru_queue.front();
    _lru_queue.pop_front();
    _cached_experts.erase(lru_expert);
    _evictions++;
    
    spdlog::debug("Evicted expert {} from cache", lru_expert);
}

double MoEExpertCache::get_hit_rate() const {
    uint64_t total = _cache_hits + _cache_misses;
    if (total == 0) return 0.0;
    return (double)_cache_hits / total;
}

void MoEExpertCache::print_stats() {
    spdlog::info("========== Expert Cache Statistics ==========");
    spdlog::info("Cache capacity: {} experts", _cache_capacity);
    spdlog::info("Current cached: {} experts", _cached_experts.size());
    spdlog::info("Cache hits: {}", _cache_hits);
    spdlog::info("Cache misses: {}", _cache_misses);
    spdlog::info("Hit rate: {:.2f}%", get_hit_rate() * 100);
    spdlog::info("Evictions: {}", _evictions);
    spdlog::info("=============================================");
}

void MoEExpertCache::reset_stats() {
    _cache_hits = 0;
    _cache_misses = 0;
    _evictions = 0;
}

