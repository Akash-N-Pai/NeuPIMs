#pragma once

#include "Common.h"
#include <deque>

/**
 * MoEExpertCache: Manages on-chip caching of frequently used experts
 * 
 * Key features:
 * - LRU (Least Recently Used) eviction policy
 * - Configurable cache size (number of experts that fit on-chip)
 * - Tracks cache hits/misses for statistics
 * - Models realistic expert reuse across batches
 */
class MoEExpertCache {
   public:
    MoEExpertCache(uint32_t cache_capacity);
    
    // Check if expert is in cache
    bool is_cached(uint32_t expert_id);
    
    // Add expert to cache (may evict LRU expert)
    void cache_expert(uint32_t expert_id);
    
    // Access expert (updates LRU)
    void access_expert(uint32_t expert_id);
    
    // Statistics
    uint64_t get_hits() const { return _cache_hits; }
    uint64_t get_misses() const { return _cache_misses; }
    double get_hit_rate() const;
    
    void print_stats();
    void reset_stats();
    
   private:
    uint32_t _cache_capacity;  // Number of experts that fit on-chip
    std::deque<uint32_t> _lru_queue;  // LRU order (front = oldest, back = newest)
    robin_hood::unordered_set<uint32_t> _cached_experts;
    
    uint64_t _cache_hits;
    uint64_t _cache_misses;
    uint64_t _evictions;
    
    void evict_lru();
};

