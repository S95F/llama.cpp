#include "llama-moe-cache.h"

#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-memory.h"
#include "llama-mmap.h"
#include "llama-model.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <unistd.h>
#endif

namespace {

// SLRU of the experts of a group of layers, the slot of each expert is kept in the slot map of its layer
struct moe_cache_slru {
    int32_t n_expert = 0;
    int32_t n_slots  = 0;
    int32_t empty    = 0; // slot of the experts that are not cached

    // [n_layer] data of the slot maps, the experts at slot n_slots or later are not cached
    // the experts computed on the CPU read zero slots after the cached ones on the device
    std::vector<int32_t *> slot_map;
    std::vector<int32_t>   key_of;   // [n_slots] il*n_expert + expert, -1 if empty
    int32_t n_filled = 0;            // slots that hold an expert

    // probation and protected LRU lists; hits promote slots to the protected half
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    int32_t head[2] = { -1, -1 };
    int32_t tail[2] = { -1, -1 };
    std::vector<bool> is_protected;
    int32_t n_protected = 0;
    int32_t max_protected = 0; // zero for plain LRU

    std::vector<uint32_t> seen; // [n_expert]
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;
    std::vector<int32_t> misses;
    std::vector<uint64_t> token_zero; // [n_tokens] zero slots used by each token

    // CRF (combined recency and frequency) = uses of an expert, decayed with a half-life of N tokens
    // only picks the misses left to the CPU, eviction does not read it
    static constexpr float crf_half_life = 128.0f; // tokens
    std::vector<float>   crf;      // [n_layer*n_expert]
    std::vector<int64_t> last_use; // [n_layer*n_expert] layer clock at the last use, -1 if not used
    std::vector<int64_t> clock;    // [n_layer] tokens seen by the layer (as added by plan() for each ubatch)

    int32_t n_sample = 0;     // slots to sample for the least used victim, zero to evict the least recently used
    std::vector<uint32_t> use; // [n_layer*n_expert] times each expert was selected, empty if n_sample is zero
    uint32_t rng = 1;
    int64_t n_fills   = 0;
    int64_t age_every = 0;     // halve the use counters after this many fills, so old popularity is forgotten

    // generation sweeps the layers of a group in order, so the next use of a layer is known, not predicted
    int32_t n_group_layers = 0;
    int64_t n_calls = 0;
    std::vector<int64_t> last_call; // [n_layer] call this layer was last planned at, -1 if never

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots, int32_t max_protected, int32_t n_sample, int32_t n_group_layers) {
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        this->empty    = n_slots;
        this->max_protected = max_protected;
        this->n_sample = n_sample;
        this->n_group_layers = n_group_layers;
        if (n_sample > 0) {
            use.assign((size_t) n_layer*n_expert, 0);
            last_call.assign(n_layer, -1);
            age_every = 8*(int64_t) n_slots;
        }
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        head[0] = 0;
        tail[0] = n_slots - 1;
        is_protected.assign(n_slots, false);
        crf.assign((size_t) n_layer*n_expert, 0.0f);
        last_use.assign((size_t) n_layer*n_expert, -1);
        clock.assign(n_layer, 0);
        seen.assign(n_expert, 0);
    }

    // an unused key gets 1 because crf[key] is 0 prior to first use
    void use_crf(int32_t il, int32_t key) {
        crf[key]      = crf[key]*std::exp2(-(clock[il] - last_use[key])/crf_half_life) + 1.0f;
        last_use[key] = clock[il];
    }

    // move slot s to the most recently used end of a segment
    void move(int32_t s, bool protect) {
        const bool from = is_protected[s];
        if (from == protect && s == tail[from]) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head[from] = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail[from] = prev[s];
        }

        prev[s] = tail[protect];
        next[s] = -1;
        if (tail[protect] >= 0) {
            next[tail[protect]] = s;
        } else {
            head[protect] = s;
        }
        tail[protect] = s;
        is_protected[s] = protect;
        n_protected += int(protect) - int(from);
    }

    void touch(int32_t s) {
        move(s, max_protected > 0);
        if (n_protected > max_protected) {
            move(head[1], false);
        }
    }

    // calls until the sweep comes back to layer il
    // the layer the sweep just passed is the furthest from its next use, the one just ahead of it the nearest
    int64_t dist(int32_t il) const {
        if (n_group_layers <= 1) {
            return 1; // a single layer, or the term is off: the victim is scored by use count only
        }
        if (last_call[il] < 0) {
            return n_group_layers;
        }
        return std::max<int64_t>(1, n_group_layers - (n_calls - last_call[il]));
    }

    // of n_sample random slots, the one whose expert is furthest from its next use, -1 if none of them can be evicted
    // a cache below one token's experts thrashes under LRU: the sweep over the layers evicts every expert before the next token reads it again
    int32_t victim_sampled(int32_t il) {
        if (++n_fills % age_every == 0) {
            for (uint32_t & u : use) {
                u >>= 1;
            }
        }

        int32_t best   = -1;
        int64_t best_d = 0;
        int64_t best_u = 0;
        for (int32_t i = 0, n = 0; i < 4*n_sample && n < n_sample; ++i) {
            rng = rng*1664525u + 1013904223u;
            const int32_t s   = (int32_t) (rng % (uint32_t) n_slots);
            const int32_t key = key_of[s];
            if (key >= 0 && key / n_expert == il && seen[key % n_expert] == seen_gen) {
                continue;
            }
            if (key < 0) {
                return s;
            }
            // time to the next use of this expert is about dist(layer) / how often the layer picks it,
            // so take the largest dist/use, compared as a ratio to keep it in integers
            const int64_t d = dist(key / n_expert);
            const int64_t u = (int64_t) use[key] + 1;
            if (best < 0 || d*best_u > best_d*u) {
                best   = s;
                best_d = d;
                best_u = u;
            }
            n++;
        }
        return best;
    }

    int32_t victim(int32_t il) {
        if (n_sample > 0) {
            const int32_t s = victim_sampled(il);
            if (s >= 0) {
                return s;
            }
        }
        for (int segment = 0; segment < 2; ++segment) {
            for (int32_t s = head[segment]; s >= 0; s = next[s]) {
                const int32_t key = key_of[s];
                // keep every expert selected by this request, including earlier fills
                if (key < 0 || key / n_expert != il || seen[key % n_expert] != seen_gen) {
                    return s;
                }
            }
        }
        GGML_ABORT("no SLRU victim");
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // give a slot to each expert selected by ids (n_tokens tokens) in layer il, evicting from probation first
    // n_cpu(n) of the n misses are left to the CPU and read a zero slot on the device, they are not cached
    // a zero slot is used at most once per token, a miss that does not get one is cached instead
    // returns false if the ids select more distinct experts than there are slots
    template <typename F>
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, int64_t n_tokens, int32_t n_used, int32_t n_zero, F && n_cpu,
            std::vector<fill> & fills, std::vector<fill> & cpu, size_t & n_hit) {
        fills.clear();
        cpu.clear();
        n_hit = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
                if (!use.empty()) {
                    use[(size_t) il*n_expert + ids[i]]++;
                }
            }
        }
        if (!last_call.empty()) {
            last_call[il] = ++n_calls;
        }
        if (uniq.size() > (size_t) n_slots) {
            return false;
        }

        int32_t * slots = slot_map[il];

        // promote hits before choosing victims for misses
        misses.clear();
        for (int32_t e : uniq) {
            use_crf(il, il*n_expert + e);
            if (slots[e] < n_slots) {
                touch(slots[e]);
                n_hit++;
            } else {
                misses.push_back(e);
            }
        }

        // the misses with the lowest CRF are left to the CPU, a miss that keeps being used gains CRF and enters the cache
        // all misses are used at the same clock, so their CRF compare without decay
        const size_t n_c = std::min(n_cpu(misses.size()), misses.size());
        if (n_c > 0) {
            std::sort(misses.begin(), misses.end(), [&](int32_t a, int32_t b) {
                const float crf_a = crf[il*n_expert + a];
                const float crf_b = crf[il*n_expert + b];
                return crf_a != crf_b ? crf_a < crf_b : a < b;
            });
            GGML_ASSERT((size_t) n_tokens*n_used == n_ids);
            token_zero.assign(n_tokens, 0);
            size_t n_keep = n_c;
            for (size_t i = 0; i < n_keep; ) {
                const int32_t e = misses[i];
                uint64_t used = 0;
                for (int64_t it = 0; it < n_tokens; ++it) {
                    for (int32_t k = 0; k < n_used; ++k) {
                        if (ids[it*n_used + k] == e) {
                            used |= token_zero[it];
                        }
                    }
                }
                int32_t z = 0;
                while (z < 64 && (used >> z & 1)) {
                    z++;
                }
                if (z >= n_zero) {
                    std::swap(misses[i], misses[--n_keep]);
                    continue;
                }
                for (int64_t it = 0; it < n_tokens; ++it) {
                    for (int32_t k = 0; k < n_used; ++k) {
                        if (ids[it*n_used + k] == e) {
                            token_zero[it] |= 1ull << z;
                        }
                    }
                }
                slots[e] = n_slots + z;
                cpu.push_back({ e, n_slots + z });
                i++;
            }
            misses.erase(misses.begin(), misses.begin() + n_keep);
        }

        // sorted misses usually get consecutive slots, so the uploads can be merged
        std::sort(misses.begin(), misses.end());
        for (int32_t e : misses) {
            const int32_t s = victim(il);
            if (key_of[s] >= 0) {
                slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = empty;
            } else {
                n_filled++;
            }
            key_of[s] = il*n_expert + e;
            slots[e] = s;
            move(s, false);
            fills.push_back({ e, s });
        }
        clock[il] += n_tokens;
        return true;
    }
};

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

static size_t llama_moe_cache_page_size() {
#if defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (size_t) si.dwPageSize;
#else
    const long page = sysconf(_SC_PAGESIZE);
    return page > 0 ? (size_t) page : 4096;
#endif
}

// sort page-aligned ranges and merge the ones that overlap or touch
static std::vector<std::pair<uintptr_t, uintptr_t>> llama_moe_cache_merge_ranges(std::vector<std::pair<uintptr_t, uintptr_t>> ranges) {
    std::sort(ranges.begin(), ranges.end());
    std::vector<std::pair<uintptr_t, uintptr_t>> merged;
    for (const auto & r : ranges) {
        if (!merged.empty() && r.first <= merged.back().second) {
            merged.back().second = std::max(merged.back().second, r.second);
        } else {
            merged.push_back(r);
        }
    }
    return merged;
}

// the host expert ranges pinned by all the MoE caches of the process, sorted
static std::mutex llama_moe_cache_pinned_mutex;
static std::vector<std::pair<uintptr_t, uintptr_t>> llama_moe_cache_pinned;

// bytes of [data, data + size) in pinned ranges
static size_t llama_moe_cache_pinned_bytes(const void * data, size_t size) {
    std::lock_guard<std::mutex> lock(llama_moe_cache_pinned_mutex);
    const auto & pinned = llama_moe_cache_pinned;
    const uintptr_t lo = (uintptr_t) data;
    const uintptr_t hi = lo + size;
    size_t res = 0;
    auto it = std::upper_bound(pinned.begin(), pinned.end(), lo, [](uintptr_t v, const std::pair<uintptr_t, uintptr_t> & r) { return v < r.second; });
    for (; it != pinned.end() && it->first < hi; ++it) {
        res += std::min(hi, it->second) - std::max(lo, it->first);
    }
    return res;
}

// drop the ranges whose middle page is in the working set of the process, where a prefetch only costs time
// Windows only, elsewhere every range is kept
static void llama_moe_cache_drop_resident(llama_memory_ranges & mr) {
#if defined(_WIN32)
    const size_t page = llama_moe_cache_page_size();
    std::vector<PSAPI_WORKING_SET_EX_INFORMATION> info(mr.size());
    for (size_t i = 0; i < mr.size(); ++i) {
        info[i].VirtualAddress = (PVOID) (((uintptr_t) mr[i].addr + mr[i].size/2) & ~(page - 1));
    }
    if (info.empty() || !QueryWorkingSetEx(GetCurrentProcess(), info.data(), (DWORD) (info.size()*sizeof(info[0])))) {
        return;
    }
    size_t n = 0;
    for (size_t i = 0; i < mr.size(); ++i) {
        if (!info[i].VirtualAttributes.Valid) {
            mr[n++] = mr[i];
        }
    }
    mr.resize(n);
#else
    GGML_UNUSED(mr);
#endif
}

}

struct llama_moe_cache::impl {
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        moe_cache_slru slru;
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        ggml_tensor * slot_map_cpu = nullptr; // looked up in place of slot_map by the graphs with a CPU part, if cpu_share > 0, its data is not used
        ggml_tensor * cpu_map  = nullptr;   // I32 [1, n_expert] in host memory, if cpu_share > 0
        std::vector<int32_t> cpu_experts;   // experts set in cpu_map
        uint32_t cpu_acc = 0;               // fraction of a miss carried over to the next plan, in 1/65536
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    struct stats {
        size_t hits   = 0;
        size_t misses = 0;
        size_t cpu    = 0; // misses computed on the CPU
        size_t bytes  = 0;
    };

    static constexpr int64_t max_batch = 32;

    int32_t n_expert_used;

    stats stats_small;  // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;   // experts copied from the cache for large batches
    stats stats_upload; // uncached experts uploaded from host memory for large batches

    // telemetry: LLAMA_MOE_CACHE_LOG=N logs the stats every N slot-map copies (0 = teardown only),
    // LLAMA_MOE_CACHE_TRACE=path writes a CSV of the expert ids selected at each slot-map copy
    size_t n_copy_calls = 0;
    int    log_every    = 0;
    FILE * trace_file   = nullptr;

    // LLAMA_MOE_COPY_LOG=path writes, at teardown, a CSV of the host time of each upload and prefetch of the slot-map copies
    // for a prefetch: ip = -1, expert = candidate ranges, n = ranges prefetched, bytes = bytes prefetched, pinned = candidate bytes
    // uploads of large batches (upload()) have call 0, n is the number of whole experts (a padding-only upload has n = 0)
    struct copy_rec {
        uint32_t call;   // as in the trace
        int16_t  layer;
        int16_t  ip;     // index of the expert tensor, -1 for a prefetch
        int32_t  expert; // first expert uploaded
        int32_t  n;      // number of experts
        uint32_t bytes;
        uint32_t pinned; // bytes in pinned ranges
        uint32_t us;
    };
    std::string copy_log_path;
    std::vector<copy_rec> copy_log;

    // LLAMA_MOE_PREFETCH: prefetch the pageable pages of a layer's missed experts in one call before uploading them.
    // 1 = all of them; 2 = only those whose middle page is not in the working set (Windows; elsewhere the same as 1)
    int prefetch = 0;

    // host expert ranges this cache registered with the device driver (pinned), sorted, released in the dtor
    std::vector<std::pair<uintptr_t, uintptr_t>> pinned_ranges;
    bool (*reg_host)(void *, size_t) = nullptr;
    void (*unreg_host)(void *) = nullptr;
    std::vector<ggml_backend_buffer_type_t> pin_host_bufts; // experts in these buffers are pinned already
    size_t pin_cap   = 0;
    size_t pin_total = 0; // host expert bytes that could be pinned

    // LLAMA_MOE_PIN_ONLINE: selections per layer and expert in the first online_n small batches, pinned by count after them
    int64_t online_n      = 0;
    int64_t online_passes = 0;
    int32_t online_layer  = -1; // first cached layer, starts a pass
    std::vector<std::vector<int64_t>> online_use;

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, std::pair<int32_t, bool>> layer_of; // slot map -> layer, and if the graph that looks it up has a CPU part

    std::vector<int32_t> ids;
    std::vector<moe_cache_slru::fill> fills;
    std::vector<moe_cache_slru::fill> cpu;

    float    cpu_share = 0.0f;
    uint32_t cpu_share_q = 0; // cpu_share in 1/65536, integer so that a simulator can reproduce the rounding
    int32_t  n_zero    = 0; // zero slots per group

    // LLAMA_MOE_CACHE_TRACE_BIN: the selected experts of each slot lookup, in the binary format of scripts/moe-cache-tune.py
    // (LLAMA_MOE_CACHE_TRACE is the CSV trace above); the script simulates ClassLRU eviction, not the policy of this branch
    FILE * trace_bin = nullptr;

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    impl(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share) :
            n_expert_used(model.hparams.n_expert_used_max()), layers(model.layers.size()), cpu_share(std::clamp(cpu_share, 0.0f, 1.0f)) {
        if (const char * env = getenv("LLAMA_MOE_CACHE_LOG")) {
            log_every = atoi(env);
        }
        if (const char * env = getenv("LLAMA_MOE_CACHE_TRACE")) {
            trace_file = fopen(env, "w");
            if (trace_file != nullptr) {
                fprintf(trace_file, "call,layer,n_ids,n_hit,n_miss,ids\n");
            } else {
                LLAMA_LOG_WARN("%s: failed to open the MoE cache trace file '%s'\n", __func__, env);
            }
        }
        if (const char * env = getenv("LLAMA_MOE_COPY_LOG")) {
            copy_log_path = env;
        }
        if (const char * env = getenv("LLAMA_MOE_PREFETCH")) {
            prefetch = std::clamp(atoi(env), 0, 2);
        }
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                continue;
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
            if (it_dev == devices.end()) {
                continue;
            }
            const int32_t id = (int32_t) (it_dev - devices.begin());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            for (const ggml_tensor * t : experts) {
                it->host_bytes         += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        // with misses on the CPU, the n_zero slots after the cached ones stay zero: the device computes these misses as zero
        // one extra slot at the end, CUDA MMQ can read past the last expert
        n_zero = cpu_share > 0.0f ? std::min<int32_t>(n_expert_used, 64) : 0;
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + n_zero + 1), alignment);
            }
            return res;
        };

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (split_by_free) {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = (double) tensor_split[it - model.devices.begin()];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        cpu_share_q = (uint32_t) std::lround(cpu_share*65536.0f);
        if (const char * path = getenv("LLAMA_MOE_CACHE_TRACE_BIN"); path != nullptr && *path != '\0') {
            trace_bin = fopen(path, "wb");
            if (trace_bin == nullptr) {
                throw std::runtime_error(std::string("failed to open the MoE cache trace ") + path);
            }
            write_trace_header(split_sum);
        }
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
            const int32_t max_slots = g.layers.size()*n_expert;
            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) {
                g.n_slots++;
            }
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            int64_t n_expert_token = 0;
            for (int32_t il : g.layers) {
                n_expert_token += model.hparams.n_expert_used(il);
            }
            if (n_expert_token == 0) {
                g.n_slots = 0;
                continue;
            }
            // under one token's experts LRU thrashes, so evict the least used of a few random slots
            int32_t n_sample = 0;
            if (g.n_slots < n_expert_token) {
                LLAMA_LOG_WARN("%s: MoE cache is too small to hold the experts of one token for %zu layers, performance may suffer\n", __func__, g.layers.size());
                n_sample = 32;
            }
            // LLAMA_MOE_CACHE_SAMPLE sets the number of slots sampled, 0 restores plain LRU eviction
            if (const char * env = getenv("LLAMA_MOE_CACHE_SAMPLE")) {
                n_sample = std::clamp(atoi(env), 0, g.n_slots);
            }
            // LLAMA_MOE_CACHE_DIST=0 scores the victim by use count only, for A/B measurement
            bool use_dist = true;
            if (const char * env = getenv("LLAMA_MOE_CACHE_DIST")) {
                use_dist = atoi(env) != 0;
            }
            // grow the protected fraction from zero at 1x to one half at 1.8x the experts used by one token
            const double cache_ratio = (double) g.n_slots/n_expert_token;
            const double ramp_progress = std::clamp((cache_ratio - 1.0)/0.8, 0.0, 1.0);
            int32_t max_protected = (int32_t) (g.n_slots*ramp_progress*0.5);
            // LLAMA_MOE_CACHE_SLRU_RAMP=0 restores the previous fixed 50/50 SLRU split
            if (const char * env = getenv("LLAMA_MOE_CACHE_SLRU_RAMP")) {
                if (atoi(env) == 0) {
                    max_protected = g.n_slots/2;
                }
            }
            g.slru.init(model.layers.size(), n_expert, g.n_slots, max_protected, n_sample, use_dist ? g.layers.size() : 0);
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size());
            n_tensors_host  += g.layers.size()*(n_zero > 0 ? 3 : 1); // slot maps and CPU map
        }
        if (n_tensors_host == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                devices[id].ctx = init_ctx(n_tensors[id]);
            }
        }
        ctx_host  = init_ctx(n_tensors_host);
        ctx_views = init_ctx(2);

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + n_zero + 1);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots + n_zero, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = { il, false };
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
                if (n_zero > 0) {
                    l.slot_map_cpu = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.slot_map_cpu, "moe_cache.slot_map_cpu-%d", il);
                    layer_of[l.slot_map_cpu] = { il, true };
                    buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map_cpu), alignment_host);
                    l.cpu_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.cpu_map, "moe_cache.cpu_map-%d", il);
                    buf_host_size += GGML_PAD(ggml_nbytes(l.cpu_map), alignment_host);
                }
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots);
        }

        // with no_alloc, dummy buffers are attached so the memory use can still be measured, see llama_context::memory_breakdown
        for (device & d : devices) {
            if (!d.ctx) {
                continue;
            }
            d.buf = llama_memory_bufs::alloc_buf(d.ctx.get(), d.buft, no_alloc, "MoE cache");
            if (!no_alloc) {
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
        }
        buf_host = llama_memory_bufs::alloc_buf(ctx_host.get(), buft_host, no_alloc, "MoE cache slot maps");
        if (!no_alloc) {
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // the CPU maps are -1
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());

            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    if (layers[il].slot_map != nullptr) {
                        int32_t * slots = (int32_t *) layers[il].slot_map->data;
                        std::fill(slots, slots + layers[il].slot_map->ne[1], g.slru.empty);
                        g.slru.slot_map[il] = slots;
                    }
                }
            }
        }

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts, %.0f%% of the misses on the CPU\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0, 100.0*cpu_share);
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%, %.1f tokens worth)\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]),
                        (double) g.n_slots/(g.layers.size()*n_expert_used));
                }
            }
        }

        if (!no_alloc) {
            pin_host_experts();
        }
    }

    // register the host expert pages of the cached layers with the device driver, so the
    // cache fills and the large-batch uploads DMA at full bus bandwidth instead of going
    // through pageable staging copies. mmap keeps the pages file-backed; registration just
    // locks them. opt-in via GGML_CUDA_REGISTER_HOST (also checked inside the backend), and
    // any range the driver refuses simply stays pageable.
    void pin_host_experts() {
        // the backend gates on the same env; checking it here too keeps the default
        // configuration free of the range bookkeeping and of misleading logs
        const char * opt_in = getenv("GGML_CUDA_REGISTER_HOST");
        if (opt_in == nullptr || atoi(opt_in) == 0) {
            return;
        }

        // the first backend registry that exports the proc serves all its devices
        // (registration is portable); mixed-backend device sets are not handled
        for (const device & d : devices) {
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(d.backend));
            reg_host   = (bool (*)(void *, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_register_host_buffer");
            unreg_host = (void (*)(void *))         ggml_backend_reg_get_proc_address(reg, "ggml_backend_unregister_host_buffer");
            if (reg_host != nullptr) {
                break;
            }
        }
        if (reg_host == nullptr) {
            return;
        }

        // experts already in a pinned host buffer (e.g. --load-mode none) must not be
        // registered again
        std::vector<ggml_backend_buffer_type_t> & host_bufts = pin_host_bufts;
        for (const device & d : devices) {
            ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(d.backend));
            if (buft != nullptr) {
                host_bufts.push_back(buft);
            }
        }

        // page-align the ranges and merge the overlapping ones: the expert tensors of a layer
        // are adjacent in the mapping, and registering the same page twice fails
        const uintptr_t page = llama_moe_cache_page_size();
        std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
        for (const layer & l : layers) {
            if (l.ig < 0) {
                continue;
            }
            for (const ggml_tensor * t : l.experts) {
                if (std::find(host_bufts.begin(), host_bufts.end(), ggml_backend_buffer_get_type(t->buffer)) != host_bufts.end()) {
                    continue;
                }
                const uintptr_t base = (uintptr_t) t->data & ~(page - 1);
                const uintptr_t end  = ((uintptr_t) t->data + ggml_nbytes(t) + page - 1) & ~(page - 1);
                ranges.push_back({ base, end });
            }
        }
        std::vector<std::pair<uintptr_t, uintptr_t>> merged = llama_moe_cache_merge_ranges(std::move(ranges));

        // pinned pages are charged to a shared budget of about half the system RAM on
        // Windows and compete with the OS everywhere; registering everything can starve
        // later device allocations, so by default pin at most a quarter of the physical
        // RAM (8 GiB when it cannot be determined). LLAMA_MOE_PIN_MIB overrides the cap,
        // and a value <= 0 disables pinning entirely.
        size_t free_ram  = 0;
        size_t total_ram = 0;
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev != nullptr) {
            ggml_backend_dev_memory(cpu_dev, &free_ram, &total_ram);
        }
        size_t cap = total_ram > 0 ? total_ram/4 : (size_t) std::min<uint64_t>(8192ull*1024*1024, SIZE_MAX);
        if (const char * env = getenv("LLAMA_MOE_PIN_MIB")) {
            const long long mib = atoll(env);
            if (mib <= 0) {
                return;
            }
            cap = (size_t) std::min<uint64_t>((uint64_t) mib*1024*1024, SIZE_MAX);
        }

        pin_cap   = cap;
        pin_total = 0;
        for (const auto & r : merged) {
            pin_total += r.second - r.first;
        }
        if (pin_total == 0) {
            return; // every expert is in a pinned host buffer already
        }

        // LLAMA_MOE_PIN_ONLINE=N: pin nothing yet, count the experts selected by the first N small batches, then pin the most selected (no trace file needed)
        if (const char * env = getenv("LLAMA_MOE_PIN_ONLINE")) {
            online_n = std::max<long long>(0, atoll(env));
        }
        if (online_n > 0) {
            if (getenv("LLAMA_MOE_PIN_PROFILE") != nullptr) {
                LLAMA_LOG_WARN("%s: LLAMA_MOE_PIN_ONLINE is set, LLAMA_MOE_PIN_PROFILE is ignored\n", __func__);
            }
            online_use = empty_use();
            for (size_t il = 0; il < layers.size() && online_layer < 0; ++il) {
                if (layers[il].ig >= 0) {
                    online_layer = (int32_t) il;
                }
            }
            LLAMA_LOG_INFO("%s: pinning nothing until %lld small batches have run, then the experts they selected most\n", __func__, (long long) online_n);
            return;
        }

        // LLAMA_MOE_PIN_PROFILE=<trace in the LLAMA_MOE_CACHE_TRACE format>: pin the experts it selects most instead of whole tensors in address order
        if (const char * env = getenv("LLAMA_MOE_PIN_PROFILE")) {
            std::vector<std::pair<uintptr_t, uintptr_t>> by_use = ranges_by_use(env);
            if (!by_use.empty()) {
                merged = std::move(by_use);
            }
        }

        register_ranges(merged);
    }

    // register ranges in order, skipping those that do not fit under the cap or that the driver refuses, and log the result
    void register_ranges(const std::vector<std::pair<uintptr_t, uintptr_t>> & ranges) {
        // other contexts must not upload across a range between its registration and its entry in the table
        std::lock_guard<std::mutex> lock(llama_moe_cache_pinned_mutex);
        size_t pinned = 0;
        for (const auto & r : pinned_ranges) {
            pinned += r.second - r.first;
        }
        const size_t n_before = pinned_ranges.size();
        size_t refused = 0;
        for (const auto & r : ranges) {
            const size_t n = r.second - r.first;
            if (pinned + n > pin_cap) {
                continue;
            }
            if (reg_host((void *) r.first, n)) {
                pinned_ranges.push_back(r);
                pinned += n;
            } else {
                refused += n;
            }
        }
        llama_moe_cache_pinned.insert(llama_moe_cache_pinned.end(), pinned_ranges.begin() + n_before, pinned_ranges.end());
        std::sort(llama_moe_cache_pinned.begin(), llama_moe_cache_pinned.end());
        std::sort(pinned_ranges.begin(), pinned_ranges.end());
        LLAMA_LOG_INFO("%s: registered %.2f MiB of host experts as pinned memory (%.2f MiB left pageable) in %zu ranges, %.2f MiB refused\n",
            __func__, pinned/1024.0/1024.0, (pin_total - std::min(pin_total, pinned))/1024.0/1024.0, pinned_ranges.size(), refused/1024.0/1024.0);
    }

    // LLAMA_MOE_PIN_ONLINE: count the selections of a small batch, or pin by count once online_n batches are counted
    // called between the read-back of the ids and the uploads, when the backend is idle
    void pin_online(int32_t il, const std::vector<int32_t> & sel) {
        if (il == online_layer && ++online_passes > online_n) {
            const int64_t t0 = ggml_time_us();
            register_ranges(ranges_by_count(online_use));
            LLAMA_LOG_INFO("%s: pinned the experts most selected in %lld small batches in %.1f ms\n", __func__,
                (long long) online_n, (ggml_time_us() - t0)/1000.0);
            online_n = 0;
            online_use.clear();
            return;
        }
        auto & use = online_use[il];
        for (int32_t e : sel) {
            if (e >= 0 && e < (int32_t) use.size()) {
                use[e]++;
            }
        }
    }

    // a zero count per expert of each cached layer
    std::vector<std::vector<int64_t>> empty_use() const {
        std::vector<std::vector<int64_t>> use(layers.size());
        for (size_t il = 0; il < layers.size(); ++il) {
            if (layers[il].ig >= 0) {
                use[il].assign(layers[il].experts[0]->ne[2], 0);
            }
        }
        return use;
    }

    // page ranges of the experts most selected in a trace, see ranges_by_count; empty if the trace has no selections
    std::vector<std::pair<uintptr_t, uintptr_t>> ranges_by_use(const char * path) const {
        std::vector<std::vector<int64_t>> use = empty_use();

        std::ifstream in(path);
        if (!in) {
            LLAMA_LOG_WARN("%s: failed to open the MoE pin profile '%s'\n", __func__, path);
            return {};
        }
        // call,layer,n_ids,n_hit,n_miss,ids - only small batches count, large ones use most experts
        std::string line;
        size_t n_sel = 0;
        while (std::getline(in, line)) {
            long long il, n_ids;
            int off = 0;
            if (sscanf(line.c_str(), "%*lld,%lld,%lld,%*lld,%*lld,%n", &il, &n_ids, &off) != 2 || off == 0 || il < 0 || il >= (long long) use.size() || n_ids > 8*n_expert_used) {
                continue;
            }
            const char * p = line.c_str() + off;
            char * end = nullptr;
            for (long long e = strtoll(p, &end, 10); end != p; e = strtoll(p, &end, 10)) {
                if (e >= 0 && e < (long long) use[il].size()) {
                    use[il][e]++;
                    n_sel++;
                }
                p = end;
            }
        }
        if (n_sel == 0) {
            LLAMA_LOG_WARN("%s: no expert selections in the MoE pin profile '%s'\n", __func__, path);
            return {};
        }
        LLAMA_LOG_INFO("%s: pinning the experts most selected in '%s' first (%zu selections)\n", __func__, path, n_sel);
        return ranges_by_count(use);
    }

    // page ranges of the experts by selection count (all tensors of an expert together), ties and unselected experts by address, first fit within the cap, most selected first
    std::vector<std::pair<uintptr_t, uintptr_t>> ranges_by_count(const std::vector<std::vector<int64_t>> & use) const {
        const uintptr_t page = llama_moe_cache_page_size();

        struct unit {
            int64_t   use;
            uintptr_t addr;
            int32_t   il;
            int64_t   e;
        };
        std::vector<unit> units;
        for (size_t il = 0; il < layers.size(); ++il) {
            for (size_t e = 0; e < use[il].size(); ++e) {
                const ggml_tensor * t = layers[il].experts[0];
                units.push_back({ use[il][e], (uintptr_t) t->data + e*t->nb[2], (int32_t) il, (int64_t) e });
            }
        }
        std::sort(units.begin(), units.end(), [](const unit & a, const unit & b) {
            return a.use != b.use ? a.use > b.use : a.addr < b.addr;
        });

        // each range with the rank of its expert
        std::vector<std::pair<std::pair<uintptr_t, uintptr_t>, size_t>> ranges;
        size_t total = 0;
        for (size_t k = 0; k < units.size(); ++k) {
            const unit & u = units[k];
            size_t n = 0;
            const size_t n_before = ranges.size();
            for (const ggml_tensor * t : layers[u.il].experts) {
                if (std::find(pin_host_bufts.begin(), pin_host_bufts.end(), ggml_backend_buffer_get_type(t->buffer)) != pin_host_bufts.end()) {
                    continue;
                }
                const uintptr_t a    = (uintptr_t) t->data + u.e*t->nb[2];
                const uintptr_t base = a & ~(page - 1);
                const uintptr_t end  = (a + t->nb[2] + page - 1) & ~(page - 1);
                ranges.push_back({ { base, end }, k });
                n += end - base;
            }
            if (total + n > pin_cap) {
                ranges.resize(n_before);
                continue;
            }
            total += n;
        }

        // merge the ranges that overlap or touch, then order them by their best ranked expert, so that if the driver refuses some (the pinned budget) they are the least selected
        std::sort(ranges.begin(), ranges.end());
        std::vector<std::pair<size_t, std::pair<uintptr_t, uintptr_t>>> merged;
        for (const auto & r : ranges) {
            if (!merged.empty() && r.first.first <= merged.back().second.second) {
                merged.back().second.second = std::max(merged.back().second.second, r.first.second);
                merged.back().first         = std::min(merged.back().first, r.second);
            } else {
                merged.push_back({ r.second, r.first });
            }
        }
        std::sort(merged.begin(), merged.end());
        std::vector<std::pair<uintptr_t, uintptr_t>> res;
        for (const auto & m : merged) {
            res.push_back(m.second);
        }
        return res;
    }

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    // trace format, little endian: "MCT1", then u32 n_layer, n_expert_used, n_group
    // per group: u32 device, n_expert, n_layers, the layers (u32), f64 split, split_sum, u64 group host bytes, device host bytes, alignment,
    //            u32 n_tensors, the bytes of one expert of each tensor (u64)
    // then one record per slot lookup: i32 layer, experts per token, n_ids, the ids (i32)
    void write_trace_header(double split_sum) {
        auto u32 = [&](uint32_t v) { fwrite(&v, sizeof(v), 1, trace_bin); };
        auto u64 = [&](uint64_t v) { fwrite(&v, sizeof(v), 1, trace_bin); };
        auto f64 = [&](double   v) { fwrite(&v, sizeof(v), 1, trace_bin); };
        fwrite("MCT1", 1, 4, trace_bin);
        u32(layers.size());
        u32(n_expert_used);
        u32(groups.size());
        for (const group & g : groups) {
            const device & d = devices[g.id];
            u32(g.id);
            u32(g.ref[0]->ne[2]);
            u32(g.layers.size());
            for (int32_t il : g.layers) {
                u32(il);
            }
            f64(d.split);
            f64(split_sum);
            u64(g.host_bytes);
            u64(d.host_bytes);
            u64(ggml_backend_buft_get_alignment(d.buft));
            u32(g.ref.size());
            for (const ggml_tensor * t : g.ref) {
                u64(t->nb[2]);
            }
        }
    }

    void write_trace(int32_t il, int32_t n_used) {
        const int32_t rec[3] = { il, n_used, (int32_t) ids.size() };
        fwrite(rec, sizeof(int32_t), 3, trace_bin);
        fwrite(ids.data(), sizeof(int32_t), ids.size(), trace_bin);
    }

    ~impl() {
        log_stats();
        if (trace_file != nullptr) {
            fclose(trace_file);
        }
        if (trace_bin != nullptr) {
            fclose(trace_bin);
        }
        if (!copy_log_path.empty()) {
            FILE * f = fopen(copy_log_path.c_str(), "w");
            if (f != nullptr) {
                fprintf(f, "call,layer,ip,expert,n,bytes,pinned,us\n");
                for (const copy_rec & r : copy_log) {
                    fprintf(f, "%u,%d,%d,%d,%d,%u,%u,%u\n", r.call, r.layer, r.ip, r.expert, r.n, r.bytes, r.pinned, r.us);
                }
                fclose(f);
            } else {
                LLAMA_LOG_WARN("%s: failed to open the MoE copy log '%s'\n", __func__, copy_log_path.c_str());
            }
        }
        if (unreg_host != nullptr && !pinned_ranges.empty()) {
            std::lock_guard<std::mutex> lock(llama_moe_cache_pinned_mutex);
            for (const auto & r : pinned_ranges) {
                unreg_host((void *) r.first);
            }
            auto & all = llama_moe_cache_pinned;
            all.erase(std::remove_if(all.begin(), all.end(), [&](const std::pair<uintptr_t, uintptr_t> & r) {
                return std::binary_search(pinned_ranges.begin(), pinned_ranges.end(), r);
            }), all.end());
        }
    }

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    // cpu_part: the graph computes the misses in the CPU map of the layer (see get_cpu_map)
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used, bool cpu_part) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];

        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
        if (n_tokens == 0 || n_tokens > max_batch || std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > groups[l.ig].n_slots) {
            return nullptr;
        }
        GGML_ASSERT(!cpu_part || l.slot_map_cpu != nullptr);
        return cpu_part ? l.slot_map_cpu : l.slot_map;
    }

    // the experts of layer il computed on the CPU: I32 [1, n_expert] in host memory, e for these experts, -1 for the others
    // nullptr if no misses of layer il are computed on the CPU for ubatches of n_tokens tokens
    ggml_tensor * get_cpu_map(int32_t il, int64_t n_tokens) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].cpu_map == nullptr) {
            return nullptr;
        }
        // the scheduler would run the CPU part on a device that offloads such batches from host weights
        ggml_tensor probe = {};
        probe.op    = GGML_OP_MUL_MAT_ID;
        probe.type  = GGML_TYPE_F32;
        probe.ne[0] = 1;
        probe.ne[1] = n_expert_used;
        probe.ne[2] = n_tokens;
        probe.ne[3] = 1;
        for (const device & d : devices) {
            if (ggml_backend_offload_op(d.backend, &probe)) {
                return nullptr;
            }
        }
        return layers[il].cpu_map;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend) {
            return 0;
        }

        // large batches only read the cache, so the experts used in generation stay in it
        const int32_t * slots = g.slru.slot_map[b.il];
        if (slots == nullptr || slots[e] >= g.n_slots) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] < g.n_slots && slots[e + n] == slots[e] + n) {
            n++;
        }

        ggml_tensor * bank = g.banks[b.ip];
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        const int32_t il      = it->second.first;
        const bool    has_cpu = it->second.second;
        layer & l = layers[il];
        group & g = groups[l.ig];

        GGML_ASSERT(backend == devices[g.id].backend);

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, unless the lookup is the first node of its split
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));
        GGML_ASSERT(sel->op == GGML_OP_RESHAPE); // of the [n_expert_used, n_tokens] ids, see build_lookup

        const int32_t n_used = (int32_t) sel->src[0]->ne[0];

        ids.resize(ggml_nelements(sel));
        ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
        ggml_backend_synchronize(backend);

        if (trace_bin != nullptr) {
            write_trace(il, n_used);
        }

        if (online_n > 0 && ids.size() <= (size_t) 8*n_expert_used) {
            pin_online(il, ids);
        }

        // part of the misses is computed on the CPU once the cache is full, the carry keeps the share right on average
        const uint32_t share_q = has_cpu && g.slru.n_filled == g.slru.n_slots ? cpu_share_q : 0;
        auto n_cpu = [&](size_t n_miss) {
            const uint64_t acc = (uint64_t) n_miss*share_q + l.cpu_acc;
            l.cpu_acc = acc & 0xffff;
            return (size_t) (acc >> 16);
        };
        size_t n_hit = 0;
        if (!g.slru.plan(il, ids.data(), ids.size(), sel->src[0]->ne[1], n_used, n_zero, n_cpu, fills, cpu, n_hit)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }
        if (has_cpu) {
            int32_t * cpu_map = (int32_t *) l.cpu_map->data;
            for (int32_t e : l.cpu_experts) {
                cpu_map[e] = -1;
            }
            l.cpu_experts.clear();
            for (const auto & f : cpu) {
                cpu_map[f.expert] = f.expert;
                l.cpu_experts.push_back(f.expert);
            }
        }

        const bool log_copies = !copy_log_path.empty();
        const uint32_t call = (uint32_t) (n_copy_calls + 1);

        // the faults of the pageable pages are then read concurrently instead of one at a time by the uploads
        if (prefetch > 0 && !fills.empty()) {
            const int64_t t0 = ggml_time_us();
            llama_memory_ranges mr;
            size_t n_cand = 0;
            for (const ggml_tensor * w : l.experts) {
                for (const auto & f : fills) {
                    const uint8_t * p = (const uint8_t *) w->data + f.expert*w->nb[2];
                    if (llama_moe_cache_pinned_bytes(p, w->nb[2]) < w->nb[2]) {
                        mr.push_back({ p, w->nb[2] });
                        n_cand += w->nb[2];
                    }
                }
            }
            const size_t n_ranges = mr.size();
            if (prefetch == 2) {
                llama_moe_cache_drop_resident(mr);
            }
            size_t n_bytes = 0;
            for (const auto & r : mr) {
                n_bytes += r.size;
            }
            const int32_t n_kept = (int32_t) mr.size();
            llama_prefetch(std::move(mr));
            if (log_copies) {
                copy_log.push_back({ call, (int16_t) il, -1, (int32_t) n_ranges, n_kept, (uint32_t) n_bytes, (uint32_t) n_cand, (uint32_t) (ggml_time_us() - t0) });
            }
        }

        // upload the missing experts, consecutive experts going to consecutive slots are uploaded together
        size_t bytes = 0;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                    n++;
                }
                const uint8_t * p = (const uint8_t *) w->data + fills[i].expert*expert_size;
                const int64_t t0 = log_copies ? ggml_time_us() : 0;
                llama_moe_cache_set_async(backend, bank, p, fills[i].slot*expert_size, n*expert_size);
                if (log_copies) {
                    const uint32_t us = (uint32_t) (ggml_time_us() - t0);
                    copy_log.push_back({ call, (int16_t) il, (int16_t) ip, fills[i].expert, (int32_t) n, (uint32_t) (n*expert_size),
                        (uint32_t) llama_moe_cache_pinned_bytes(p, n*expert_size), us });
                }
                bytes += n*expert_size;
                i += n;
            }
        }

        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size() + cpu.size();
        st.cpu    += cpu.size();
        st.bytes  += bytes;

        n_copy_calls++;
        if (trace_file != nullptr) {
            fprintf(trace_file, "%zu,%d,%zu,%zu,%zu,", n_copy_calls, il, ids.size(), n_hit, fills.size() + cpu.size());
            for (size_t i = 0; i < ids.size(); ++i) {
                fprintf(trace_file, i == 0 ? "%d" : " %d", ids[i]);
            }
            fputc('\n', trace_file);
        }
        if (log_every > 0 && n_copy_calls % (size_t) log_every == 0) {
            LLAMA_LOG_INFO("llama_moe_cache: stats after %zu slot-map copies:\n", n_copy_calls);
            log_stats();
        }

        // the next copy synchronizes the backend before it changes the slot map again
        ggml_backend_tensor_set_async(backend, dst, l.slot_map->data, 0, ggml_nbytes(l.slot_map));

        return true;
    }

    // a large-batch upload in the copy log, as call 0: the host read these experts outside the slot-map copies
    void log_upload(const void * data, size_t size, uint32_t us) {
        const uintptr_t p = (uintptr_t) data;
        for (size_t il = 0; il < layers.size(); ++il) {
            for (size_t ip = 0; ip < layers[il].experts.size(); ++ip) {
                const ggml_tensor * w = layers[il].experts[ip];
                const uintptr_t base = (uintptr_t) w->data;
                if (p >= base && p < base + ggml_nbytes(w)) {
                    copy_log.push_back({ 0, (int16_t) il, (int16_t) ip, (int32_t) ((p - base)/w->nb[2]), (int32_t) (size/w->nb[2]),
                        (uint32_t) size, (uint32_t) llama_moe_cache_pinned_bytes(data, size), us });
                    return;
                }
            }
        }
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu (%zu on the CPU), hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, st.misses, st.cpu, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
        if (stats_upload.misses > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu uploads of uncached experts, %.2f MiB\n", stats_upload.misses, stats_upload.bytes/1024.0/1024.0);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share) :
    pimpl(new impl(model, no_alloc, backends, bufts, size, cpu_share)) {
}

llama_moe_cache::~llama_moe_cache() = default;

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

llama_moe_cache_lookup llama_moe_cache::build_lookup(const llm_graph_context & g, const llama_moe_cache_layer & layer, ggml_tensor * selected_experts) const {
    const int32_t il = layer.il;
    ggml_context * ctx0 = g.ctx0;

    // part of the misses may be computed from the host experts, but not with weights before the FFN, LoRAs or rotated experts
    bool cpu_part = !layer.weight_before_ffn && g.loras->empty();
    for (ggml_tensor * w : { layer.up_exps, layer.gate_exps, layer.down_exps, layer.gate_up_exps }) {
        cpu_part = cpu_part && !(w != nullptr && g.hdmd != nullptr && g.hdmd->rot.count(w) > 0);
    }

    ggml_tensor * cpu_map  = cpu_part ? pimpl->get_cpu_map(il, selected_experts->ne[1]) : nullptr;
    ggml_tensor * slot_map = pimpl->get_slot_map(il, selected_experts->ne[1], selected_experts->ne[0], cpu_map != nullptr);
    if (slot_map == nullptr) {
        return {};
    }
    for (ggml_tensor * w : { layer.up_exps, layer.gate_exps, layer.down_exps, layer.gate_up_exps }) {
        if (w == nullptr) {
            continue;
        }
        ggml_tensor * experts = pimpl->get_experts(w);
        if (experts == nullptr) {
            return {};
        }
        // the device of the cache must compute the matmul, else the scheduler copies the cache to another backend
        ggml_tensor * b = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, experts->ne[0], w == layer.down_exps ? selected_experts->ne[0] : 1, selected_experts->ne[1]);
        if (!ggml_backend_supports_op(pimpl->backend(il), ggml_mul_mat_id(ctx0, experts, b, selected_experts))) {
            return {};
        }
    }

    ggml_tensor * ids = selected_experts;
    if (!ggml_is_contiguous(ids)) {
        ids = ggml_cont(ctx0, ids);
    }
    ids = ggml_reshape_1d(ctx0, ids, ggml_nelements(ids));

    // the slot map is a host weight, so the scheduler starts a new split here and copies it with the copy callback
    // the callback reads the selected experts, uploads the missing ones and updates the slot map
    ggml_tensor * slots = ggml_get_rows(ctx0, slot_map, ids); // [1, n_expert_used*n_tokens]
    if (!ggml_backend_supports_op(pimpl->backend(il), slots)) {
        return {};
    }
    ggml_backend_sched_set_tensor_backend(g.sched, slots, pimpl->backend(il));
    g.cb(slots, "ffn_moe_slots", il);

    // add the lookup now: the transform of a Hadamard-folded expert reads host weights and would start its split first
    ggml_build_forward_expand(g.gf, slots);

    // the misses computed from the host experts, read after the lookup has picked them
    ggml_tensor * cpu_ids = nullptr;
    if (cpu_map != nullptr) {
        cpu_ids = ggml_get_rows(ctx0, cpu_map, ids); // [1, n_expert_used*n_tokens]
        cpu_ids = ggml_reshape_2d(ctx0, cpu_ids, selected_experts->ne[0], selected_experts->ne[1]);
    }

    return { il, ggml_reshape_2d(ctx0, slots, selected_experts->ne[0], selected_experts->ne[1]), cpu_ids };
}

ggml_tensor * llama_moe_cache::build_experts(const llm_graph_context & g, const llama_moe_cache_lookup & lookup,
        ggml_tensor * cur, ggml_tensor * selected_experts, llama_moe_cache_build_experts_t build, void * user_data) const {
    GGML_ASSERT(lookup.slots != nullptr);

    ggml_tensor * experts = build(cur, selected_experts, lookup.slots, true, user_data);
    if (lookup.cpu_ids != nullptr) {
        // the cached experts and the misses computed from the host experts, each reads the others as zero
        // the cached experts come first in the graph, so that they are computed while the misses are
        ggml_build_forward_expand(g.gf, experts);
        experts = ggml_add(g.ctx0, experts, build(cur, selected_experts, lookup.cpu_ids, false, user_data));
        g.cb(experts, "ffn_moe_down_merged", lookup.il);
    }

    return experts;
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

void llama_moe_cache::upload(ggml_backend_t backend, ggml_tensor * dst, const void * data, size_t offset, size_t size) {
    const int64_t t0 = pimpl->copy_log_path.empty() ? 0 : ggml_time_us();
    llama_moe_cache_set_async(backend, dst, data, offset, size);
    pimpl->stats_upload.misses++;
    pimpl->stats_upload.bytes += size;
    if (!pimpl->copy_log_path.empty()) {
        pimpl->log_upload(data, size, (uint32_t) (ggml_time_us() - t0));
    }
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    return res;
}

void llama_moe_cache_set_async(ggml_backend_t backend, ggml_tensor * dst, const void * data, size_t offset, size_t size) {
    // also split where a copy enters a pinned range, so that its pinned part goes by DMA
    std::lock_guard<std::mutex> lock(llama_moe_cache_pinned_mutex);
    const auto & pinned = llama_moe_cache_pinned;
    uintptr_t p = (uintptr_t) data;
    while (size > 0) {
        size_t n = size;
        const auto it = std::upper_bound(pinned.begin(), pinned.end(), p, [](uintptr_t v, const std::pair<uintptr_t, uintptr_t> & r) { return v < r.second; });
        if (it != pinned.end()) {
            n = std::min<size_t>(n, it->first <= p ? it->second - p : it->first - p);
        }
        ggml_backend_tensor_set_async(backend, dst, (const void *) p, offset, n);
        p      += n;
        offset += n;
        size   -= n;
    }
}

bool llama_moe_cache_has_pinned() {
    std::lock_guard<std::mutex> lock(llama_moe_cache_pinned_mutex);
    return !llama_moe_cache_pinned.empty();
}
