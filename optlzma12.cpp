#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <divsufsort.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

// ============================================================================
// LZMA Constants & Configuration
// ============================================================================
constexpr int kNumStates = 12;
constexpr int MIN_MATCH = 2;
constexpr int MIN_MATCH_SA = 5;
constexpr int MAX_MATCH = 273;
constexpr int REPS = 4;
constexpr uint32_t RC_INFINITY_PRICE = 0x7FFFFFFF;
constexpr uint32_t CHUNK_SIZE = 4096;
constexpr int kMatchFinderDepth = 1024;

// ============================================================================
// Open Addressing Hash Configuration (24 to 39 bits)
// ============================================================================
// EVICT_THRESHOLD: Maximum distance a match is allowed to be considered valid.
// Increased to 8MB (8,388,608 bytes). Entries older than this are treated as 
// tombstones/empty slots and evicted, naturally keeping the table sparse.
constexpr uint32_t EVICT_THRESHOLD = 8 * 1024 * 1024; // 8,388,608

// HASH_SIZE ESTIMATION:
// To maintain O(1) near-instant insertion/lookup in an open addressing table,
// the maximum load factor (α) should be ≤ 0.25. 
// Since we evict entries older than 8.38M positions, there are at most 8.38M 
// valid entries at any given time per bit-length table.
// Target Size = 8,388,608 * 4 = 33,554,432 (which is exactly 2^25).
constexpr uint32_t HASH_BITS = 25;
constexpr uint32_t HASH_SIZE = 1 << HASH_BITS; // 33,554,432
constexpr uint32_t HASH_MASK = HASH_SIZE - 1;

// Memory Footprint Note:
// 16 tables (for 24 to 39 bits inclusive) * 33,554,432 entries * 4 bytes per entry 
// = ~2.14 Gigabytes strictly for the open addressing hash maps.

// The MSB (highest bit) marks an empty or uninitialized slot.
// This restricts the maximum supported file size to 2 GB (0x7FFFFFFF).
constexpr uint32_t EMPTY_SLOT = 0x80000000;

struct LZMAParams {
    int lc = 3;
    int lp = 0;
    int pb = 2;
};

// ============================================================================
// Entropy Pricing Subsystem
// ============================================================================
uint32_t ProbPrices[2048];

void InitProbPrices() {
    for (int i = 1; i < 2047; i++) {
        ProbPrices[i] = (uint32_t)(-std::log2((double)i / 2048.0) * 128.0);
    }
    ProbPrices[0] = ProbPrices[1];
    ProbPrices[2047] = ProbPrices[2046];
}

inline uint32_t GetPrice(uint16_t prob, uint32_t bit) {
    return ProbPrices[bit ? (2048 - prob) : prob];
}

uint32_t TreePrice(const uint16_t* probs, int numBits, uint32_t symbol) {
    uint32_t price = 0, m = 1;
    for (int i = numBits - 1; i >= 0; i--) {
        uint32_t bit = (symbol >> i) & 1;
        price += GetPrice(probs[m], bit);
        m = (m << 1) | bit;
    }
    return price;
}

uint32_t ReverseTreePrice(const uint16_t* probs, int numBits, uint32_t symbol) {
    uint32_t price = 0, m = 1;
    for (int i = 0; i < numBits; i++) {
        uint32_t bit = (symbol >> i) & 1;
        price += GetPrice(probs[m], bit);
        m = (m << 1) | bit;
    }
    return price;
}

// 64-bit Fibonacci Hashing
// Mixes up to 39 bits of exact state down to our 25-bit table size.
// The multiplier is the golden ratio for 64-bit integers, ensuring optimal 
// bit diffusion across the upper bounds.
inline uint32_t HashBitPrefix(uint64_t val) {
    uint64_t hash = val * 0x9E3779B97F4A7C15ULL;
    return (uint32_t)(hash >> (64 - HASH_BITS));
}

// ============================================================================
// LZMA Probability Context Models
// ============================================================================
struct LZMAProbabilities {
    uint16_t is_match[kNumStates * 16];
    uint16_t is_rep[kNumStates];
    uint16_t is_rep_g0[kNumStates];
    uint16_t is_rep_g1[kNumStates];
    uint16_t is_rep_g2[kNumStates];
    uint16_t is_rep0_long[kNumStates * 16];
    uint16_t pos_slot[4 * 64];
    uint16_t pos_spec_tree[10 * 64];
    uint16_t prob_align[16];
    std::vector<uint16_t> prob_literals;

    struct LenTree {
        uint16_t choice1 = 1024, choice2 = 1024;
        std::vector<uint16_t> low, mid, high;
        void Init(int pb) {
            low.assign((1 << pb) * 16, 1024);
            mid.assign((1 << pb) * 16, 1024);
            high.assign(256, 1024);
        }
    } match_len, rep_len;

    void Init(int lc, int lp, int pb) {
        std::fill(std::begin(is_match), std::end(is_match), 1024);
        std::fill(std::begin(is_rep), std::end(is_rep), 1024);
        std::fill(std::begin(is_rep_g0), std::end(is_rep_g0), 1024);
        std::fill(std::begin(is_rep_g1), std::end(is_rep_g1), 1024);
        std::fill(std::begin(is_rep_g2), std::end(is_rep_g2), 1024);
        std::fill(std::begin(is_rep0_long), std::end(is_rep0_long), 1024);
        std::fill(std::begin(pos_slot), std::end(pos_slot), 1024);
        std::fill(std::begin(pos_spec_tree), std::end(pos_spec_tree), 1024);
        std::fill(std::begin(prob_align), std::end(prob_align), 1024);
        prob_literals.assign((1 << (lc + lp)) * 0x300, 1024);
        match_len.Init(pb);
        rep_len.Init(pb);
    }
};

uint32_t GetLenPrice(const LZMAProbabilities::LenTree& tree, uint32_t len, uint32_t posState) {
    uint32_t symbol = len - MIN_MATCH;
    uint32_t price = 0;
    if (symbol < 8) {
        price += GetPrice(tree.choice1, 0) + TreePrice(&tree.low[posState * 16], 3, symbol);
    } else if (symbol < 16) {
        price += GetPrice(tree.choice1, 1) + GetPrice(tree.choice2, 0) + TreePrice(&tree.mid[posState * 16], 3, symbol - 8);
    } else {
        price += GetPrice(tree.choice1, 1) + GetPrice(tree.choice2, 1) + TreePrice(tree.high.data(), 8, symbol - 16);
    }
    return price;
}

uint32_t GetLiteralPrice(uint8_t symbol, uint8_t matchByte, uint32_t state, const uint16_t* probs) {
    uint32_t price = 0, m = 1;
    if (state >= 7) {
        for (int i = 7; i >= 0; i--) {
            uint32_t matchBit = (matchByte >> i) & 1;
            uint32_t bit = (symbol >> i) & 1;
            price += GetPrice(probs[(0x100 + (matchBit << 8)) + m], bit);
            m = (m << 1) | bit;
            if (matchBit != bit) {
                i--;
                for (; i >= 0; i--) {
                    uint32_t b = (symbol >> i) & 1;
                    price += GetPrice(probs[m], b);
                    m = (m << 1) | b;
                }
                break;
            }
        }
    } else {
        for (int i = 7; i >= 0; i--) {
            uint32_t bit = (symbol >> i) & 1;
            price += GetPrice(probs[m], bit);
            m = (m << 1) | bit;
        }
    }
    return price;
}

// ============================================================================
// LZMA Range Encoder
// ============================================================================
struct LZMAEncoder {
    std::ostream& out_stream;
    uint64_t low = 0;
    uint32_t range = 0xFFFFFFFF;
    uint8_t cache = 0;
    uint32_t cache_size = 1;
    uint8_t out_buf[8192];
    size_t buf_pos = 0;

    LZMAEncoder(std::ostream& os) : out_stream(os) {}

    void WriteByte(uint8_t b) {
        out_buf[buf_pos++] = b;
        if (buf_pos == sizeof(out_buf)) FlushStream();
    }

    void FlushStream() {
        if (buf_pos > 0) { out_stream.write(reinterpret_cast<const char*>(out_buf), buf_pos); buf_pos = 0; }
    }

    void ShiftLow() {
        if ((uint32_t)low < 0xFF000000 || (uint32_t)(low >> 32) != 0) {
            uint8_t temp = cache;
            do { WriteByte((uint8_t)(temp + (uint8_t)(low >> 32))); temp = 0xFF; } while (--cache_size != 0);
            cache = (uint8_t)((uint32_t)low >> 24);
        }
        cache_size++;
        low = (uint32_t)low << 8;
    }

    void WriteHeader(uint64_t uncompressed_size, const LZMAParams& params) {
        WriteByte((params.pb * 5 + params.lp) * 9 + params.lc);
        uint32_t dict_size = 4096;
        while (dict_size < uncompressed_size && dict_size < (1 << 23)) dict_size <<= 1;
        for (int i = 0; i < 4; i++) WriteByte((dict_size >> (i * 8)) & 0xFF);
        for (int i = 0; i < 8; i++) WriteByte((uncompressed_size >> (i * 8)) & 0xFF);
    }

    void EncodeBit(uint16_t& prob, int bit) {
        uint32_t newBound = (range >> 11) * prob;
        if (bit == 0) { range = newBound; prob += ((1 << 11) - prob) >> 5; } 
        else { low += newBound; range -= newBound; prob -= prob >> 5; }
        while (range < (1 << 24)) { range <<= 8; ShiftLow(); }
    }

    void EncodeDirectBits(uint32_t value, int numBits) {
        for (int i = numBits - 1; i >= 0; i--) {
            range >>= 1;
            if ((value >> i) & 1) low += range;
            while (range < (1 << 24)) { range <<= 8; ShiftLow(); }
        }
    }

    void EncodeTree(uint16_t* probs, int numBits, uint32_t symbol) {
        uint32_t m = 1;
        for (int i = numBits - 1; i >= 0; i--) {
            uint32_t bit = (symbol >> i) & 1;
            EncodeBit(probs[m], bit);
            m = (m << 1) | bit;
        }
    }

    void EncodeReverseTree(uint16_t* probs, int numBits, uint32_t symbol) {
        uint32_t m = 1;
        for (int i = 0; i < numBits; i++) {
            uint32_t bit = (symbol >> i) & 1;
            EncodeBit(probs[m], bit);
            m = (m << 1) | bit;
        }
    }

    void EncodeLen(LZMAProbabilities::LenTree& tree, uint32_t len, uint32_t posState) {
        uint32_t symbol = len - MIN_MATCH;
        if (symbol < 8) { 
            EncodeBit(tree.choice1, 0); EncodeTree(&tree.low[posState * 16], 3, symbol); 
        } else if (symbol < 16) { 
            EncodeBit(tree.choice1, 1); EncodeBit(tree.choice2, 0); EncodeTree(&tree.mid[posState * 16], 3, symbol - 8); 
        } else { 
            EncodeBit(tree.choice1, 1); EncodeBit(tree.choice2, 1); EncodeTree(tree.high.data(), 8, symbol - 16); 
        }
    }

    void Flush() { for (int i = 0; i < 5; i++) ShiftLow(); FlushStream(); }
};

inline uint32_t GetPosSlot(uint32_t v) {
    if (v < 4) return v;
    uint32_t temp = v; int numBits = 0;
    while (temp >= 2) { temp >>= 1; numBits++; }
    return (numBits << 1) | ((v >> (numBits - 1)) & 1);
}

// DP Graph Node
struct OptimalNode {
    uint32_t price = RC_INFINITY_PRICE; 
    uint32_t pos_prev = 0;
    uint32_t back_prev = 0xFFFFFFFF;
    uint32_t state = 0;
    uint32_t backs[REPS] = {1, 1, 1, 1};
};

struct Token { uint32_t len, back_prev; };

// ============================================================================
// Compression Engine
// ============================================================================
void CompressLZMA(const std::vector<uint8_t>& data, std::ostream& out_stream, const LZMAParams& params) {
    uint32_t n = data.size();
    LZMAEncoder rc(out_stream);
    rc.WriteHeader(n, params);

    if (n == 0) { rc.Flush(); return; }
    
    // Safety boundary logic for open-addressing table MSB utilization
    if (n >= 0x80000000) {
        std::cerr << "File exceeds 2GB limit for open-addressing scheme." << std::endl;
        return;
    }

    std::vector<saidx_t> SA(n);
    std::vector<int> ISA(n), LCP(n);
    divsufsort(data.data(), SA.data(), n);
    for (uint32_t i = 0; i < n; i++) ISA[SA[i]] = i;
    int h = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (ISA[i] > 0) {
            int j = SA[ISA[i] - 1];
            while (i + h < n && j + h < n && data[i + h] == data[j + h]) h++;
            LCP[ISA[i]] = h;
            if (h > 0) h--;
        } else LCP[ISA[i]] = 0;
    }

    LZMAProbabilities probs;
    probs.Init(params.lc, params.lp, params.pb);

    std::vector<OptimalNode> opts(CHUNK_SIZE + 1);
    std::vector<uint32_t> best_dist(MAX_MATCH * 8 + 8, 0xFFFFFFFF);

    // Exact Direct Tables [16..23 bits]
    std::vector<std::vector<uint32_t>> bit_history(8);
    for (int i = 0; i < 8; ++i) bit_history[i].assign(1 << (16 + i), 0xFFFFFFFF);

    // Open Addressing Hash Tables [24..39 bits] 
    // Spans 16 distinct bit lengths (24 inclusive to 39 inclusive).
    std::vector<std::vector<uint32_t>> hash_history(16);
    for (int i = 0; i < 16; ++i) {
        // Initializes with highest bit set to flag the slot as naturally empty
        hash_history[i].assign(HASH_SIZE, EMPTY_SLOT);
    }

    uint32_t current_state = 0;
    uint32_t current_reps[REPS] = {1, 1, 1, 1};
    uint32_t numPosStates = 1 << params.pb;
    uint32_t posStateMask = numPosStates - 1;

    for (uint32_t chunk_start = 0; chunk_start < n;) {
        uint32_t chunk_len = std::min(CHUNK_SIZE, n - chunk_start);

        for (uint32_t i = 0; i <= chunk_len; i++) opts[i].price = RC_INFINITY_PRICE;
        opts[0].price = 0; opts[0].state = current_state;
        for (int r = 0; r < REPS; r++) opts[0].backs[r] = current_reps[r];

        for (uint32_t i = 0; i < chunk_len; i++) {
            if (opts[i].price == RC_INFINITY_PRICE) continue;

            uint32_t global_pos = chunk_start + i;
            uint32_t posState = global_pos & posStateMask;
            uint32_t state = opts[i].state;
            const uint32_t* backs = opts[i].backs;

            uint8_t prevByte = global_pos > 0 ? data[global_pos - 1] : 0;
            uint8_t curByte = data[global_pos];
            uint32_t max_limit = std::min((uint32_t)MAX_MATCH, chunk_len - i);

            uint32_t fill_limit = max_limit * 8 + 8;
            std::fill(best_dist.begin(), best_dist.begin() + fill_limit, 0xFFFFFFFF);

            // ====================================================================
            // MATCH FINDER A: Suffix Array Search (Large Matches)
            // ====================================================================
            int rank = ISA[global_pos];
            int lcp_min = MAX_MATCH, depth = 0;

            for (int k = rank; k > 0; k--) {
                lcp_min = std::min(lcp_min, LCP[k]);
                if (lcp_min < MIN_MATCH_SA) break;
                saidx_t sa_pos = SA[k - 1];
                if (sa_pos < (saidx_t)global_pos) {
                    uint32_t dist = global_pos - sa_pos;
                    uint32_t len_bound = std::min((uint32_t)lcp_min, max_limit);
                    uint32_t match_bits = 0;
                    if (len_bound < max_limit) {
                        uint8_t a = data[global_pos + len_bound], b = data[global_pos - dist + len_bound];
                        while (match_bits < 8 && ((a >> (7 - match_bits)) & 1) == ((b >> (7 - match_bits)) & 1)) match_bits++;
                    }
                    uint32_t bit_len = len_bound * 8 + match_bits;
                    for (uint32_t bl = MIN_MATCH * 8; bl <= bit_len; bl++) {
                        if (dist < best_dist[bl]) best_dist[bl] = dist;
                    }
                }
                if (++depth >= kMatchFinderDepth) break;
            }

            lcp_min = MAX_MATCH; depth = 0;
            for (int k = rank; k < (int)n - 1; k++) {
                lcp_min = std::min(lcp_min, LCP[k + 1]);
                if (lcp_min < MIN_MATCH_SA) break;
                saidx_t sa_pos = SA[k + 1];
                if (sa_pos < (saidx_t)global_pos) {
                    uint32_t dist = global_pos - sa_pos;
                    uint32_t len_bound = std::min((uint32_t)lcp_min, max_limit);
                    uint32_t match_bits = 0;
                    if (len_bound < max_limit) {
                        uint8_t a = data[global_pos + len_bound], b = data[global_pos - dist + len_bound];
                        while (match_bits < 8 && ((a >> (7 - match_bits)) & 1) == ((b >> (7 - match_bits)) & 1)) match_bits++;
                    }
                    uint32_t bit_len = len_bound * 8 + match_bits;
                    for (uint32_t bl = MIN_MATCH * 8; bl <= bit_len; bl++) {
                        if (dist < best_dist[bl]) best_dist[bl] = dist;
                    }
                }
                if (++depth >= kMatchFinderDepth) break;
            }

            // ====================================================================
            // MATCH FINDER B: Direct Tables [16..23 bits]
            // ====================================================================
            if (global_pos + 2 < n) {
                uint32_t val24 = ((uint32_t)data[global_pos] << 16) | ((uint32_t)data[global_pos + 1] << 8) | data[global_pos + 2];
                for (int bit_len = 16; bit_len <= 23; ++bit_len) {
                    uint32_t prefix = (val24 >> (24 - bit_len)) & ((1U << bit_len) - 1);
                    uint32_t prev_pos = bit_history[bit_len - 16][prefix];
                    if (prev_pos != 0xFFFFFFFF && prev_pos < global_pos) {
                        uint32_t dist = global_pos - prev_pos;
                        if (dist < best_dist[bit_len]) best_dist[bit_len] = dist;
                    }
                    bit_history[bit_len - 16][prefix] = global_pos;
                }
            }

            // ====================================================================
            // MATCH FINDER C: Open Addressing Hash Table [24..39 bits]
            // ====================================================================
            // We require 5 bytes (40 bits) of lookahead to support up to 39-bit match validation.
            if (global_pos + 4 < n) {
                // Buffer the next 5 bytes into a 64-bit integer
                uint64_t val40 = ((uint64_t)data[global_pos] << 32) |
                                 ((uint64_t)data[global_pos + 1] << 24) |
                                 ((uint64_t)data[global_pos + 2] << 16) |
                                 ((uint64_t)data[global_pos + 3] << 8) |
                                 ((uint64_t)data[global_pos + 4]);

                for (int bit_len = 24; bit_len <= 39; ++bit_len) {
                    // Extract exactly `bit_len` MSB bits to form our state key
                    uint64_t prefix = val40 >> (40 - bit_len);
                    uint32_t h_idx = HashBitPrefix(prefix);
                    uint32_t insert_idx = 0xFFFFFFFF;

                    // Execute Open Addressing with Linear Probing
                    while (true) {
                        uint32_t entry = hash_history[bit_len - 24][h_idx];
                        
                        // Condition 1: Untouched Slot Check
                        // If the high bit is set, the slot is definitively empty. We break out
                        // and write our current position here.
                        if ((entry & EMPTY_SLOT) != 0) {
                            if (insert_idx == 0xFFFFFFFF) insert_idx = h_idx;
                            break; 
                        }

                        // Reconstruct raw absolute position (guaranteed under 2GB)
                        uint32_t prev_pos = entry & ~EMPTY_SLOT;
                        uint32_t dist = global_pos - prev_pos;

                        // Condition 2: Evaluate 8MB Eviction Threshold
                        // If the historical occurrence is too old, it violates LZMA sliding window limits.
                        // We mark this index for overwriting but MUST continue probing, 
                        // as a newer occurrence of our exact string might exist further down the probe chain.
                        if (dist > EVICT_THRESHOLD) {
                            if (insert_idx == 0xFFFFFFFF) insert_idx = h_idx;
                        } 
                        else {
                            // Condition 3: Exact Collision Verification
                            // The entry is valid and recent. We reconstruct the 40-bit string 
                            // natively from the raw data buffer at the historic location to eliminate hash collisions.
                            uint64_t prev_val40 = ((uint64_t)data[prev_pos] << 32) |
                                                  ((uint64_t)data[prev_pos + 1] << 24) |
                                                  ((uint64_t)data[prev_pos + 2] << 16) |
                                                  ((uint64_t)data[prev_pos + 3] << 8) |
                                                  ((uint64_t)data[prev_pos + 4]);
                            uint64_t prev_prefix = prev_val40 >> (40 - bit_len);

                            if (prev_prefix == prefix) {
                                // Bit-exact match validated. Commit the distance.
                                if (dist < best_dist[bit_len]) best_dist[bit_len] = dist;
                                // We overwrite the matching key position so future queries get the shortest distance.
                                insert_idx = h_idx;
                                break;
                            }
                        }
                        
                        // Linear probe increment (wraps cleanly due to power-of-two modulo mask)
                        h_idx = (h_idx + 1) & HASH_MASK;
                    }

                    // Register new observation (stripping away any empty-slot metadata inherently since val < 2GB)
                    hash_history[bit_len - 24][insert_idx] = global_pos;
                }
            }

            // Downward Monotonicity Propagation:
            // A verified 39-bit match guarantees all smaller prefixes (38, 37...) are inherently matched.
            for (int bl = 39; bl > 16; --bl) {
                if (best_dist[bl] < best_dist[bl - 1]) best_dist[bl - 1] = best_dist[bl];
            }

            // ====================================================================
            // DP STEP 1: Literal Option
            // ====================================================================
            uint32_t litState = ((global_pos & ((1 << params.lp) - 1)) << params.lc) | (prevByte >> (8 - params.lc));
            uint8_t matchByte = global_pos > backs[0] ? data[global_pos - backs[0]] : 0;
            uint32_t lit_price = opts[i].price + GetPrice(probs.is_match[state * numPosStates + posState], 0) +
                                 GetLiteralPrice(curByte, matchByte, state, &probs.prob_literals[litState * 0x300]);

            if (lit_price < opts[i + 1].price) {
                opts[i + 1].price = lit_price; opts[i + 1].pos_prev = i; opts[i + 1].back_prev = 0xFFFFFFFF;
                opts[i + 1].state = state < 4 ? 0 : (state < 10 ? state - 3 : state - 6);
                for (int r = 0; r < REPS; r++) opts[i + 1].backs[r] = backs[r];
            }

            if (max_limit >= MIN_MATCH) {
                uint32_t match_prob_price = opts[i].price + GetPrice(probs.is_match[state * numPosStates + posState], 1);

                // ================================================================
                // DP STEP 2: Repetition Options
                // ================================================================
                uint32_t rep_prob_price = match_prob_price + GetPrice(probs.is_rep[state], 1);
                for (int rep_index = 0; rep_index < REPS; rep_index++) {
                    uint32_t dist = backs[rep_index];
                    if (global_pos >= dist) {
                        uint32_t len = 0;
                        while (len < max_limit && data[global_pos + len] == data[global_pos - dist + len]) len++;

                        if (len >= MIN_MATCH || (rep_index == 0 && len == 1)) {
                            uint32_t rep_price = rep_prob_price;
                            if (rep_index == 0) {
                                rep_price += GetPrice(probs.is_rep_g0[state], 0);
                                if (len == 1) {
                                    uint32_t short_price = rep_price + GetPrice(probs.is_rep0_long[state * numPosStates + posState], 0);
                                    if (short_price < opts[i + 1].price) {
                                        opts[i + 1].price = short_price; opts[i + 1].pos_prev = i; opts[i + 1].back_prev = 0;
                                        opts[i + 1].state = state < 7 ? 9 : 11;
                                        for (int r = 0; r < REPS; r++) opts[i + 1].backs[r] = backs[r];
                                    }
                                    continue;
                                }
                                rep_price += GetPrice(probs.is_rep0_long[state * numPosStates + posState], 1);
                            } else {
                                rep_price += GetPrice(probs.is_rep_g0[state], 1);
                                if (rep_index == 1) rep_price += GetPrice(probs.is_rep_g1[state], 0);
                                else rep_price += GetPrice(probs.is_rep_g1[state], 1) + GetPrice(probs.is_rep_g2[state], rep_index - 2);
                            }

                            for (uint32_t l = MIN_MATCH; l <= len; l++) {
                                uint32_t current_rep_price = rep_price + GetLenPrice(probs.rep_len, l, posState);
                                if (current_rep_price < opts[i + l].price) {
                                    opts[i + l].price = current_rep_price; opts[i + l].pos_prev = i; opts[i + l].back_prev = rep_index;
                                    opts[i + l].state = state < 7 ? 8 : 11;
                                    opts[i + l].backs[0] = dist;
                                    if (rep_index == 1) { opts[i+l].backs[1]=backs[0]; opts[i+l].backs[2]=backs[2]; opts[i+l].backs[3]=backs[3]; }
                                    else if (rep_index == 2) { opts[i+l].backs[1]=backs[0]; opts[i+l].backs[2]=backs[1]; opts[i+l].backs[3]=backs[3]; }
                                    else if (rep_index == 3) { opts[i+l].backs[1]=backs[0]; opts[i+l].backs[2]=backs[1]; opts[i+l].backs[3]=backs[2]; }
                                    else for(int r=1; r<REPS; r++) opts[i+l].backs[r] = backs[r];
                                }
                            }
                        }
                    }
                }

                // ================================================================
                // DP STEP 3: Normal Match
                // ================================================================
                uint32_t normal_match_price = match_prob_price + GetPrice(probs.is_rep[state], 0);
                for (uint32_t l = MIN_MATCH; l <= max_limit; l++) {
                    uint32_t best_dist_for_l = 0xFFFFFFFF, best_lookahead_price = RC_INFINITY_PRICE, best_m_price = RC_INFINITY_PRICE, prev_dist = 0xFFFFFFFF;

                    for (int b = 7; b >= 0; b--) {
                        uint32_t dist = best_dist[l * 8 + b];
                        if (dist == 0xFFFFFFFF || dist == prev_dist) continue;
                        prev_dist = dist;

                        uint32_t distSymbol = dist - 1;
                        uint32_t lenState = std::min(l - 2, (uint32_t)3);
                        uint32_t slot = GetPosSlot(distSymbol);
                        uint32_t m_price = normal_match_price + GetLenPrice(probs.match_len, l, posState) + TreePrice(&probs.pos_slot[lenState * 64], 6, slot);

                        if (slot >= 4) {
                            int footerBits = (slot >> 1) - 1;
                            uint32_t base = (2 | (slot & 1)) << footerBits;
                            uint32_t directVal = distSymbol - base;
                            if (slot < 14) m_price += ReverseTreePrice(&probs.pos_spec_tree[(slot - 4) * 64], footerBits, directVal);
                            else m_price += (footerBits - 4) * 128 + ReverseTreePrice(probs.prob_align, 4, directVal & 15);
                        }

                        uint32_t lookahead_price = m_price;
                        if (l < max_limit) {
                            uint32_t state2 = state < 7 ? 7 : 10;
                            uint32_t posState2 = (global_pos + l) & posStateMask;
                            lookahead_price += GetPrice(probs.is_match[state2 * numPosStates + posState2], 0);

                            uint8_t curByte2 = data[global_pos + l], matchByte2 = data[global_pos + l - dist], prevByte2 = data[global_pos + l - 1];
                            uint32_t litState2 = (((global_pos + l) & ((1 << params.lp) - 1)) << params.lc) | (prevByte2 >> (8 - params.lc));
                            lookahead_price += GetLiteralPrice(curByte2, matchByte2, state2, &probs.prob_literals[litState2 * 0x300]);
                        }

                        if (lookahead_price < best_lookahead_price) {
                            best_lookahead_price = lookahead_price; best_m_price = m_price; best_dist_for_l = dist;
                        }
                    }

                    if (best_dist_for_l != 0xFFFFFFFF && best_m_price < opts[i + l].price) {
                        opts[i + l].price = best_m_price; opts[i + l].pos_prev = i; opts[i + l].back_prev = best_dist_for_l + REPS;
                        opts[i + l].state = state < 7 ? 7 : 10; opts[i + l].backs[0] = best_dist_for_l;
                        for (int r = 1; r < REPS; r++) opts[i + l].backs[r] = backs[r - 1];
                    }
                }
            }
        }

        uint32_t trace = chunk_len;
        std::vector<Token> chunk_tokens;
        while (trace > 0) {
            uint32_t prev = opts[trace].pos_prev;
            chunk_tokens.push_back({trace - prev, opts[trace].back_prev});
            trace = prev;
        }
        std::reverse(chunk_tokens.begin(), chunk_tokens.end());

        uint32_t global_pos = chunk_start;
        for (const Token& token : chunk_tokens) {
            uint32_t posState = global_pos & posStateMask;
            uint8_t prevByte = global_pos > 0 ? data[global_pos - 1] : 0;

            if (token.back_prev == 0xFFFFFFFF) {
                rc.EncodeBit(probs.is_match[current_state * numPosStates + posState], 0);
                uint8_t symbol = data[global_pos];
                uint32_t litState = ((global_pos & ((1 << params.lp) - 1)) << params.lc) | (prevByte >> (8 - params.lc));
                uint16_t* lit_probs = &probs.prob_literals[litState * 0x300];
                uint32_t m = 1;

                if (current_state >= 7) {
                    uint8_t matchByte = data[global_pos - current_reps[0]];
                    for (int i = 7; i >= 0; i--) {
                        uint32_t matchBit = (matchByte >> i) & 1, bit = (symbol >> i) & 1;
                        rc.EncodeBit(lit_probs[(0x100 + (matchBit << 8)) + m], bit); m = (m << 1) | bit;
                        if (matchBit != bit) {
                            i--;
                            for (; i >= 0; i--) { uint32_t b = (symbol >> i) & 1; rc.EncodeBit(lit_probs[m], b); m = (m << 1) | b; }
                            break;
                        }
                    }
                } else {
                    for (int i = 7; i >= 0; i--) { uint32_t bit = (symbol >> i) & 1; rc.EncodeBit(lit_probs[m], bit); m = (m << 1) | bit; }
                }
                current_state = current_state < 4 ? 0 : (current_state < 10 ? current_state - 3 : current_state - 6);
                global_pos++;
            }
            else if (token.back_prev < REPS) {
                rc.EncodeBit(probs.is_match[current_state * numPosStates + posState], 1); rc.EncodeBit(probs.is_rep[current_state], 1);
                if (token.back_prev == 0) {
                    rc.EncodeBit(probs.is_rep_g0[current_state], 0);
                    if (token.len == 1) {
                        rc.EncodeBit(probs.is_rep0_long[current_state * numPosStates + posState], 0);
                        current_state = current_state < 7 ? 9 : 11; global_pos++; continue;
                    }
                    rc.EncodeBit(probs.is_rep0_long[current_state * numPosStates + posState], 1);
                } else {
                    rc.EncodeBit(probs.is_rep_g0[current_state], 1);
                    if (token.back_prev == 1) rc.EncodeBit(probs.is_rep_g1[current_state], 0);
                    else { rc.EncodeBit(probs.is_rep_g1[current_state], 1); rc.EncodeBit(probs.is_rep_g2[current_state], token.back_prev - 2); }
                }
                rc.EncodeLen(probs.rep_len, token.len, posState);
                uint32_t dist = current_reps[token.back_prev];
                for (int r = token.back_prev; r > 0; r--) current_reps[r] = current_reps[r - 1];
                current_reps[0] = dist; current_state = current_state < 7 ? 8 : 11; global_pos += token.len;
            } else {
                rc.EncodeBit(probs.is_match[current_state * numPosStates + posState], 1); rc.EncodeBit(probs.is_rep[current_state], 0);
                rc.EncodeLen(probs.match_len, token.len, posState);

                uint32_t dist = token.back_prev - REPS, distSymbol = dist - 1;
                uint32_t lenState = std::min(token.len - 2, (uint32_t)3), slot = GetPosSlot(distSymbol);

                rc.EncodeTree(&probs.pos_slot[lenState * 64], 6, slot);
                if (slot >= 4) {
                    int footerBits = (slot >> 1) - 1;
                    uint32_t base = (2 | (slot & 1)) << footerBits;
                    uint32_t directVal = distSymbol - base;
                    if (slot < 14) rc.EncodeReverseTree(&probs.pos_spec_tree[(slot - 4) * 64], footerBits, directVal);
                    else { rc.EncodeDirectBits(directVal >> 4, footerBits - 4); rc.EncodeReverseTree(probs.prob_align, 4, directVal & 15); }
                }
                for (int r = 3; r > 0; r--) current_reps[r] = current_reps[r - 1];
                current_reps[0] = dist; current_state = current_state < 7 ? 7 : 10; global_pos += token.len;
            }
        }
        chunk_start += chunk_len;
    }
    rc.Flush();
}

// ============================================================================
// CLI Entry Point
// ============================================================================
void print_help(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [OPTION]... [FILE]...\n"
              << "LZMA Compressor using Open Addressing Hashing (24-39 bits).\n\n"
              << "  --lzma=lc=x,lp=y,pb=z  override standard LZMA model settings\n"
              << "  -c, --stdout           write on standard output, keep files unchanged\n"
              << "  -k, --keep             keep (don't delete) input files\n"
              << "  -h, --help             display this help and exit\n\n";
}

bool parse_lzma_args(const std::string& arg, LZMAParams& params) {
    std::string props = arg.substr(7);
    size_t start = 0;
    while (start < props.length()) {
        size_t end = props.find(',', start);
        if (end == std::string::npos) end = props.length();
        std::string kv = props.substr(start, end - start);
        size_t eq = kv.find('=');
        if (eq != std::string::npos) {
            std::string key = kv.substr(0, eq);
            try {
                int val = std::stoi(kv.substr(eq + 1));
                if (key == "lc") params.lc = val;
                else if (key == "lp") params.lp = val;
                else if (key == "pb") params.pb = val;
                else return false;
            } catch (...) { return false; }
        } else return false;
        start = end + 1;
    }
    return (params.lc >= 0 && params.lc <= 8 && params.lp >= 0 && params.lp <= 4 && params.pb >= 0 && params.pb <= 4);
}

bool read_entire_stream(std::istream& in, std::vector<uint8_t>& buffer) {
    char temp[8192];
    while (in.read(temp, sizeof(temp))) buffer.insert(buffer.end(), temp, temp + in.gcount());
    buffer.insert(buffer.end(), temp, temp + in.gcount());
    return true;
}

int main(int argc, char** argv) {
    InitProbPrices();
    bool to_stdout = false, keep_files = false;
    LZMAParams params;
    std::vector<std::string> files;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-c" || arg == "--stdout") to_stdout = true;
        else if (arg == "-k" || arg == "--keep") keep_files = true;
        else if (arg == "-h" || arg == "--help") { print_help(argv[0]); return 0; }
        else if (arg.rfind("--lzma=", 0) == 0) {
            if (!parse_lzma_args(arg, params)) { std::cerr << argv[0] << ": invalid parameter string\n"; return 1; }
        }
        else if (arg[0] == '-' && arg != "-") { std::cerr << argv[0] << ": unrecognized option\n"; return 1; }
        else files.push_back(arg);
    }

#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY); _setmode(_fileno(stdout), _O_BINARY);
#endif

    if (files.empty()) { files.push_back("-"); to_stdout = true; }

    for (const auto& file : files) {
        std::vector<uint8_t> input_data;
        bool is_stdin = (file == "-");

        if (is_stdin) read_entire_stream(std::cin, input_data);
        else {
            std::ifstream ifs(file, std::ios::binary);
            if (!ifs) { std::cerr << argv[0] << ": " << file << ": No such file\n"; continue; }
            read_entire_stream(ifs, input_data);
        }

        if (to_stdout || is_stdin) CompressLZMA(input_data, std::cout, params);
        else {
            std::string out_file = file + ".lzma";
            std::ofstream ofs(out_file, std::ios::binary);
            if (!ofs) { std::cerr << argv[0] << ": " << out_file << ": Permission denied\n"; continue; }
            CompressLZMA(input_data, ofs, params);
            ofs.close();
            if (!keep_files && ofs.good()) std::remove(file.c_str());
        }
    }
    return 0;
}
