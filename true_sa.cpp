#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <string_view>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <random>
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <lzma.h>

/**
 * ==============================================================================
 * ALGORITHM OVERVIEW: Simulated Annealing with Threaded BFS Greedy Refinement
 * ==============================================================================
 * 
 * 1. Frequency Analysis & Victim Selection
 *    - Scans all input files to tally occurrences of every byte value (0-255).
 *    - Identifies the "Victim Index" (lowest frequency byte), making it the 
 *      focal point for continuous remapping swaps.
 * 
 * 2. State Initialization & Persistent Seeding
 *    - Establishes a true Baseline using the Identity Mapping.
 *    - Attempts to load a persistent seed matching the LZMA context parameters.
 *    - Randomly reshuffles the identity mapping (Fisher-Yates / std::shuffle) 
 *      for the Initial State if no seed is available.
 * 
 * 3. Simulated Annealing Phase (Main Thread)
 *    - Exponential Cooling Schedule lowers Temperature (T) over the timeout.
 *    - Evaluates random swaps at the victim index using the Metropolis 
 *      Acceptance Criterion. Normalized energy uses a 100,000x multiplier.
 * 
 * 4. Metropolis Acceptance Criterion
 *    - Uses a normalized Energy scale E = (size / baseline_size) * 100000.
 *    - If a swap DECREASES or EQUALS the energy (ΔE <= 0), it is 
 *      unconditionally accepted.
 *    - If a swap INCREASES the energy (ΔE > 0), it is probabilistically
 *      accepted using the Boltzmann distribution: P(accept) = exp(-ΔE / T).
 * 
 * 5. Concurrent Greedy Refinement (Worker Thread)
 *    - Isolates greedy exploration into an asynchronous worker thread running 
 *      a Breadth-First Search (BFS) up to depth 7. Strictly follows paths where
 *      compressed size improves or stays equal (<=).
 *    - Tracks unique paths using a 64-bit integer state ID.
 *    - Uses a 24-bit indexed bit array for depths <= 3, and a hash set for depths 4-7.
 *    - Dynamically resets its search space whenever the main Annealing thread 
 *      discovers a state better than the worker thread's current best.
 *    - Restarts the BFS search with a new better root node even before
 *      reaching depth 7 if an improvement is found within the greedy depth.
 * 
 * 6. Final Extreme Evaluation & Seed Retention
 *    - Benchmarks the final SA best state and final Greedy best state against 
 *      the identity mapping using the LZMA2 "6e" preset, an 8MB dictionary, 
 *      and context parameters (lc, lp, pb) for maximum compression.
 *    - Commits the globally optimal mapping back to the persistent seed file
 *      if it improves upon the initial loaded state.
 * ==============================================================================
 */

// Thread-safe stdout/stderr output synchronization
static std::recursive_mutex stderr_mtx;

struct FileData {
    std::string filename;
    std::vector<uint8_t> in_buf;
};

// Thread-local evaluation workspace to prevent buffer reallocation and race conditions
struct LZMAWorkspace {
    std::vector<uint8_t> remapped_buf;
    std::vector<uint8_t> out_buf;

    explicit LZMAWorkspace(size_t max_file_size) {
        remapped_buf.resize(std::max<size_t>(max_file_size, 1));
        size_t out_capacity = lzma_stream_buffer_bound(max_file_size);
        out_buf.resize(std::max<size_t>(out_capacity, 1));
    }
};

using RemapTable = std::array<uint8_t, 256>;

// ------------------------------------------------------------------------------------------
// Rigid Seed File Parser & Permutation Validator
// ------------------------------------------------------------------------------------------
int load_seed_from_file(const char *filename, int lc, int lp, int pb, unsigned char *out_remap) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        return 0; // Fail silently, allows natural fallback to random shuffle
    }

    char target_decl[64];
    snprintf(target_decl, sizeof(target_decl), "remap_seed_%d%d%d", lc, lp, pb);

    char line[512];
    int found_decl = 0;

    // Scan for dynamic target array declaration
    while (fgets(line, sizeof(line), file)) {
        if (strstr(line, target_decl) != NULL) {
            found_decl = 1;
            break;
        }
    }

    if (!found_decl) {
        fclose(file);
        return 0; 
    }

    int in_array = 0;
    if (strchr(line, '{')) in_array = 1;
    else {
        while (fgets(line, sizeof(line), file)) {
            if (strchr(line, '{')) {
                in_array = 1;
                break;
            }
        }
    }

    if (!in_array) {
        fclose(file);
        return 0; 
    }

    // Extract exact byte values
    int count = 0, c;
    while (count < 256 && (c = fgetc(file)) != EOF) {
        if (c == '0') {
            int next = fgetc(file);
            if (next == 'x' || next == 'X') {
                unsigned int val;
                if (fscanf(file, "%2x", &val) == 1) out_remap[count++] = (unsigned char)val;
            } else ungetc(next, file); 
        } else if (c == '\'') {
            int char_val = fgetc(file);
            if (char_val == '\\') { 
                int escaped = fgetc(file);
                if (escaped == '\'') char_val = '\'';
                else if (escaped == '\\') char_val = '\\';
            }
            out_remap[count++] = (unsigned char)char_val;
            while ((c = fgetc(file)) != EOF && c != '\'') { }
        } else if (c == '}') break; 
    }
    fclose(file);

    if (count != 256) {
        fprintf(stderr, "Error: Parsed %d bytes, expected 256.\n", count);
        return 0;
    }

    /**
     * Algorithm: Validating Permutation
     * We iterate through the parsed 256-byte array and use a frequency map (boolean array)
     * to guarantee no duplicates exist. Since the input array size is constrained to 256 
     * and bounds are enforced by the uint8_t types, checking for uniqueness mathematically 
     * proves it is a valid 0-255 mapping.
     */
    int seen[256] = {0};
    for (int i = 0; i < 256; i++) {
        if (seen[out_remap[i]]) {
            fprintf(stderr, "Error: Seed file array contains duplicate value 0x%02X.\n", out_remap[i]);
            return 0; 
        }
        seen[out_remap[i]] = 1;
    }

    return 1;
}

/**
 * ==============================================================================
 * Algorithm: Safe File Stream Patching (Seed Upgrade)
 * ==============================================================================
 * Streams the original file to a temporary .new file character by character. 
 * When the target declaration signature is matched, it safely skips the old 
 * array initialization payload and injects the new optimized permutation, 
 * preserving all surrounding code, comments, and file structure. Atomically
 * overwrites the target file upon completion. Handles file creation if missing.
 */
void save_seed_to_file(const char *filepath, int lc, int lp, int pb, const uint8_t *best_remap) {
    char new_path[1024];
    snprintf(new_path, sizeof(new_path), "%s.new", filepath);
    
    FILE *in = fopen(filepath, "r");
    FILE *out = fopen(new_path, "w");
    
    if (!out) {
        std::lock_guard<std::recursive_mutex> lock(stderr_mtx);
        fprintf(stderr, "/* Error: Could not open temp file '%s' for writing. */\n", new_path);
        if (in) fclose(in);
        return;
    }

    char target_decl[64];
    snprintf(target_decl, sizeof(target_decl), "remap_seed_%d%d%d", lc, lp, pb);
    
    if (!in) {
        // File does not exist, initialize a new file format
        fprintf(out, "unsigned char %s[256] = {\n    ", target_decl);
        for (int i = 0; i < 256; i++) {
            fprintf(out, "0x%02x%s", best_remap[i], (i == 255) ? "" : ((i + 1) % 16 == 0 ? ",\n    " : ", "));
        }
        fprintf(out, "\n};\n");
    } else {
        int target_len = strlen(target_decl);
        int match_idx = 0;
        int in_target = 0;
        int c;
        int found_existing = 0;

        while ((c = fgetc(in)) != EOF) {
            if (!in_target) {
                fputc(c, out);
                if (c == target_decl[match_idx]) {
                    match_idx++;
                    if (match_idx == target_len) {
                        in_target = 1; // Discovered the specific C array declaration block!
                        found_existing = 1;
                    }
                } else {
                    match_idx = (c == target_decl[0]) ? 1 : 0;
                }
            } else {
                fputc(c, out);
                // Search dynamically for the opening block '{' regardless of newlines/whitespace
                if (c == '{') {
                    fprintf(out, "\n    ");
                    for (int i = 0; i < 256; i++) {
                        fprintf(out, "0x%02x%s", best_remap[i], (i == 255) ? "" : ((i + 1) % 16 == 0 ? ",\n    " : ", "));
                    }
                    fprintf(out, "\n}");
                    
                    // Fast-forward input reader past the obsolete payload until the matching '}'
                    int skip_c;
                    while ((skip_c = fgetc(in)) != EOF) {
                        if (skip_c == '}') break; 
                    }
                    in_target = 0;
                    match_idx = 0;
                }
            }
        }
        
        if (!found_existing) {
            // Append securely to the end if the file existed but didn't have this target
            fprintf(out, "\nunsigned char %s[256] = {\n    ", target_decl);
            for (int i = 0; i < 256; i++) {
                fprintf(out, "0x%02x%s", best_remap[i], (i == 255) ? "" : ((i + 1) % 16 == 0 ? ",\n    " : ", "));
            }
            fprintf(out, "\n};\n");
        }
        fclose(in);
    }
    
    fclose(out);
    
    if (rename(new_path, filepath) != 0) {
        std::lock_guard<std::recursive_mutex> lock(stderr_mtx);
        fprintf(stderr, "\n/* Error: Failed to atomicly rename '%s' to '%s'. */\n\n", new_path, filepath);
    } else {
        std::lock_guard<std::recursive_mutex> lock(stderr_mtx);
        fprintf(stderr, "\n/* Successfully saved optimized seed configuration to: %s */\n\n", filepath);
    }
}


void print_help(const char* prog_name) {
    std::lock_guard<std::recursive_mutex> lock(stderr_mtx);
    std::cout << "Usage: " << prog_name << " [OPTIONS] <input_file_1> [input_file_2 ...]\n\n"
              << "Options:\n"
              << "  --help               Show this help message and exit.\n"
              << "  --timeout=SEC        Set timeout in seconds for annealing (default: 600).\n"
              << "  --temperature=FLOAT  Set initial temperature for annealing (default: 50.0).\n"
              << "  --bfs-depth=INT      Set initial max BFS depth for Greedy Refinement (default: 1).\n"
              << "  --dict=KB            Set LZMA dictionary size in kilobytes (default: 4).\n"
              << "  --lc=BITS            Set LZMA literal context bits (1-4, default: 3).\n"
              << "  --lp=BITS            Set LZMA literal position bits (0-4, default: 0).\n"
              << "  --pb=BITS            Set LZMA position bits (0-4, default: 2).\n"
              << "  --seedfile=FILE      Set persistent file to load/save remap seed configuration.\n\n";
}

/**
 * Algorithm: LZMA Compression Wrapper
 * Utilizes lzma_lzma_preset to apply base settings, then conditionally overrides
 * specific parameters (dict, lc, lp, pb). This allows for fast low-level evaluation 
 * during search and extreme settings during final reporting.
 */
size_t compress_buffer(const uint8_t* in_buf, size_t in_len, uint8_t* out_buf, size_t out_capacity, 
                        uint32_t preset, int dict_param_kb, int lc_param, int lp_param, int pb_param) {
    lzma_options_lzma opt;
    if (lzma_lzma_preset(&opt, preset)) return out_capacity + 1;

    opt.dict_size = ((uint32_t)dict_param_kb * 1024 < LZMA_DICT_SIZE_MIN) ? LZMA_DICT_SIZE_MIN : (uint32_t)dict_param_kb * 1024; 
    opt.lc = lc_param; 
    opt.lp = lp_param; 
    opt.pb = pb_param;

    lzma_filter filters[2] = { { LZMA_FILTER_LZMA2, &opt }, { LZMA_VLI_UNKNOWN, nullptr } };
    size_t out_pos = 0;
    lzma_ret ret = lzma_stream_buffer_encode(filters, LZMA_CHECK_CRC32, nullptr, in_buf, in_len, out_buf, &out_pos, out_capacity);
    return (ret == LZMA_OK) ? out_pos : out_capacity + 1;
}

size_t evaluate_remap(const RemapTable& remap_table, const std::vector<FileData>& files, 
                      LZMAWorkspace& ws, uint32_t preset, int dict, int lc, int lp, int pb) {
    size_t total = 0;
    for (const auto& file : files) {
        for (size_t j = 0; j < file.in_buf.size(); j++) {
            ws.remapped_buf[j] = remap_table[file.in_buf[j]];
        }
        total += compress_buffer(ws.remapped_buf.data(), file.in_buf.size(), ws.out_buf.data(), ws.out_buf.size(), preset, dict, lc, lp, pb);
    }
    return total;
}

void print_remap_table_as_source(const char* var_name, const RemapTable& remap) {
    std::lock_guard<std::recursive_mutex> lock(stderr_mtx);
    fprintf(stderr, "unsigned char %s[256] = {\n    ", var_name);
    for (int i = 0; i < 256; i++) {
        if (remap[i] >= 32 && remap[i] <= 126) {
            if (remap[i] == '\'') fprintf(stderr, "'\\''"); 
            else if (remap[i] == '\\') fprintf(stderr, "'\\\\'"); 
            else fprintf(stderr, "'%c' ", remap[i]); 
        } else {
            fprintf(stderr, "0x%02x", remap[i]); 
        }
        if (i < 255) fprintf(stderr, ", ");
        if ((i + 1) % 16 == 0 && i < 255) fprintf(stderr, "\n    ");
    }
    fprintf(stderr, "\n};\n");
    fflush(stderr); 
}

/**
 * Algorithm: Comprehensive Progress Reporter
 * Triggers strictly when a swap yields a smaller compressed size compared to the 
 * immediate previous step. It provides comparative analytics across all states.
 */
void report_improvement(const RemapTable& remap, unsigned long long iter, double elapsed, double timeout,
                        int dict, int lc, int lp, int pb,
                        size_t new_size, size_t prev_size, size_t initial_size, size_t baseline_size, double T) {
    std::lock_guard<std::recursive_mutex> lock(stderr_mtx);

    double remaining = std::max(0.0, timeout - elapsed);

    fprintf(stderr, "========================================================================\n");
    fprintf(stderr, "ANNEALING IMPROVEMENT FOUND AT ITERATION %llu\n", iter);
    fprintf(stderr, "Time: %.1fs elapsed, %.1fs remaining\n", elapsed, remaining);
    fprintf(stderr, "LZMA Settings: dict=%dKB, lc=%d, lp=%d, pb=%d\n", dict, lc, lp, pb);

    double pct_id = baseline_size ? ((double)((long long)new_size - (long long)baseline_size) / baseline_size) * 100.0 : 0.0;
    double pct_init = initial_size ? ((double)((long long)new_size - (long long)initial_size) / initial_size) * 100.0 : 0.0;
    double pct_prev = prev_size ? ((double)((long long)new_size - (long long)prev_size) / prev_size) * 100.0 : 0.0;

    fprintf(stderr, "Total Compressed Size: %zu bytes\n", new_size);
    fprintf(stderr, "Reduction vs Identity Mapping : %+10.3f%%\n", pct_id);
    fprintf(stderr, "Reduction vs Initial State    : %+10.3f%%\n", pct_init);
    fprintf(stderr, "Reduction vs Previous Step    : %+10.3f%%\n", pct_prev);

    /**
     * Algorithm: Regression Acceptance Probability Calculation
     * Calculates the probability P(accept) = exp(-ΔE / T) for specific size increases.
     * Since ΔE = (percent_increase / 100) * 100000 = percent_increase * 1000,
     * +0.01% -> ΔE = 10.0
     * +0.10% -> ΔE = 100.0
     * +1.00% -> ΔE = 1000.0
     */
    double prob_0_01 = (T > 0.0) ? exp(-10.0 / T) : 0.0;
    double prob_0_10 = (T > 0.0) ? exp(-100.0 / T) : 0.0;
    double prob_1_00 = (T > 0.0) ? exp(-1000.0 / T) : 0.0;

    fprintf(stderr, "Temperature: %.4f\n", T);
    fprintf(stderr, "P(Accept) for Regressions     : +0.01%%: %.8f | +0.1%%: %.8f | +1%%: %.8f\n", 
            prob_0_01, prob_0_10, prob_1_00);

    fprintf(stderr, "\nCurrent Remap Table Dump:\n");
    print_remap_table_as_source("current_remap", remap);
    fprintf(stderr, "========================================================================\n\n");
}

/**
 * Algorithm: Asynchronous Greedy Worker Thread (BFS)
 * Manages concurrent local greedy exploration. Runs Breadth-First Search. 
 * Evaluates all victim-index swaps, queuing states that maintain 
 * or decrease total size. Instantly aborts and resets state when a new Annealing
 * best is found.
 */
class GreedyWorker {
public:
    struct Task {
        RemapTable remap;
        size_t size;
    };

    struct BFSNode {
        uint64_t state;
        size_t size;
        int depth;
    };

    GreedyWorker(const std::vector<FileData>& files, size_t max_file_size, int victim_idx,
                 int dict, int lc, int lp, int pb, int start_bfs_depth)
        : files_(files), max_file_size_(max_file_size), victim_index_(victim_idx),
          dict_(dict), lc_(lc), lp_(lp), pb_(pb), start_bfs_depth_(start_bfs_depth),
          stop_flag_(false), new_task_flag_(false), has_best_(false),
          best_size_(std::numeric_limits<size_t>::max()) 
    {
        worker_thread_ = std::thread(&GreedyWorker::run, this);
    }

    ~GreedyWorker() {
        stop();
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            stop_flag_ = true;
        }
        cv_.notify_one();
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }

    size_t get_best_size() {
        std::lock_guard<std::mutex> lock(best_mtx_);
        return best_size_;
    }

    bool get_best(RemapTable& out_remap, size_t& out_size) {
        std::lock_guard<std::mutex> lock(best_mtx_);
        if (!has_best_) return false;
        out_remap = best_remap_;
        out_size = best_size_;
        return true;
    }

    void notify_new_annealing_best(const RemapTable& remap, size_t size) {
        std::lock_guard<std::mutex> lock(mtx_);
        {
            std::lock_guard<std::mutex> b_lock(best_mtx_);
            if (size < best_size_) {
                best_size_ = size;
                best_remap_ = remap;
                has_best_ = true;
            }
        }
        pending_task_ = Task{remap, size};
        new_task_flag_ = true;
        cv_.notify_one();
    }

private:
    static RemapTable reconstruct_remap(uint64_t state, const RemapTable& root_remap, int victim_index) {
        RemapTable remap = root_remap;
        while (true) {
            uint8_t idx = static_cast<uint8_t>(state & 0xFF);
            if (idx == victim_index) break;
            std::swap(remap[victim_index], remap[idx]);
            state >>= 8;
        }
        return remap;
    }

    void run() {
        LZMAWorkspace ws(max_file_size_);

        while (true) {
            Task task;
            {
                std::unique_lock<std::mutex> lock(mtx_);
                cv_.wait(lock, [this] { return new_task_flag_ || stop_flag_; });
                if (stop_flag_ && !new_task_flag_) break;
                task = pending_task_;
                new_task_flag_ = false;
            }

            {
                std::lock_guard<std::recursive_mutex> s_lock(stderr_mtx);
                fprintf(stderr, "========================================================================\n");
                fprintf(stderr, "/* [Greedy Worker] Resetting thread to refine new Annealing Best (Size: %zu bytes) */\n", task.size);
                fprintf(stderr, "========================================================================\n\n");
            }

            run_bfs(task, ws);
        }
    }

    void run_bfs(const Task& start_task, LZMAWorkspace& ws) {
        int cumulative_depth = 0;
        int current_max_depth = start_bfs_depth_;
        Task current_root_task = start_task;
        std::vector<uint64_t> visited_bits((1ULL << 24) / 64, 0);

        while (true) {
again:;
            uint64_t root_state = victim_index_;

            std::queue<BFSNode> q;
            std::unordered_set<uint64_t> visited_hash;
            std::fill(visited_bits.begin(), visited_bits.end(), 0);

            q.push({root_state, current_root_task.size, 0});
            uint32_t root_idx = root_state & 0xFFFFFF;
            visited_bits[root_idx >> 6] |= (1ULL << (root_idx & 63));

            uint64_t best_found_state = root_state;
            int best_found_depth = 0;
            size_t best_found_size = current_root_task.size;

            RemapTable base_remap = current_root_task.remap;

            while (!q.empty()) {
                if (new_task_flag_ || stop_flag_) return;

                BFSNode curr = q.front();
                if (curr.depth >= current_max_depth) {
                    // Consider restarting the search with a new root node
                    if (best_found_size < current_root_task.size) {
                        std::lock_guard<std::recursive_mutex> s_lock(stderr_mtx);
                        fprintf(stderr, "========================================================================\n");
                        fprintf(stderr, "/* [Greedy Worker] RESTARTING BFS SEARCH WITH A BETTER ROOT NODE */\n");
                        current_root_task.remap = reconstruct_remap(best_found_state, base_remap, victim_index_);
                        current_root_task.size = best_found_size;
                        current_max_depth = start_bfs_depth_;
                        cumulative_depth += best_found_depth;
                        goto again;
                    } else {
                        std::lock_guard<std::recursive_mutex> s_lock(stderr_mtx);
                        fprintf(stderr, "========================================================================\n");
                        current_max_depth++;
                        if (current_max_depth > 7) {
                            fprintf(stderr, "/* [Greedy Worker] CAN'T INCREASE MAX BFS DEPTH TO %d */\n", current_max_depth);
                            return;
                        } else {
                            fprintf(stderr, "/* [Greedy Worker] INCREASING BFS MAX DEPTH TO %d AND RESUMING */\n", current_max_depth);
                            continue;
                        }
                    }
                }
                q.pop();
                RemapTable curr_remap = reconstruct_remap(curr.state, base_remap, victim_index_);

                for (int i = 0; i < 256; i++) {
                    if (i == victim_index_) continue;
                    if (new_task_flag_ || stop_flag_) return;

                    uint64_t mask = (curr.depth == 0) ? 0 : ((1ULL << (curr.depth * 8)) - 1);
                    uint64_t next_state = (curr.state & mask) 
                                        | ((uint64_t)i << (curr.depth * 8)) 
                                        | ((uint64_t)victim_index_ << ((curr.depth + 1) * 8));

                    int next_depth = curr.depth + 1;
                    bool is_visited = false;

                    if (next_depth <= 3) {
                        uint32_t idx = next_state & 0xFFFFFF;
                        uint32_t word_idx = idx >> 6;
                        uint32_t bit_idx = idx & 63;
                        is_visited = (visited_bits[word_idx] & (1ULL << bit_idx)) != 0;
                        if (!is_visited) visited_bits[word_idx] |= (1ULL << bit_idx);
                    } else if (next_depth <= 7) {
                        is_visited = !visited_hash.insert(next_state).second;
                    } else {
                        continue;
                    }

                    if (is_visited) continue;

                    RemapTable next_remap = curr_remap;
                    std::swap(next_remap[victim_index_], next_remap[i]);

                    size_t next_size = evaluate_remap(next_remap, files_, ws, 0, dict_, lc_, lp_, pb_);

                    if (next_size <= curr.size) {
                        bool is_new_best = false;
                        {
                            std::lock_guard<std::mutex> lock(best_mtx_);
                            if (next_size < best_size_) {
                                best_size_ = next_size;
                                best_remap_ = next_remap;
                                has_best_ = true;
                                is_new_best = true;
                            }
                        }

                        if (is_new_best) {
                            std::lock_guard<std::recursive_mutex> s_lock(stderr_mtx);
                            fprintf(stderr, "========================================================================\n");
                            fprintf(stderr, "/* [Greedy Worker] NEW GREEDY BEST FOUND AT BFS DEPTH %d */\n", next_depth);
                            fprintf(stderr, "Started Refinement From Baseline Size : %zu bytes\n", start_task.size);
                            fprintf(stderr, "New Greedy Best Size                  : %zu bytes\n", next_size);
                            fprintf(stderr, "Absolute Byte Saved vs Start Baseline : %lld bytes\n", (long long)start_task.size - (long long)next_size);

                            fprintf(stderr, "\nInitial State Worker Started From:\n");
                            print_remap_table_as_source("initial_state_remap", start_task.remap);
                            if (start_task.remap != current_root_task.remap) {
                                fprintf(stderr, "\nCurrent BFS Checkpoint:\n");
                                print_remap_table_as_source("bfs_checkpoint_remap", current_root_task.remap);
                            }
                            fprintf(stderr, "\nCurrent Greedy Best Remap Table Dump (%d steps from start):\n", cumulative_depth + next_depth);
                            print_remap_table_as_source("greedy_best_remap", next_remap);
                            fprintf(stderr, "========================================================================\n\n");
                        }

                        if (next_size < best_found_size) {
                            best_found_size = next_size;
                            best_found_state = next_state;
                            best_found_depth = next_depth;
                        }

                        q.push({next_state, next_size, next_depth});
                    }
                }
            }

            if (new_task_flag_ || stop_flag_) return;
        }
    }

    const std::vector<FileData>& files_;
    size_t max_file_size_;
    int victim_index_;
    int dict_, lc_, lp_, pb_;
    int start_bfs_depth_;

    std::thread worker_thread_;
    std::mutex mtx_;
    std::condition_variable cv_;
    bool stop_flag_;
    bool new_task_flag_;
    Task pending_task_;

    std::mutex best_mtx_;
    bool has_best_;
    size_t best_size_;
    RemapTable best_remap_;
};

int main(int argc, char** argv) {
    int timeout = 600;
    double T_start = 50.0;
    int bfs_depth = 1;
    int dict_param_kb = 4, lc_param = 3, lp_param = 0, pb_param = 2;
    std::string seedfile = "";
    std::vector<std::string> filenames;

    for (int i = 1; i < argc; i++) {
        std::string_view arg(argv[i]);
        if (arg == "--help" || arg == "-h") { print_help(argv[0]); return EXIT_SUCCESS; }
        else if (arg.starts_with("--timeout=")) timeout = std::stoi(argv[i] + 10);
        else if (arg.starts_with("--temperature=")) T_start = std::stod(argv[i] + 14);
        else if (arg.starts_with("--bfs-depth=")) bfs_depth = std::stoi(argv[i] + 12);
        else if (arg.starts_with("--dict=")) dict_param_kb = std::stoi(argv[i] + 7);
        else if (arg.starts_with("--lc=")) lc_param = std::stoi(argv[i] + 5);
        else if (arg.starts_with("--lp=")) lp_param = std::stoi(argv[i] + 5);
        else if (arg.starts_with("--pb=")) pb_param = std::stoi(argv[i] + 5);
        else if (arg.starts_with("--seedfile=")) seedfile = arg.substr(11);
        else filenames.emplace_back(argv[i]);
    }

    if (filenames.empty()) {
        std::lock_guard<std::recursive_mutex> lock(stderr_mtx);
        std::cerr << "Error: Missing input files.\n";
        return EXIT_FAILURE;
    }

    std::mt19937 rng(std::random_device{}());

    std::vector<FileData> files(filenames.size());
    size_t max_file_size = 0;

    // File reading and memory initialization
    for (size_t i = 0; i < filenames.size(); i++) {
        files[i].filename = filenames[i];
        FILE* f = fopen(filenames[i].c_str(), "rb");
        if (!f) return EXIT_FAILURE;
        fseek(f, 0, SEEK_END);
        files[i].in_buf.resize(ftell(f));
        fseek(f, 0, SEEK_SET);
        if (files[i].in_buf.size() > max_file_size) max_file_size = files[i].in_buf.size();
        if (!files[i].in_buf.empty() && fread(files[i].in_buf.data(), 1, files[i].in_buf.size(), f) != files[i].in_buf.size()) {
            fclose(f);
            return EXIT_FAILURE;
        }
        fclose(f);
    }

    LZMAWorkspace main_ws(max_file_size);

    /**
     * Algorithm: Frequency Analysis & Victim Index Identification
     * Iterate over the byte layout of all files to construct a global frequency 
     * histogram. The index mapping to the lowest frequency becomes the "victim", 
     * serving as the continuous focal point for swaps in the SA loop.
     */
    std::array<unsigned long long, 256> byte_freq{};
    for (const auto& file : files) {
        for (uint8_t byte_val : file.in_buf) {
            byte_freq[byte_val]++;
        }
    }

    int victim_index = 0;
    for (int i = 1; i < 256; i++) {
        if (byte_freq[i] < byte_freq[victim_index]) {
            victim_index = i;
        }
    }

    RemapTable identity_remap;
    for (int i = 0; i < 256; i++) identity_remap[i] = static_cast<uint8_t>(i);

    // Evaluate Baseline using fast preset 0
    size_t baseline_sum = evaluate_remap(identity_remap, files, main_ws, 0, dict_param_kb, lc_param, lp_param, pb_param);
    double safe_baseline = baseline_sum > 0 ? static_cast<double>(baseline_sum) : 1.0;

    RemapTable current_remap;
    bool seed_loaded = false;

    if (!seedfile.empty()) {
        if (load_seed_from_file(seedfile.c_str(), lc_param, lp_param, pb_param, current_remap.data())) {
            seed_loaded = true;
            std::lock_guard<std::recursive_mutex> lock(stderr_mtx);
            fprintf(stderr, "/* Seed configuration loaded successfully from %s for %d%d%d */\n", seedfile.c_str(), lc_param, lp_param, pb_param);
        }
    }

    if (!seed_loaded) {
        /**
         * Algorithm: Fisher-Yates Randomization (std::shuffle)
         * Generates an unbiased, uniformly random permutation of the identity mapping 
         * to serve as the initial chaotic state for the simulated annealing landscape.
         * Used whenever a seed cannot be resolved.
         */
        current_remap = identity_remap;
        std::shuffle(current_remap.begin(), current_remap.end(), rng);
    }

    size_t initial_sum = evaluate_remap(current_remap, files, main_ws, 0, dict_param_kb, lc_param, lp_param, pb_param);
    size_t current_sum = initial_sum;
    size_t sa_best_sum = initial_sum;
    RemapTable sa_best_remap = current_remap;

    {
        std::lock_guard<std::recursive_mutex> lock(stderr_mtx);
        fprintf(stderr, "/* Simulated Annealing Initialized */\n");
        fprintf(stderr, "Victim Index   : 0x%02X (Frequency: %llu bytes)\n", victim_index, byte_freq[victim_index]);
        fprintf(stderr, "Initial Temp   : %.2f\n", T_start);
        fprintf(stderr, "Baseline Size  : %zu bytes (Identity)\n", baseline_sum);
        fprintf(stderr, "Initial Size   : %zu bytes (%s)\n\n", initial_sum, seed_loaded ? "Seed Loaded" : "Shuffled");
    }

    // Instantiate and launch asynchronous Greedy Refinement Worker Thread
    GreedyWorker greedy_worker(files, max_file_size, victim_index, dict_param_kb, lc_param, lp_param, pb_param, bfs_depth);
    greedy_worker.notify_new_annealing_best(sa_best_remap, sa_best_sum);

    auto start_time = std::chrono::steady_clock::now();
    auto last_report_time = start_time;
    unsigned long long iterations = 0;
    double report_delay = static_cast<double>(timeout) / 100.0;

    /**
     * State Variables for Postponed Reporting
     * These capture all necessary snapshot metrics to delay a stderr print
     * if the throttling window hasn't yet elapsed.
     */
    bool has_pending_report = false;
    RemapTable pending_remap;
    unsigned long long pending_iter = 0;
    double pending_elapsed = 0.0;
    size_t pending_new_size = 0, pending_prev_size = 0;
    double pending_T = 0.0; 

    // SA Parameters
    double T_end = 0.1;
    double T = T_start;

    std::uniform_int_distribution<int> dist256(0, 255);
    std::uniform_real_distribution<double> dist_prob(0.0, 1.0);

    // =========================================================
    // TRUE TEXTBOOK SIMULATED ANNEALING LOOP
    // =========================================================
    while (true) {
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        double since_last_report = std::chrono::duration<double>(now - last_report_time).count();

        /**
         * Algorithm: Delayed Report Dispatcher
         * At the top of every iteration, evaluates if a postponed report exists 
         * and if the throttle window has finally cleared. If both are true, 
         * the most recent queued improvement is flushed to the terminal.
         */
        if (has_pending_report && since_last_report >= report_delay) {
            report_improvement(pending_remap, pending_iter, pending_elapsed, timeout, 
                               dict_param_kb, lc_param, lp_param, pb_param, 
                               pending_new_size, pending_prev_size, initial_sum, baseline_sum, pending_T);
            last_report_time = now;
            has_pending_report = false;
        }

        if (elapsed >= timeout) break;
        iterations++;

        /**
         * Algorithm: Exponential Cooling Schedule
         * T_current = T_start * (T_end / T_start)^(elapsed / timeout)
         * Guarantees T will smoothly drop from T_start down to T_end 
         * at the exact moment the timeout is reached.
         */
        double progress = elapsed / timeout;
        T = T_start * std::pow(T_end / T_start, progress);

        // Pick a random target distinct from the victim
        int swap_idx;
        do {
            swap_idx = dist256(rng);
        } while (swap_idx == victim_index);

        // Generate test state
        RemapTable test_remap = current_remap;
        std::swap(test_remap[victim_index], test_remap[swap_idx]);

        // Evaluate neighbor state using preset 0 for fast search
        size_t test_sum = evaluate_remap(test_remap, files, main_ws, 0, dict_param_kb, lc_param, lp_param, pb_param);

        /**
         * Algorithm: Normalized Energy Delta Calculation
         * Evaluates relative change in compressed size rather than absolute bytes.
         * Energy E = (compressed_size / baseline_size) * 100000
         */
        double delta = (((double)test_sum - (double)current_sum) / safe_baseline) * 100000.0;

        /**
         * Algorithm: Metropolis Acceptance Criterion
         * Standard Simulated Annealing state selection logic using normalized delta.
         */
        if (delta <= 0.0) {
            // ACCEPT: Decrease (or no change) in energy
            if (delta < 0.0) { 
                bool is_global_best = (test_sum < sa_best_sum);
                
                /**
                 * Algorithm: Overwriting Queue Logic
                 * Output immediately if it's a new global best, or if the throttle
                 * interval allows it. Otherwise, postpone the snapshot.
                 */
                if (is_global_best || since_last_report >= report_delay) {
                    report_improvement(test_remap, iterations, elapsed, timeout, 
                                       dict_param_kb, lc_param, lp_param, pb_param, 
                                       test_sum, current_sum, initial_sum, baseline_sum, T);
                    last_report_time = now;
                    has_pending_report = false;
                } else {
                    pending_remap = test_remap;
                    pending_iter = iterations;
                    pending_elapsed = elapsed;
                    pending_new_size = test_sum;
                    pending_prev_size = current_sum;
                    pending_T = T;
                    has_pending_report = true;
                }
            }

            current_sum = test_sum;
            current_remap = test_remap;

            // Track SA best and notify worker if new best is found
            if (current_sum < sa_best_sum) {
                sa_best_sum = current_sum;
                sa_best_remap = current_remap;
                
                if (sa_best_sum < greedy_worker.get_best_size()) {
                    greedy_worker.notify_new_annealing_best(sa_best_remap, sa_best_sum);
                }
            }
        } else {
            // PROBABILISTIC ACCEPT: Increase in energy (Allows escaping local minima)
            double prob = std::exp(-delta / T);
            if (dist_prob(rng) < prob) {
                current_sum = test_sum;
                current_remap = test_remap;
            }
        }
    }

    /**
     * Algorithm: End-of-Run Final Flush
     * Ensure any snapshot trapped in the postponed queue when timeout hits gets output cleanly.
     */
    if (has_pending_report) {
        report_improvement(pending_remap, pending_iter, pending_elapsed, timeout, 
                           dict_param_kb, lc_param, lp_param, pb_param, 
                           pending_new_size, pending_prev_size, initial_sum, baseline_sum, pending_T);
    }

    {
        std::lock_guard<std::recursive_mutex> lock(stderr_mtx);
        fprintf(stderr, "/* Simulated Annealing Terminated. Iterations: %llu */\n", iterations);
        fprintf(stderr, "/* Best SA Compressed Size (Fast Settings): %zu bytes */\n\n", sa_best_sum);
    }

    // Stop worker thread cleanly before running final evaluation
    greedy_worker.stop();

    // =========================================================
    // FINAL EVALUATION (Extreme Settings) & PERSISTENT SAVE
    // =========================================================
    /**
     * Algorithm: Absolute Performance Benchmarking
     * Benchmarks both the last Annealing best and the last Greedy best mappings 
     * through a high-cost LZMA compression sequence (Preset "6e", 8MB dict).
     */
    {
        std::lock_guard<std::recursive_mutex> lock(stderr_mtx);
        fprintf(stderr, "========================================================================\n");
        fprintf(stderr, "FINAL EVALUATION (Preset: 6e, Dictionary: 8MB)\n");
        fprintf(stderr, "========================================================================\n");
    }

    uint32_t final_preset = 6 | LZMA_PRESET_EXTREME;
    int final_dict_kb = 8192; // 8MB

    size_t final_baseline = evaluate_remap(identity_remap, files, main_ws, final_preset, final_dict_kb, lc_param, lp_param, pb_param);
    size_t final_sa_best = evaluate_remap(sa_best_remap, files, main_ws, final_preset, final_dict_kb, lc_param, lp_param, pb_param);
    
    double sa_pct = final_baseline ? ((double)((long long)final_sa_best - (long long)final_baseline) / final_baseline) * 100.0 : 0.0;

    {
        std::lock_guard<std::recursive_mutex> lock(stderr_mtx);
        fprintf(stderr, "Identity Remap Baseline Size : %zu bytes\n", final_baseline);
        fprintf(stderr, "Last Annealing Best Size    : %zu bytes (%+10.3f%% vs Identity)\n", final_sa_best, sa_pct);
    }

    RemapTable greedy_best_remap;
    size_t greedy_fast_best = 0;
    bool has_greedy_best = greedy_worker.get_best(greedy_best_remap, greedy_fast_best);
    
    RemapTable final_optimal_remap = sa_best_remap;
    
    if (has_greedy_best) {
        size_t final_greedy_best = evaluate_remap(greedy_best_remap, files, main_ws, final_preset, final_dict_kb, lc_param, lp_param, pb_param);
        double greedy_pct = final_baseline ? ((double)((long long)final_greedy_best - (long long)final_baseline) / final_baseline) * 100.0 : 0.0;
        
        std::lock_guard<std::recursive_mutex> lock(stderr_mtx);
        fprintf(stderr, "Last Greedy Best Size       : %zu bytes (%+10.3f%% vs Identity)\n\n", final_greedy_best, greedy_pct);
        
        if (final_sa_best < final_greedy_best) {
            fprintf(stderr, "\nFinal Optimal Remap Table (Annealing Best):\n");
            print_remap_table_as_source("optimal_remap", sa_best_remap);
        } else {
            final_optimal_remap = greedy_best_remap;
            fprintf(stderr, "Final Optimal Remap Table (Greedy Worker Best):\n");
            print_remap_table_as_source("optimal_remap", greedy_best_remap);
        }
    } else {
        std::lock_guard<std::recursive_mutex> lock(stderr_mtx);
        fprintf(stderr, "\nFinal Optimal Remap Table (Annealing Best):\n");
        print_remap_table_as_source("optimal_remap", sa_best_remap);
    }

    {
        std::lock_guard<std::recursive_mutex> lock(stderr_mtx);
        fprintf(stderr, "========================================================================\n");
    }

    // Determine if the best run yielded an improvement against the initial starting configuration
    bool improvement_found = (sa_best_sum < initial_sum);
    if (has_greedy_best && greedy_fast_best < initial_sum) {
        improvement_found = true;
    }

    // Persist the seed if an improvement was found over the loaded state
    if (!seedfile.empty() && improvement_found) {
        save_seed_to_file(seedfile.c_str(), lc_param, lp_param, pb_param, final_optimal_remap.data());
    } else if (!seedfile.empty() && !improvement_found) {
        std::lock_guard<std::recursive_mutex> lock(stderr_mtx);
        fprintf(stderr, "\n/* Optimization concluded without improving upon the loaded seed. No file update executed. */\n");
    }

    return EXIT_SUCCESS;
}
