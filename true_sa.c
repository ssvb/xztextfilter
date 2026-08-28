#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <lzma.h>

/**
 * ==============================================================================
 * ALGORITHM OVERVIEW: Simulated Annealing with Greedy Refinement
 * ==============================================================================
 * 
 * 1. Frequency Analysis & Victim Selection
 *    - Scans all input files to tally occurrences of every byte value (0-255).
 *    - Identifies the "Victim Index" (lowest frequency byte), making it the 
 *      focal point for continuous remapping swaps.
 * 
 * 2. State Initialization
 *    - Establishes a true Baseline using the Identity Mapping.
 *    - Randomly reshuffles the identity mapping (Fisher-Yates) for the Initial State.
 * 
 * 3. Simulated Annealing Phase
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
 * 5. Post-Annealing Greedy Refinement
 *    - Iterates 0-255, swapping the victim index with each candidate.
 *    - Configurable Search Depth (--greedy-depth): Can evaluate sequences
 *      of multiple swaps recursively to escape shallow local minima.
 *    - Adopts *only* strict size improvements.
 *    - Repeats full 256-element passes until a local optimum is confirmed
 *      (zero improvements in a single pass).
 * 
 * 6. Final Extreme Evaluation
 *    - Benchmarks the absolute best mapping found against the identity mapping
 *      using the LZMA2 "6e" preset, an 8MB dictionary, and the user-specified
 *      context parameters (lc, lp, pb) for maximum compression.
 * ==============================================================================
 */

typedef struct {
    const char *filename;
    uint8_t *in_buf;
    size_t file_size;
} FileData;

void print_help(const char *prog_name) {
    printf("Usage: %s [OPTIONS] <input_file_1> [input_file_2 ...]\n\n", prog_name);
    printf("Options:\n");
    printf("  --help               Show this help message and exit.\n");
    printf("  --timeout=SEC        Set timeout in seconds for annealing (default: 60).\n");
    printf("  --tstart=FLOAT       Set initial temperature for annealing (default: 20.0).\n");
    printf("  --greedy-depth=NUM   Set search depth for greedy refinement (default: 1).\n");
    printf("  --dict=KB            Set LZMA dictionary size in kilobytes (default: 4).\n");
    printf("  --lc=BITS            Set LZMA literal context bits (1-4, default: 3).\n");
    printf("  --lp=BITS            Set LZMA literal position bits (0-4, default: 0).\n");
    printf("  --pb=BITS            Set LZMA position bits (0-4, default: 2).\n");
    printf("\n");
}

/**
 * Algorithm: LZMA Compression Wrapper
 * Utilizes lzma_lzma_preset to apply base settings, then conditionally overrides
 * specific parameters (dict, lc, lp, pb). This allows for fast low-level evaluation 
 * during search and extreme settings during final reporting.
 */
size_t compress_buffer(const uint8_t *in_buf, size_t in_len, uint8_t *out_buf, size_t out_capacity, uint32_t preset, int dict_param_kb, int lc_param, int lp_param, int pb_param) {
    lzma_options_lzma opt;
    if (lzma_lzma_preset(&opt, preset)) return out_capacity + 1;
    
    // Explicit overrides for the active options
    opt.dict_size = ((uint32_t)dict_param_kb * 1024 < LZMA_DICT_SIZE_MIN) ? LZMA_DICT_SIZE_MIN : (uint32_t)dict_param_kb * 1024; 
    opt.lc = lc_param; 
    opt.lp = lp_param; 
    opt.pb = pb_param;                         
    
    lzma_filter filters[2] = { { .id = LZMA_FILTER_LZMA2, .options = &opt }, { .id = LZMA_VLI_UNKNOWN, .options = NULL } };
    size_t out_pos = 0;
    lzma_ret ret = lzma_stream_buffer_encode(filters, LZMA_CHECK_CRC32, NULL, in_buf, in_len, out_buf, &out_pos, out_capacity);
    return (ret == LZMA_OK) ? out_pos : out_capacity + 1;
}

size_t evaluate_remap(const uint8_t *remap_table, int file_count, FileData *files, 
                      uint8_t *remapped_buf, uint8_t *out_buf, size_t out_capacity, 
                      uint32_t preset, int dict, int lc, int lp, int pb) {
    size_t total = 0;
    for (int i = 0; i < file_count; i++) {
        for (size_t j = 0; j < files[i].file_size; j++) {
            remapped_buf[j] = remap_table[files[i].in_buf[j]];
        }
        total += compress_buffer(remapped_buf, files[i].file_size, out_buf, out_capacity, preset, dict, lc, lp, pb);
    }
    return total;
}

void print_remap_table_as_source(const char *var_name, const uint8_t *remap) {
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
void report_improvement(const uint8_t *remap, unsigned long long iter, double elapsed, double timeout,
                        int dict, int lc, int lp, int pb,
                        size_t new_size, size_t prev_size, size_t initial_size, size_t baseline_size, double T) {
    
    double remaining = timeout - elapsed;
    if (remaining < 0) remaining = 0.0;

    fprintf(stderr, "========================================================================\n");
    fprintf(stderr, "IMPROVEMENT FOUND AT ITERATION %llu\n", iter);
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
 * Algorithm: Unified Recursive Multi-Depth Greedy Search
 * Consolidates the optimization logic by evaluating both top-level directly
 * and sequencing multiple swaps recursively. At the root level (depth 1), 
 * it systematically sweeps all 256 indices, allowing it to capture multiple 
 * localized improvements without breaking iteration. At deeper depths, 
 * it returns immediately upon discovering any strict improvement to bubble 
 * the optimized state back up the call stack, naturally preventing loop 
 * duplication in main.
 */
int explore_swaps(uint8_t *test_remap, size_t *current_sum, int current_depth, int max_depth, 
                  int victim_index, int file_count, FileData *files, 
                  uint8_t *remapped_buf, uint8_t *out_buf, size_t out_capacity, 
                  int dict_param_kb, int lc_param, int lp_param, int pb_param,
                  unsigned long long pass_num) {
    
    if (current_depth > max_depth) return 0;
    
    int any_improvement = 0;

    for (int i = 0; i < 256; i++) {
        if (i == victim_index) continue;
        
        uint8_t next_remap[256];
        memcpy(next_remap, test_remap, 256);
        
        // Apply the deeper candidate swap
        uint8_t tmp = next_remap[victim_index];
        next_remap[victim_index] = next_remap[i];
        next_remap[i] = tmp;
        
        size_t test_sum = evaluate_remap(next_remap, file_count, files, remapped_buf, out_buf, out_capacity, 0, dict_param_kb, lc_param, lp_param, pb_param);
        
        // If this combination yields a strict improvement, capture it
        if (test_sum < *current_sum) {
            *current_sum = test_sum;
            memcpy(test_remap, next_remap, 256); // Propagate the improved state back up
            any_improvement = 1;
            
            if (current_depth == 1) {
                fprintf(stderr, "Greedy Refinement: Pass %llu, swap with 0x%02X yielded new best size: %zu bytes\n", 
                        pass_num, i, *current_sum);
            } else {
                return 1; // Bubble up immediately for deeper depths
            }
        }
        // Otherwise, continue exploring down to max_depth
        else if (current_depth < max_depth) {
            if (explore_swaps(next_remap, current_sum, current_depth + 1, max_depth, victim_index, 
                              file_count, files, remapped_buf, out_buf, out_capacity, 
                              dict_param_kb, lc_param, lp_param, pb_param, pass_num)) {
                
                memcpy(test_remap, next_remap, 256);
                any_improvement = 1;
                
                if (current_depth == 1) {
                    fprintf(stderr, "Greedy Refinement: Pass %llu, multi-depth swap via 0x%02X yielded new best size: %zu bytes\n", 
                            pass_num, i, *current_sum);
                } else {
                    return 1; // Bubble up immediately
                }
            }
        }
    }
    return any_improvement;
}

int main(int argc, char **argv) {
    int timeout = 60;
    double T_start = 20.0; // Default initial temperature
    int greedy_depth = 1;  // Default greedy multi-swap depth
    int dict_param_kb = 4, lc_param = 3, lp_param = 0, pb_param = 2;
    
    const char **filenames = malloc(argc * sizeof(char*));
    int file_count = 0;

    /**
     * Algorithm: Command Line Parsing
     * Extracts runtime configuration including the --tstart flag,
     * which dictates the initial entropy/chaos of the annealing process.
     */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) { print_help(argv[0]); free(filenames); return EXIT_SUCCESS; }
        else if (strncmp(argv[i], "--timeout=", 10) == 0) timeout = atoi(argv[i] + 10);
        else if (strncmp(argv[i], "--tstart=", 9) == 0) T_start = atof(argv[i] + 9);
        else if (strncmp(argv[i], "--greedy-depth=", 15) == 0) {
            greedy_depth = atoi(argv[i] + 15);
            if (greedy_depth < 1) greedy_depth = 1;
        }
        else if (strncmp(argv[i], "--dict=", 7) == 0) dict_param_kb = atoi(argv[i] + 7);
        else if (strncmp(argv[i], "--lc=", 5) == 0) lc_param = atoi(argv[i] + 5);
        else if (strncmp(argv[i], "--lp=", 5) == 0) lp_param = atoi(argv[i] + 5);
        else if (strncmp(argv[i], "--pb=", 5) == 0) pb_param = atoi(argv[i] + 5);
        else filenames[file_count++] = argv[i];
    }

    if (file_count == 0) {
        fprintf(stderr, "Error: Missing input files.\n");
        free(filenames);
        return EXIT_FAILURE;
    }

    srand((unsigned int)time(NULL));

    FileData *files = malloc(file_count * sizeof(FileData));
    size_t max_file_size = 0;
    
    // File reading and memory initialization
    for (int i = 0; i < file_count; i++) {
        files[i].filename = filenames[i];
        FILE *f = fopen(filenames[i], "rb");
        if (!f) return EXIT_FAILURE;
        fseek(f, 0, SEEK_END); files[i].file_size = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
        if (files[i].file_size > max_file_size) max_file_size = files[i].file_size;
        files[i].in_buf = malloc(files[i].file_size > 0 ? files[i].file_size : 1);
        if (files[i].file_size > 0 && fread(files[i].in_buf, 1, files[i].file_size, f) != files[i].file_size) return EXIT_FAILURE;
        fclose(f);
    }

    uint8_t *remapped_buf = malloc(max_file_size > 0 ? max_file_size : 1);
    size_t out_capacity = lzma_stream_buffer_bound(max_file_size);
    uint8_t *out_buf = malloc(out_capacity > 0 ? out_capacity : 1);

    /**
     * Algorithm: Frequency Analysis & Victim Index Identification
     * Iterate over the byte layout of all files to construct a global frequency 
     * histogram. The index mapping to the lowest frequency becomes the "victim", 
     * serving as the continuous focal point for swaps in the SA loop.
     */
    unsigned long long byte_freq[256] = {0};
    for (int i = 0; i < file_count; i++) {
        for (size_t j = 0; j < files[i].file_size; j++) {
            byte_freq[files[i].in_buf[j]]++;
        }
    }
    
    int victim_index = 0;
    for (int i = 1; i < 256; i++) {
        if (byte_freq[i] < byte_freq[victim_index]) {
            victim_index = i;
        }
    }

    uint8_t identity_remap[256];
    for (int i = 0; i < 256; i++) identity_remap[i] = i;
    
    // Evaluate Baseline using fast preset 0
    size_t baseline_sum = evaluate_remap(identity_remap, file_count, files, remapped_buf, out_buf, out_capacity, 0, dict_param_kb, lc_param, lp_param, pb_param);
    double safe_baseline = baseline_sum > 0 ? (double)baseline_sum : 1.0;

    /**
     * Algorithm: Fisher-Yates Randomization
     * Generates an unbiased, uniformly random permutation of the identity mapping 
     * to serve as the initial chaotic state for the simulated annealing landscape.
     */
    uint8_t current_remap[256];
    memcpy(current_remap, identity_remap, 256);
    for (int i = 255; i > 0; i--) {
        int j = rand() % (i + 1);
        uint8_t tmp = current_remap[i]; 
        current_remap[i] = current_remap[j]; 
        current_remap[j] = tmp;
    }
    
    size_t initial_sum = evaluate_remap(current_remap, file_count, files, remapped_buf, out_buf, out_capacity, 0, dict_param_kb, lc_param, lp_param, pb_param);
    size_t current_sum = initial_sum;
    size_t best_sum = initial_sum;
    
    uint8_t best_remap[256];
    memcpy(best_remap, current_remap, 256);

    fprintf(stderr, "/* Simulated Annealing Initialized */\n");
    fprintf(stderr, "Victim Index   : 0x%02X (Frequency: %llu bytes)\n", victim_index, byte_freq[victim_index]);
    fprintf(stderr, "Initial Temp   : %.2f\n", T_start);
    fprintf(stderr, "Baseline Size  : %zu bytes (Identity)\n", baseline_sum);
    fprintf(stderr, "Initial Size   : %zu bytes (Shuffled)\n\n", initial_sum);

    time_t start_time = time(NULL);
    time_t last_report_time = 0; 
    unsigned long long iterations = 0;
    
    /**
     * State Variables for Postponed Reporting
     * These capture all necessary snapshot metrics to delay a stderr print
     * if the throttling window hasn't yet elapsed.
     */
    int has_pending_report = 0;
    uint8_t pending_remap[256];
    unsigned long long pending_iter = 0;
    double pending_elapsed = 0.0;
    size_t pending_new_size = 0;
    size_t pending_prev_size = 0;
    double pending_T = 0.0; // Captures the exact Temperature at the time of improvement

    // SA Parameters
    double T_end = 0.1;
    double T = T_start;

    // =========================================================
    // TRUE TEXTBOOK SIMULATED ANNEALING LOOP
    // =========================================================
    while (1) {
        time_t now = time(NULL);
        double elapsed = difftime(now, start_time);
        
        /**
         * Algorithm: Delayed Report Dispatcher
         * At the top of every iteration, evaluates if a postponed report exists 
         * and if the 5-second blackout window has finally cleared. If both are 
         * true, the most recent queued improvement is flushed to the terminal.
         */
        if (has_pending_report && difftime(now, last_report_time) >= 5.0) {
            report_improvement(pending_remap, pending_iter, pending_elapsed, timeout, 
                               dict_param_kb, lc_param, lp_param, pb_param, 
                               pending_new_size, pending_prev_size, initial_sum, baseline_sum, pending_T);
            last_report_time = now;
            has_pending_report = 0;
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
        T = T_start * pow(T_end / T_start, progress);

        // Pick a random target distinct from the victim
        int swap_idx;
        do {
            swap_idx = rand() % 256;
        } while (swap_idx == victim_index);

        // Generate test state
        uint8_t test_remap[256];
        memcpy(test_remap, current_remap, 256);
        
        // Single Swap at Victim Index
        uint8_t tmp = test_remap[victim_index];
        test_remap[victim_index] = test_remap[swap_idx];
        test_remap[swap_idx] = tmp;

        // Evaluate the neighbor state using preset 0 for fast search
        size_t test_sum = evaluate_remap(test_remap, file_count, files, remapped_buf, out_buf, out_capacity, 0, dict_param_kb, lc_param, lp_param, pb_param);

        /**
         * Algorithm: Normalized Energy Delta Calculation
         * Evaluates the relative change in compressed size rather than absolute bytes.
         * Energy E = (compressed_size / baseline_size) * 100000
         * By normalizing against the baseline size, the energy landscape becomes 
         * invariant to the raw size of the input files.
         */
        double delta = (((double)test_sum - (double)current_sum) / safe_baseline) * 100000.0;

        /**
         * Algorithm: Metropolis Acceptance Criterion
         * Standard Simulated Annealing state selection logic, now using normalized delta.
         */
        if (delta <= 0.0) {
            // ACCEPT: Decrease (or no change) in energy
            if (delta < 0.0) { // Strictly smaller means an improvement worth reporting
                
                /**
                 * Algorithm: Overwriting Queue Logic
                 * If the 5-second interval allows it, output immediately. 
                 * Otherwise, postpone the snapshot. If another improvement is 
                 * found before the timer resets, the new state simply overwrites 
                 * the older postponed state, ensuring we only queue the best jump.
                 */
                if (difftime(now, last_report_time) >= 5.0) {
                    report_improvement(test_remap, iterations, elapsed, timeout, 
                                       dict_param_kb, lc_param, lp_param, pb_param, 
                                       test_sum, current_sum, initial_sum, baseline_sum, T);
                    last_report_time = now;
                    has_pending_report = 0;
                } else {
                    // Cache the current snapshot into the pending queue
                    memcpy(pending_remap, test_remap, 256);
                    pending_iter = iterations;
                    pending_elapsed = elapsed;
                    pending_new_size = test_sum;
                    pending_prev_size = current_sum;
                    pending_T = T;
                    has_pending_report = 1;
                }
            }
            
            current_sum = test_sum;
            memcpy(current_remap, test_remap, 256);
            
            // Track global best (just in case the final state isn't the absolute lowest)
            if (current_sum < best_sum) {
                best_sum = current_sum;
                memcpy(best_remap, current_remap, 256);
            }
        } else {
            // PROBABILISTIC ACCEPT: Increase in energy (Allows escaping local minima)
            double prob = exp(-delta / T);
            double r = (double)rand() / RAND_MAX;
            
            if (r < prob) {
                // Adopt the worse state to explore the neighborhood
                current_sum = test_sum;
                memcpy(current_remap, test_remap, 256);
            }
        }
    }

    /**
     * Algorithm: End-of-Run Final Flush
     * Ensure any snapshot trapped in the postponed queue when the timeout 
     * is reached gets output cleanly before program termination.
     */
    if (has_pending_report) {
        report_improvement(pending_remap, pending_iter, pending_elapsed, timeout, 
                           dict_param_kb, lc_param, lp_param, pb_param, 
                           pending_new_size, pending_prev_size, initial_sum, baseline_sum, pending_T);
    }

    fprintf(stderr, "/* Simulated Annealing Terminated. Iterations: %llu */\n", iterations);
    fprintf(stderr, "/* Best Found Compressed Size (Fast Settings): %zu bytes */\n\n", best_sum);

    // =========================================================
    // POST-ANNEALING GREEDY REFINEMENT
    // =========================================================
    /**
     * Algorithm: Hill-Climbing Search (Unified)
     * Exploits the best SA state by continuously looping through discrete swaps 
     * involving the victim index. By utilizing explore_swaps directly starting at 
     * depth 1, we cleanly process the entire 0-255 index array without duplicating 
     * any iteration logic. It continues performing full recursive cycles until 
     * an entire pass confirms no further optimizations exist.
     */
    fprintf(stderr, "/* Starting Post-Annealing Greedy Refinement (Depth: %d) ... */\n", greedy_depth);
    
    // Seed the search with the absolute best SA mapping
    memcpy(current_remap, best_remap, 256);
    current_sum = best_sum;
    int improvement_found;
    unsigned long long refinement_iters = 0;

    do {
        refinement_iters++;
        
        // Single unified entry point covering both top-level passes and multi-depth explorations
        improvement_found = explore_swaps(current_remap, &current_sum, 1, greedy_depth, victim_index, 
                                          file_count, files, remapped_buf, out_buf, out_capacity, 
                                          dict_param_kb, lc_param, lp_param, pb_param, refinement_iters);
        
        if (improvement_found) {
            // Track global best securely to handle improvements from deep swaps
            if (current_sum < best_sum) {
                best_sum = current_sum;
                memcpy(best_remap, current_remap, 256);
            }
        }
        
    } while (improvement_found);

    fprintf(stderr, "/* Greedy Refinement Terminated after %llu full passes. */\n", refinement_iters);
    fprintf(stderr, "/* Refined Compressed Size (Fast Settings): %zu bytes */\n\n", best_sum);

    // =========================================================
    // FINAL EVALUATION (Extreme Settings)
    // =========================================================
    /**
     * Algorithm: Absolute Performance Benchmarking
     * Pushes the optimized remap table through a high-cost LZMA compression sequence.
     * Uses preset "6e" (6 | LZMA_PRESET_EXTREME) and an 8MB dict size, while
     * retaining the user-defined command-line overrides for lc, lp, and pb
     * to ascertain the ultimate byte reduction capabilities of the refined 
     * entropy model versus the unmodified baseline.
     */
    fprintf(stderr, "========================================================================\n");
    fprintf(stderr, "FINAL EVALUATION (Preset: 6e, Dictionary: 8MB)\n");
    fprintf(stderr, "========================================================================\n");
    
    uint32_t final_preset = 6 | LZMA_PRESET_EXTREME;
    int final_dict_kb = 8192; // 8MB

    // Because we're passing the base parameters in explicitly, we apply the user-defined lc_param, lp_param, and pb_param
    size_t final_baseline = evaluate_remap(identity_remap, file_count, files, remapped_buf, out_buf, out_capacity, final_preset, final_dict_kb, lc_param, lp_param, pb_param);
    size_t final_best = evaluate_remap(best_remap, file_count, files, remapped_buf, out_buf, out_capacity, final_preset, final_dict_kb, lc_param, lp_param, pb_param);
    
    double final_pct = final_baseline ? ((double)((long long)final_best - (long long)final_baseline) / final_baseline) * 100.0 : 0.0;

    fprintf(stderr, "Identity Remap Size : %zu bytes\n", final_baseline);
    fprintf(stderr, "Best Remap Size     : %zu bytes\n", final_best);
    fprintf(stderr, "Total Reduction     : %+10.3f%%\n\n", final_pct);
    
    print_remap_table_as_source("optimal_remap", best_remap);
    fprintf(stderr, "========================================================================\n");

    // Memory Cleanup
    for (int i = 0; i < file_count; i++) free(files[i].in_buf);
    free(files);
    free(filenames);
    free(remapped_buf); 
    free(out_buf);
    
    return EXIT_SUCCESS;
}
