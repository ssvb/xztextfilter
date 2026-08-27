#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <lzma.h>

/**
 * ==============================================================================
 * ALGORITHM OVERVIEW: Textbook Simulated Annealing
 * ==============================================================================
 * 
 * 1. Frequency Analysis & Victim Selection
 *    - Scans all input files to tally the occurrences of every byte value (0-255).
 *    - Identifies the "Victim Index", which is the byte value with the absolute
 *      lowest global frequency. This byte's mapping will be aggressively 
 *      targeted during the search to limit catastrophic disruption of highly 
 *      frequent bytes.
 * 
 * 2. State Initialization
 *    - Establishes a true Baseline using the Identity Mapping (0->0, 1->1...).
 *    - Randomly reshuffles the identity mapping using the Fisher-Yates algorithm
 *      to create the chaotic Initial State for the annealing process.
 * 
 * 3. Textbook Simulated Annealing (Exponential Cooling)
 *    - Evaluates random neighbor states by swapping the remap table element at
 *      the Victim Index with another element at a random index.
 *    - Uses a deterministic Exponential Cooling Schedule based on the execution
 *      timeout parameter to slowly lower the "Temperature" (T) over time.
 * 
 * 4. Metropolis Acceptance Criterion
 *    - Uses a normalized Energy scale E = (size / baseline_size) * 1000.
 *    - If a swap DECREASES or EQUALS the energy (ΔE <= 0), it is 
 *      unconditionally accepted.
 *    - If a swap INCREASES the energy (ΔE > 0), it is probabilistically
 *      accepted using the Boltzmann distribution: P(accept) = exp(-ΔE / T).
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
    printf("  --help            Show this help message and exit.\n");
    printf("  --timeout=SEC     Set timeout in seconds for annealing (default: 60).\n");
    printf("  --tstart=FLOAT    Set initial temperature for annealing (default: 5000.0).\n");
    printf("  --dict=KB         Set LZMA dictionary size in kilobytes (default: 4).\n");
    printf("  --lc=BITS         Set LZMA literal context bits (1-4, default: 3).\n");
    printf("  --lp=BITS         Set LZMA literal position bits (0-4, default: 0).\n");
    printf("  --pb=BITS         Set LZMA position bits (0-4, default: 2).\n");
    printf("\n");
}

/**
 * Algorithm: LZMA Compression Wrapper
 * Sets up LZMA2 filters and compresses an in-memory buffer. 
 * Returns the output size or out_capacity + 1 if compression fails/exceeds limits.
 */
size_t compress_buffer(const uint8_t *in_buf, size_t in_len, uint8_t *out_buf, size_t out_capacity, int dict_param_kb, int lc_param, int lp_param, int pb_param) {
    lzma_options_lzma opt;
    if (lzma_lzma_preset(&opt, 0)) return out_capacity + 1;
    opt.dict_size = ((uint32_t)dict_param_kb * 1024 < LZMA_DICT_SIZE_MIN) ? LZMA_DICT_SIZE_MIN : (uint32_t)dict_param_kb * 1024; 
    opt.lc = lc_param; opt.lp = lp_param; opt.pb = pb_param;                         
    lzma_filter filters[2] = { { .id = LZMA_FILTER_LZMA2, .options = &opt }, { .id = LZMA_VLI_UNKNOWN, .options = NULL } };
    size_t out_pos = 0;
    lzma_ret ret = lzma_stream_buffer_encode(filters, LZMA_CHECK_CRC32, NULL, in_buf, in_len, out_buf, &out_pos, out_capacity);
    return (ret == LZMA_OK) ? out_pos : out_capacity + 1;
}

/**
 * Algorithm: Aggregate Evaluation
 * Applies the provided remap_table to all input files and calculates 
 * the total sum of their compressed sizes.
 */
size_t evaluate_remap(const uint8_t *remap_table, int file_count, FileData *files, 
                      uint8_t *remapped_buf, uint8_t *out_buf, size_t out_capacity, 
                      int dict, int lc, int lp, int pb) {
    size_t total = 0;
    for (int i = 0; i < file_count; i++) {
        for (size_t j = 0; j < files[i].file_size; j++) {
            remapped_buf[j] = remap_table[files[i].in_buf[j]];
        }
        total += compress_buffer(remapped_buf, files[i].file_size, out_buf, out_capacity, dict, lc, lp, pb);
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
                        size_t new_size, size_t prev_size, size_t initial_size, size_t baseline_size) {
    
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
    fprintf(stderr, "\nCurrent Remap Table Dump:\n");
    print_remap_table_as_source("current_remap", remap);
    fprintf(stderr, "========================================================================\n\n");
}

int main(int argc, char **argv) {
    int timeout = 60;
    double T_start = 5000.0; // Default initial temperature
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

    // Evaluate Identity (Baseline) Mapping
    uint8_t identity_remap[256];
    for (int i = 0; i < 256; i++) identity_remap[i] = i;
    size_t baseline_sum = evaluate_remap(identity_remap, file_count, files, remapped_buf, out_buf, out_capacity, dict_param_kb, lc_param, lp_param, pb_param);
    
    // Prevent division by zero during energy normalization if baseline is somehow 0
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
    
    size_t initial_sum = evaluate_remap(current_remap, file_count, files, remapped_buf, out_buf, out_capacity, dict_param_kb, lc_param, lp_param, pb_param);
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
                               pending_new_size, pending_prev_size, initial_sum, baseline_sum);
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

        // Evaluate the neighbor state
        size_t test_sum = evaluate_remap(test_remap, file_count, files, remapped_buf, out_buf, out_capacity, dict_param_kb, lc_param, lp_param, pb_param);

        /**
         * Algorithm: Normalized Energy Delta Calculation
         * Evaluates the relative change in compressed size rather than absolute bytes.
         * Energy E = (compressed_size / baseline_size) * 1000
         * By normalizing against the baseline size, the energy landscape becomes 
         * invariant to the raw size of the input files, making the Temperature (T)
         * scale highly reusable across different datasets.
         */
        double delta = (((double)test_sum - (double)current_sum) / safe_baseline) * 1000.0;

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
                                       test_sum, current_sum, initial_sum, baseline_sum);
                    last_report_time = now;
                    has_pending_report = 0; // Clear any stale flags
                } else {
                    // Cache the current snapshot into the pending queue
                    memcpy(pending_remap, test_remap, 256);
                    pending_iter = iterations;
                    pending_elapsed = elapsed;
                    pending_new_size = test_sum;
                    pending_prev_size = current_sum;
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
                           pending_new_size, pending_prev_size, initial_sum, baseline_sum);
    }

    fprintf(stderr, "/* Simulated Annealing Terminated. Iterations: %llu */\n", iterations);
    fprintf(stderr, "/* Best Found Compressed Size: %zu bytes */\n\n", best_sum);

    // Memory Cleanup
    for (int i = 0; i < file_count; i++) free(files[i].in_buf);
    free(files);
    free(filenames);
    free(remapped_buf); 
    free(out_buf);
    
    return EXIT_SUCCESS;
}
