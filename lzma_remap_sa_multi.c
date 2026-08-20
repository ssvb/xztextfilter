#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <lzma.h>

/**
 * ==============================================================================
 * ALGORITHM OVERVIEW: Multi-Target Simulated Annealing with Defragmentation
 * ==============================================================================
 * 
 * 1. The Global Objective (Multi-File Optimization)
 *    The objective function calculates the SUM of the compressed sizes of all 
 *    input files. This guides the search toward a universal byte permutation 
 *    that minimizes entropy across the entire dataset.
 * 
 * 2. Tunable LZMA Markov Chain Modeling
 *    - dict (Dictionary Size): Configurable via '--dict=' in KB (default: 4).
 *      Defaults to the 4KB minimum to drastically increase evaluation speed and 
 *      to force the objective function to rely heavily on the literal/entropy 
 *      coding (Markov chains) rather than finding long historical LZ matches.
 *    - lc (Literal Context): Configurable via '--lc=' (default: 3). 
 *      Determines how many of the highest bits of the PREVIOUS byte are used as 
 *      the context to select the probability model for encoding the CURRENT byte. 
 *    - lp (Literal Position): Configurable via '--lp=' (default: 0).
 *      Determines how many of the lowest bits of the byte's position are used 
 *      for literal encoding.
 *    - pb (Match Position): Configurable via '--pb=' (default: 2).
 *      Determines how many of the lowest bits of the byte's position are used 
 *      for match encoding. Default pb=2 aligns with standard LZMA file settings.
 * 
 * 3. Competing Entropy & Probabilistic Acceptance
 *    - Global improvements are accepted unconditionally.
 *    - Global regressions are accepted probabilistically: P = exp(-dE / T). 
 *    This allows the algorithm to escape local minima. The temperature T cools 
 *    linearly over the timeout period.
 * 
 * 4. Dual Mutation Strategy (Random Swap vs. Defragmentation)
 *    - Random Swap (75%): Swaps two entirely random indices in the mapping.
 *    - Defragmentation (25%): Picks an index 'i', targets the value one higher 
 *      than remap[i], finds where that target value is currently located ('j'), 
 *      and swaps remap[i+1] with remap[j]. This encourages creating contiguous, 
 *      ascending byte sequences, which greatly benefits LZMA's Markov chain 
 *      modeling when evaluating serialized data streams.
 * 
 * 5. State & Milestone Tracking & Formatting
 *    The algorithm maintains distinct evaluation anchors to track progress:
 *    - Nonremapped (Baseline): Size under pure identity mapping.
 *    - Initial: Size evaluated using the provided seed_remap array.
 *    - Previous (Milestone): Size at the last recorded global best.
 * ==============================================================================
 */

#define INITIAL_TEMPERATURE 50.0

/* 
 * REPLACE THIS BLOCK with the output from stderr to use the best mapping 
 * as the new seed for the next run. Initialized here to the Identity Mapping
 * using spaced character literals for perfect column alignment.
 */
unsigned char seed_remap[256] = {
    0xc9, 0x16, 0x17, 0xb5, ' ' , '!' , 0xdd, 0xac, '(' , ')' , 0xf9, 'v' , 't' , 0xdb, 0xd0, 0x08, 
    0x93, 0x94, 0x95, 0x9e, 0xb8, 0x85, 0x99, 0xb9, 0xe4, '{' , '\'', 0x83, 0x9b, 0x8c, 0x8d, 0x0a, 
    0xbe, 0xc0, 0xa1, 'X' , 0xa8, 0xa9, 0xc4, 0x9d, 0x05, 0xd1, 0xe7, 0xc6, 0xad, 0xae, 0xcd, 0xbf, 
    0xe5, 0xb1, 0x9c, 0xd2, 0xfd, 0xfe, 0xda, 0xaf, 0xb0, 0xa5, 0xa6, 0xa7, 0xff, 'H' , 0xec, 0xd6, 
    0xd7, 'T' , '^' , '_' , ']' , 'E' , '>' , '?' , 'Y' , 'S' , '.' , 0xbd, '&' , 'P' , 'Q' , 'R' , 
    'N' , 'O' , 'A' , 'B' , 'C' , 'D' , '/' , 'L' , 'U' , '[' , '@' , 0xe1, 0xc3, 0xa0, 0x96, '8' , 
    '4' , '`' , 'u' , 'h' , 'i' , 'b' , 'k' , 'l' , 'f' , 'g' , 'p' , 'q' , '~' , 0x7f, 'm' , 'e' , 
    '}' , ',' , '|' , 'o' , 'n' , 'c' , 'x' , 'y' , 'z' , 'r' , '-' , '6' , 0x1b, 0x15, 'V' , 0x90, 
    0xaa, 0xe3, 0xf6, 0x86, '1' , '%' , 0xbc, 0x11, 0x91, 0x92, 0xf1, 0x1f, 'd' , '#' , '$' , 'w' , 
    0xdf, 'W' , 0x04, 0xe6, 0xd5, 0xe0, '5' , '\\', 0xc2, 0xf2, 0xcc, 0x1c, 0xa4, 0xe8, 0xfa, 0xea, 
    0xeb, 0x0d, 0x0f, 'J' , 0xce, 0xd9, 0x02, 'Z' , 0x9f, 'a' , 0xab, '3' , 0xee, 0xb4, 0xf7, '0' , 
    0x87, 0xed, 0x84, 0x98, 0x82, 0x18, 0xfb, 'I' , 0xc5, 0xba, 0xf5, 0xd4, '*' , 'F' , 'K' , 0x8e, 
    0x9a, 0xf8, 0x0c, 0x8b, 's' , 0x07, 0xb7, 0xfc, 0x8f, 0xa2, 0xe2, 0x03, '9' , 0x10, 0xe9, 0x89, 
    0xdc, 0x00, 0xf4, '2' , 0xef, 0xf0, 0x0b, 0x8a, 0xcf, 0x81, 0x14, 0x80, 0xd8, 0xca, 0x01, 0x1d, 
    '<' , '=' , 0xde, 0xd3, 0xbb, '+' , 'M' , '7' , 0x09, 'j' , 0x1e, 0x0e, '"' , 0xc8, 0xa3, 0x88, 
    0xc7, 0xf3, 0x97, 0xb6, 0xb3, 0xcb, 0x19, 0x1a, 'G' , 0x12, 0x06, ':' , 0x13, ';' , 0xc1, 0xb2
};

typedef struct {
    const char *filename;
    uint8_t *in_buf;
    size_t file_size;
    size_t baseline_comp_size;  // Pure identity baseline ("Nonremapped")
    size_t initial_comp_size;   // Seed remap ("Initial")
    size_t current_comp_size;   // Evaluated size of the current working SA state
    size_t milestone_comp_size; // Evaluated size at the last global best ("Previous")
    size_t temp_comp_size;      // Evaluated size during a candidate mutation
} FileData;

void print_help(const char *prog_name) {
    printf("Usage: %s [OPTIONS] <input_file_1> [input_file_2 ...]\n\n", prog_name);
    printf("Options:\n");
    printf("  --help          Show this help message and exit.\n");
    printf("  --timeout=SEC   Set timeout in seconds (default: 600).\n");
    printf("  --dict=KB       Set LZMA dictionary size in kilobytes (default: 4).\n");
    printf("  --lc=BITS       Set LZMA literal context bits (0-4, default: 3).\n");
    printf("  --lp=BITS       Set LZMA literal position bits (0-4, default: 0).\n");
    printf("  --pb=BITS       Set LZMA position bits (0-4, default: 2).\n");
    printf("\n");
    printf("Description:\n");
    printf("  Optimizes byte mappings to minimize LZMA compressed size across provided files.\n");
    printf("  Uses Simulated Annealing and continuously outputs the best mapping to stderr.\n");
}

// Function to evaluate the objective (compress buffer with tuned LZMA properties)
size_t compress_buffer(const uint8_t *in_buf, size_t in_len, uint8_t *out_buf, size_t out_capacity, int dict_param_kb, int lc_param, int lp_param, int pb_param) {
    lzma_options_lzma opt;
    if (lzma_lzma_preset(&opt, 0)) {
        fprintf(stderr, "Error: Failed to set LZMA preset.\n");
        exit(EXIT_FAILURE);
    }
    
    // LZMA requires a minimum dictionary size of 4096 bytes (4 KB).
    uint32_t dict_bytes = (uint32_t)dict_param_kb * 1024;
    opt.dict_size = (dict_bytes < LZMA_DICT_SIZE_MIN) ? LZMA_DICT_SIZE_MIN : dict_bytes; 
    
    opt.lc = lc_param;                         
    opt.lp = lp_param;                         
    opt.pb = pb_param;                         

    lzma_filter filters[2] = {
        { .id = LZMA_FILTER_LZMA2, .options = &opt },
        { .id = LZMA_VLI_UNKNOWN, .options = NULL }
    };

    size_t out_pos = 0;
    lzma_ret ret = lzma_stream_buffer_encode(
        filters, LZMA_CHECK_CRC32, NULL, 
        in_buf, in_len, out_buf, &out_pos, out_capacity
    );

    return (ret == LZMA_OK) ? out_pos : out_capacity + 1;
}

// Helper function to print bytes with space padding for alignment
void print_byte_literal(FILE *out, uint8_t b) {
    if (b >= 32 && b <= 126) {
        if (b == '\'') {
            fprintf(out, "'\\''"); 
        } else if (b == '\\') {
            fprintf(out, "'\\\\'"); 
        } else {
            fprintf(out, "'%c' ", b); 
        }
    } else {
        fprintf(out, "0x%02x", b); 
    }
}

// Dumps the current best mapping exactly as a perfectly aligned C array
void print_remap_table_as_source(const uint8_t *remap) {
    fprintf(stderr, "unsigned char seed_remap[256] = {\n    ");
    for (int i = 0; i < 256; i++) {
        print_byte_literal(stderr, remap[i]);
        if (i < 255) {
            fprintf(stderr, ", ");
        }
        if ((i + 1) % 16 == 0 && i < 255) {
            fprintf(stderr, "\n    ");
        }
    }
    fprintf(stderr, "\n};\n\n");
    fflush(stderr); 
}

int main(int argc, char **argv) {
    int timeout = 600; 
    int dict_param_kb = 4; // Updated default for increased speed & entropy coding priority
    int lc_param = 3; 
    int lp_param = 0;
    int pb_param = 2;
    
    const char **filenames = malloc(argc * sizeof(char*));
    int file_count = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_help(argv[0]);
            free(filenames);
            return EXIT_SUCCESS;
        } else if (strncmp(argv[i], "--timeout=", 10) == 0) {
            timeout = atoi(argv[i] + 10);
        } else if (strncmp(argv[i], "--timout=", 9) == 0) { 
            timeout = atoi(argv[i] + 9);
        } else if (strncmp(argv[i], "--dict=", 7) == 0) {
            dict_param_kb = atoi(argv[i] + 7);
        } else if (strncmp(argv[i], "--lc=", 5) == 0) {
            lc_param = atoi(argv[i] + 5);
        } else if (strncmp(argv[i], "--lp=", 5) == 0) {
            lp_param = atoi(argv[i] + 5);
        } else if (strncmp(argv[i], "--pb=", 5) == 0) {
            pb_param = atoi(argv[i] + 5);
        } else {
            filenames[file_count++] = argv[i];
        }
    }

    if (file_count == 0) {
        fprintf(stderr, "Error: No input files specified.\n");
        print_help(argv[0]);
        free(filenames);
        return EXIT_FAILURE;
    }

    FileData *files = malloc(file_count * sizeof(FileData));
    size_t max_file_size = 0;
    int max_filename_len = 0;

    for (int i = 0; i < file_count; i++) {
        files[i].filename = filenames[i];
        
        // Calculate max filename length for perfect alignment in report
        int fn_len = (int)strlen(filenames[i]);
        if (fn_len > max_filename_len) {
            max_filename_len = fn_len;
        }

        FILE *f = fopen(filenames[i], "rb");
        if (!f) {
            fprintf(stderr, "Failed to open %s\n", filenames[i]);
            return EXIT_FAILURE;
        }

        fseek(f, 0, SEEK_END);
        files[i].file_size = (size_t)ftell(f);
        fseek(f, 0, SEEK_SET);

        if (files[i].file_size > max_file_size) {
            max_file_size = files[i].file_size;
        }

        files[i].in_buf = malloc(files[i].file_size);
        if (fread(files[i].in_buf, 1, files[i].file_size, f) != files[i].file_size) {
            fprintf(stderr, "Error reading %s\n", filenames[i]);
            return EXIT_FAILURE;
        }
        fclose(f);
    }

    uint8_t *remapped_buf = malloc(max_file_size);
    size_t out_capacity = lzma_stream_buffer_bound(max_file_size);
    uint8_t *out_buf = malloc(out_capacity);

    // 1. Evaluate True Baseline ("Nonremapped" / Identity Mapping)
    size_t baseline_sum = 0;
    for (int i = 0; i < file_count; i++) {
        for (size_t j = 0; j < files[i].file_size; j++) {
            remapped_buf[j] = files[i].in_buf[j];
        }
        size_t s = compress_buffer(remapped_buf, files[i].file_size, out_buf, out_capacity, dict_param_kb, lc_param, lp_param, pb_param);
        files[i].baseline_comp_size = s;
        baseline_sum += s;
    }
    
    fprintf(stderr, "Original Identity Mapping Total Size: %zu bytes\n\n", baseline_sum);

    // 2. Evaluate User Seed ("Initial")
    uint8_t current_remap[256];
    memcpy(current_remap, seed_remap, 256);

    size_t initial_sum = 0;
    for (int i = 0; i < file_count; i++) {
        for (size_t j = 0; j < files[i].file_size; j++) {
            remapped_buf[j] = current_remap[files[i].in_buf[j]];
        }
        size_t s = compress_buffer(remapped_buf, files[i].file_size, out_buf, out_capacity, dict_param_kb, lc_param, lp_param, pb_param);
        files[i].initial_comp_size = s;
        files[i].current_comp_size = s;
        files[i].milestone_comp_size = s; // First "Previous" milestone is the Initial state
        initial_sum += s;
    }
    
    size_t current_sum = initial_sum;
    size_t best_sum = initial_sum;
    fprintf(stderr, "Starting Seed Total Size: %zu bytes\n\n", best_sum);

    srand((unsigned int)time(NULL));
    time_t start_time = time(NULL);
    unsigned long long iterations = 0;

    // =========================================================
    // THE OPTIMIZATION LOOP
    // =========================================================
    while (1) {
        time_t now = time(NULL);
        double elapsed = difftime(now, start_time);
        if (elapsed >= timeout) break;

        iterations++;
        double progress = elapsed / (double)timeout;
        double temperature = INITIAL_TEMPERATURE * (1.0 - progress);
        if (temperature < 0.001) temperature = 0.001; 

        // 3. Mutation Phase (Choose between Random or Defragment)
        int is_defrag = (rand() % 4 == 0); // 25% chance of defragmenting
        int idx1, idx2;
        const char *op_type;

        if (is_defrag) {
            int i = rand() % 255; 
            if (current_remap[i] == 0xFF) continue; 
            
            uint8_t target_val = current_remap[i] + 1;
            int j = 0;
            for (; j < 256; j++) {
                if (current_remap[j] == target_val) break;
            }
            
            idx1 = i + 1;
            idx2 = j;
            op_type = "defragment";
            
            if (idx1 == idx2) continue; 
        } else {
            idx1 = rand() % 256;
            idx2 = rand() % 256;
            if (idx1 == idx2) continue;
            op_type = "random";
        }

        // Apply Swap
        uint8_t temp = current_remap[idx1];
        current_remap[idx1] = current_remap[idx2];
        current_remap[idx2] = temp;

        // 4. Evaluation Phase
        size_t new_sum = 0;
        for (int i = 0; i < file_count; i++) {
            for (size_t j = 0; j < files[i].file_size; j++) {
                remapped_buf[j] = current_remap[files[i].in_buf[j]];
            }
            files[i].temp_comp_size = compress_buffer(remapped_buf, files[i].file_size, out_buf, out_capacity, dict_param_kb, lc_param, lp_param, pb_param);
            new_sum += files[i].temp_comp_size;
        }

        // 5. Metropolis Acceptance Criterion
        int accept = 0;
        if (new_sum < current_sum) {
            accept = 1; 
        } else {
            double diff = (double)(new_sum - current_sum);
            double p_accept = exp(-diff / temperature);
            double r = (double)rand() / RAND_MAX;
            if (r < p_accept) {
                accept = 1; 
            }
        }

        if (accept) {
            // Apply temp state to current running state
            current_sum = new_sum;
            for (int i = 0; i < file_count; i++) {
                files[i].current_comp_size = files[i].temp_comp_size;
            }
            
            // Check against GLOBAL absolute minimum (Milestone)
            if (new_sum < best_sum) {
                
                // Calculate total deltas
                long long total_delta_base = (long long)new_sum - (long long)baseline_sum;
                double total_pct_base = ((double)total_delta_base / (double)baseline_sum) * 100.0;
                
                long long total_delta_init = (long long)new_sum - (long long)initial_sum;
                double total_pct_init = ((double)total_delta_init / (double)initial_sum) * 100.0;
                
                long long total_delta_prev = (long long)new_sum - (long long)best_sum;
                double total_pct_prev = ((double)total_delta_prev / (double)best_sum) * 100.0;

                fprintf(stderr, "/* NEW GLOBAL BEST: %zu bytes (Iter %llu, Temp %.2f, Op: %s, dict:%dK/lc:%d/lp:%d/pb:%d) */\n", 
                        new_sum, iterations, temperature, op_type, dict_param_kb, lc_param, lp_param, pb_param);
                
                fprintf(stderr, "/* TOTAL IMPROVEMENT:\n");
                fprintf(stderr, "      vs Nonremapped: %lld bytes (%+.2f%%)\n", total_delta_base, total_pct_base);
                fprintf(stderr, "      vs Initial:     %lld bytes (%+.2f%%)\n", total_delta_init, total_pct_init);
                fprintf(stderr, "      vs Previous:    %lld bytes (%+.2f%%)\n", total_delta_prev, total_pct_prev);
                fprintf(stderr, "*/\n");
                
                fprintf(stderr, "/* FILE METRICS [vs Nonremapped]:\n");
                for (int i = 0; i < file_count; i++) {
                    long long delta = (long long)files[i].temp_comp_size - (long long)files[i].baseline_comp_size;
                    double pct = ((double)delta / (double)files[i].baseline_comp_size) * 100.0;
                    fprintf(stderr, "   - %-*s : %10zu bytes (%+10lld B, %+7.2f%%)\n", 
                            max_filename_len, files[i].filename, files[i].temp_comp_size, delta, pct);
                }
                
                fprintf(stderr, "   FILE METRICS [vs Initial]:\n");
                for (int i = 0; i < file_count; i++) {
                    long long delta = (long long)files[i].temp_comp_size - (long long)files[i].initial_comp_size;
                    double pct = ((double)delta / (double)files[i].initial_comp_size) * 100.0;
                    fprintf(stderr, "   - %-*s : %+10lld bytes (%+7.2f%%)\n", 
                            max_filename_len, files[i].filename, delta, pct);
                }
                
                fprintf(stderr, "   FILE METRICS [vs Previous]:\n");
                for (int i = 0; i < file_count; i++) {
                    long long delta = (long long)files[i].temp_comp_size - (long long)files[i].milestone_comp_size;
                    double pct = ((double)delta / (double)files[i].milestone_comp_size) * 100.0;
                    fprintf(stderr, "   - %-*s : %+10lld bytes (%+7.2f%%)\n", 
                            max_filename_len, files[i].filename, delta, pct);
                    
                    // Lock in this file's state for the new sequential milestone
                    files[i].milestone_comp_size = files[i].temp_comp_size;
                }
                fprintf(stderr, "*/\n");
                
                // Lock in the new global minimum
                best_sum = new_sum;
                print_remap_table_as_source(current_remap);
            }
        } else {
            // Rejection: Undo swap
            current_remap[idx2] = current_remap[idx1];
            current_remap[idx1] = temp;
        }
    }

    fprintf(stderr, "/* Search finished. Iterations: %llu. Final Best Sum: %zu bytes */\n", iterations, best_sum);

    // Cleanup
    for (int i = 0; i < file_count; i++) {
        free(files[i].in_buf);
    }
    free(files);
    free(filenames);
    free(remapped_buf); 
    free(out_buf);
    
    return EXIT_SUCCESS;
}
