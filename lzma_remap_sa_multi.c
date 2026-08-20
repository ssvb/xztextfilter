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
 *    - Baseline: Size under pure identity mapping (fixed anchor).
 *    - Milestone: Size at the last recorded global best (ratchets down).
 *    - Tabular Formatting: Calculates maximum filename width dynamically to 
 *      horizontally align all metric columns (sizes, byte deltas, percentages).
 * ==============================================================================
 */

#define INITIAL_TEMPERATURE 50.0

/* 
 * REPLACE THIS BLOCK with the output from stderr to use the best mapping 
 * as the new seed for the next run. Initialized here to the Identity Mapping
 * using spaced character literals for perfect column alignment.
 */
unsigned char seed_remap[256] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
    ' ' , '!' , '"' , '#' , '$' , '%' , '&' , '\'', '(' , ')' , '*' , '+' , ',' , '-' , '.' , '/' ,
    '0' , '1' , '2' , '3' , '4' , '5' , '6' , '7' , '8' , '9' , ':' , ';' , '<' , '=' , '>' , '?' ,
    '@' , 'A' , 'B' , 'C' , 'D' , 'E' , 'F' , 'G' , 'H' , 'I' , 'J' , 'K' , 'L' , 'M' , 'N' , 'O' ,
    'P' , 'Q' , 'R' , 'S' , 'T' , 'U' , 'V' , 'W' , 'X' , 'Y' , 'Z' , '[' , '\\', ']' , '^' , '_' ,
    '`' , 'a' , 'b' , 'c' , 'd' , 'e' , 'f' , 'g' , 'h' , 'i' , 'j' , 'k' , 'l' , 'm' , 'n' , 'o' ,
    'p' , 'q' , 'r' , 's' , 't' , 'u' , 'v' , 'w' , 'x' , 'y' , 'z' , '{' , '|' , '}' , '~' , 0x7f,
    0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e, 0x8f,
    0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0x9b, 0x9c, 0x9d, 0x9e, 0x9f,
    0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xab, 0xac, 0xad, 0xae, 0xaf,
    0xb0, 0xb1, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xbb, 0xbc, 0xbd, 0xbe, 0xbf,
    0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xcb, 0xcc, 0xcd, 0xce, 0xcf,
    0xd0, 0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xdb, 0xdc, 0xdd, 0xde, 0xdf,
    0xe0, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xeb, 0xec, 0xed, 0xee, 0xef,
    0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff
};

typedef struct {
    const char *filename;
    uint8_t *in_buf;
    size_t file_size;
    size_t baseline_comp_size;  // Pure identity baseline
    size_t current_comp_size;   // Evaluated size of the current working SA state
    size_t milestone_comp_size; // Evaluated size at the last global best
    size_t temp_comp_size;      // Evaluated size during a candidate mutation
} FileData;

void print_help(const char *prog_name) {
    printf("Usage: %s [OPTIONS] <input_file_1> [input_file_2 ...]\n\n", prog_name);
    printf("Options:\n");
    printf("  --help          Show this help message and exit.\n");
    printf("  --timeout=SEC   Set timeout in seconds (default: 600).\n");
    printf("  --lc=BITS       Set LZMA literal context bits (0-4, default: 3).\n");
    printf("  --lp=BITS       Set LZMA literal position bits (0-4, default: 0).\n");
    printf("  --pb=BITS       Set LZMA position bits (0-4, default: 2).\n");
    printf("\n");
    printf("Description:\n");
    printf("  Optimizes byte mappings to minimize LZMA compressed size across provided files.\n");
    printf("  Uses Simulated Annealing and continuously outputs the best mapping to stderr.\n");
}

// Function to evaluate the objective (compress buffer with tuned LZMA properties)
size_t compress_buffer(const uint8_t *in_buf, size_t in_len, uint8_t *out_buf, size_t out_capacity, int lc_param, int lp_param, int pb_param) {
    lzma_options_lzma opt;
    if (lzma_lzma_preset(&opt, 0)) {
        fprintf(stderr, "Error: Failed to set LZMA preset.\n");
        exit(EXIT_FAILURE);
    }
    
    opt.dict_size = LZMA_DICT_SIZE_MIN; 
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

    // 1. Evaluate True Baseline (Identity Mapping)
    size_t baseline_sum = 0;
    for (int i = 0; i < file_count; i++) {
        for (size_t j = 0; j < files[i].file_size; j++) {
            remapped_buf[j] = files[i].in_buf[j];
        }
        size_t s = compress_buffer(remapped_buf, files[i].file_size, out_buf, out_capacity, lc_param, lp_param, pb_param);
        files[i].baseline_comp_size = s;
        baseline_sum += s;
    }
    
    fprintf(stderr, "Original Identity Mapping Total Size: %zu bytes\n\n", baseline_sum);

    // 2. Evaluate User Seed
    uint8_t current_remap[256];
    memcpy(current_remap, seed_remap, 256);

    size_t current_sum = 0;
    for (int i = 0; i < file_count; i++) {
        for (size_t j = 0; j < files[i].file_size; j++) {
            remapped_buf[j] = current_remap[files[i].in_buf[j]];
        }
        size_t s = compress_buffer(remapped_buf, files[i].file_size, out_buf, out_capacity, lc_param, lp_param, pb_param);
        files[i].current_comp_size = s;
        files[i].milestone_comp_size = s; // The seed is our first milestone
        current_sum += s;
    }
    
    size_t best_sum = current_sum;
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
            files[i].temp_comp_size = compress_buffer(remapped_buf, files[i].file_size, out_buf, out_capacity, lc_param, lp_param, pb_param);
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
                
                long long total_delta_prev = (long long)new_sum - (long long)best_sum;
                double total_pct_prev = ((double)total_delta_prev / (double)best_sum) * 100.0;

                fprintf(stderr, "/* NEW GLOBAL BEST: %zu bytes (Iter %llu, Temp %.2f, Op: %s, lc:%d/lp:%d/pb:%d) */\n", 
                        new_sum, iterations, temperature, op_type, lc_param, lp_param, pb_param);
                
                fprintf(stderr, "/* TOTAL IMPROVEMENT:\n");
                fprintf(stderr, "      vs Identity:  %lld bytes (%+.2f%%)\n", total_delta_base, total_pct_base);
                fprintf(stderr, "      vs Milestone: %lld bytes (%+.2f%%)\n", total_delta_prev, total_pct_prev);
                fprintf(stderr, "*/\n");
                
                /*
                 * Horizontally aligned column output using max_filename_len and fixed-width specifiers:
                 * - Filename left-aligned to longest filename length (%-*s)
                 * - Byte sizes, deltas, and percentages right-aligned to fixed widths
                 */
                fprintf(stderr, "/* FILE METRICS [vs Identity]:\n");
                for (int i = 0; i < file_count; i++) {
                    long long delta_base = (long long)files[i].temp_comp_size - (long long)files[i].baseline_comp_size;
                    double pct_base = ((double)delta_base / (double)files[i].baseline_comp_size) * 100.0;
                    fprintf(stderr, "   - %-*s : %10zu bytes (%+10lld B, %+7.2f%%)\n", 
                            max_filename_len, files[i].filename, files[i].temp_comp_size, delta_base, pct_base);
                }
                
                fprintf(stderr, "   FILE METRICS [vs Previous Milestone]:\n");
                for (int i = 0; i < file_count; i++) {
                    long long delta_prev = (long long)files[i].temp_comp_size - (long long)files[i].milestone_comp_size;
                    double pct_prev = ((double)delta_prev / (double)files[i].milestone_comp_size) * 100.0;
                    fprintf(stderr, "   - %-*s : %+10lld bytes (%+7.2f%%)\n", 
                            max_filename_len, files[i].filename, delta_prev, pct_prev);
                    
                    // Lock in this file's state for the new milestone
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
