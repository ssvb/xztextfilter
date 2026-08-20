#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <lzma.h>

/**
 * ==============================================================================
 * ALGORITHM OVERVIEW: Multi-Target Simulated Annealing
 * ==============================================================================
 * 
 * 1. The Global Objective (Multi-File Optimization)
 *    Instead of minimizing the compressed size of a single file, the objective 
 *    function now calculates the SUM of the compressed sizes of all input files.
 *    This forces the algorithm to find a single, universal byte permutation that 
 *    generalizes well across the entire dataset.
 * 
 * 2. Competing Entropy (Local Regression for Global Gain)
 *    Because the files may have vastly different underlying byte distributions, 
 *    a swap that heavily optimizes File A might slightly de-optimize File B.
 *    By summing the sizes, the algorithm naturally performs tradeoffs, sacrificing 
 *    compression on one file if it yields a mathematically larger overall saving.
 * 
 * 3. LZMA Model Tuning (lc=1, lp=0, pb=0)
 *    - lc (Literal Context): 1
 *    - lp (Literal Position): 0
 *    - pb (Match Position): 0
 *    Position-independent compression ensures the permutation is evaluated purely 
 *    on sequence structure, preventing alignment artifacts from skewing results.
 * 
 * 4. Thermodynamic Acceptance (Simulated Annealing)
 *    Energy (E) = Sum of all compressed bitstreams.
 *    Delta (dE) = New Energy - Current Energy.
 *    We accept all improving mutations (dE < 0). For regressions (dE > 0), 
 *    we accept them probabilistically based on P = exp(-dE / T). The temperature
 *    T cools linearly from INITIAL_TEMPERATURE down to 0 over the timeout period.
 * ==============================================================================
 */

#define INITIAL_TEMPERATURE 50.0

/* 
 * REPLACE THIS BLOCK with the output from stderr to use the best mapping 
 * as the new seed for the next run. Initialized here to the Identity Mapping.
 */
unsigned char seed_remap[256] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
    0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f,
    0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f,
    0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f,
    0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x5b, 0x5c, 0x5d, 0x5e, 0x5f,
    0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f,
    0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x7b, 0x7c, 0x7d, 0x7e, 0x7f,
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
    size_t current_comp_size; // The size under the current working table
    size_t temp_comp_size;    // The size during a candidate mutation
    size_t best_comp_size;    // The size recorded during the last global best
} FileData;

// Function to evaluate the objective (compress buffer with tuned LZMA properties)
size_t compress_buffer(const uint8_t *in_buf, size_t in_len, uint8_t *out_buf, size_t out_capacity) {
    lzma_options_lzma opt;
    if (lzma_lzma_preset(&opt, 0)) {
        fprintf(stderr, "Error: Failed to set LZMA preset.\n");
        exit(EXIT_FAILURE);
    }
    
    opt.dict_size = LZMA_DICT_SIZE_MIN; 
    opt.lc = 1;                         
    opt.lp = 0;                         
    opt.pb = 0;                         

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

// Dumps the current best mapping exactly as a C array
void print_remap_table_as_source(const uint8_t *remap) {
    fprintf(stderr, "unsigned char seed_remap[256] = {\n    ");
    for (int i = 0; i < 256; i++) {
        fprintf(stderr, "0x%02x%s", remap[i], (i < 255) ? ", " : "");
        if ((i + 1) % 16 == 0 && i < 255) {
            fprintf(stderr, "\n    ");
        }
    }
    fprintf(stderr, "\n};\n\n");
    fflush(stderr); 
}

int main(int argc, char **argv) {
    int timeout = 600; // Default timeout: 10 minutes (600 seconds)
    
    // Allocate space to store pointers to the provided filenames
    const char **filenames = malloc(argc * sizeof(char*));
    int file_count = 0;

    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--timeout=", 10) == 0) {
            timeout = atoi(argv[i] + 10);
        } else if (strncmp(argv[i], "--timout=", 9) == 0) {
            timeout = atoi(argv[i] + 9);
        } else {
            filenames[file_count++] = argv[i];
        }
    }

    if (file_count == 0) {
        fprintf(stderr, "Usage: %s [--timeout=<seconds>] <input_file_1> [input_file_2 ...]\n", argv[0]);
        return EXIT_FAILURE;
    }

    FileData *files = malloc(file_count * sizeof(FileData));
    size_t max_file_size = 0;

    // Load all files into memory
    for (int i = 0; i < file_count; i++) {
        files[i].filename = filenames[i];
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

    // Shared working buffers dimensioned to the largest file in the batch
    uint8_t *remapped_buf = malloc(max_file_size);
    size_t out_capacity = lzma_stream_buffer_bound(max_file_size);
    uint8_t *out_buf = malloc(out_capacity);

    uint8_t current_remap[256];
    memcpy(current_remap, seed_remap, 256);

    // Initial evaluation phase: Calculate baseline energy (sum of all file sizes)
    size_t current_sum = 0;
    for (int i = 0; i < file_count; i++) {
        for (size_t j = 0; j < files[i].file_size; j++) {
            remapped_buf[j] = current_remap[files[i].in_buf[j]];
        }
        size_t s = compress_buffer(remapped_buf, files[i].file_size, out_buf, out_capacity);
        
        files[i].current_comp_size = s;
        files[i].best_comp_size = s;
        current_sum += s;
    }
    
    size_t best_sum = current_sum;
    fprintf(stderr, "Starting seed total size: %zu bytes\n\n", best_sum);

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

        // 1. Mutation Phase (Create a neighbor state)
        int idx1 = rand() % 256;
        int idx2 = rand() % 256;
        if (idx1 == idx2) continue;

        uint8_t temp = current_remap[idx1];
        current_remap[idx1] = current_remap[idx2];
        current_remap[idx2] = temp;

        // 2. Evaluation Phase (Evaluate all files under new mapping)
        size_t new_sum = 0;
        for (int i = 0; i < file_count; i++) {
            for (size_t j = 0; j < files[i].file_size; j++) {
                remapped_buf[j] = current_remap[files[i].in_buf[j]];
            }
            files[i].temp_comp_size = compress_buffer(remapped_buf, files[i].file_size, out_buf, out_capacity);
            new_sum += files[i].temp_comp_size;
        }

        // 3. Metropolis Acceptance Criterion
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
            // Apply temp state to current state
            current_sum = new_sum;
            for (int i = 0; i < file_count; i++) {
                files[i].current_comp_size = files[i].temp_comp_size;
            }
            
            // Check against GLOBAL absolute minimum
            if (current_sum < best_sum) {
                best_sum = current_sum;
                
                fprintf(stderr, "/* NEW GLOBAL BEST: %zu bytes (Iter %llu, Temp %.2f) */\n", 
                        best_sum, iterations, temperature);
                
                // Report per-file metrics tracking regressions and improvements
                fprintf(stderr, "/* FILE DELTAS:\n");
                for (int i = 0; i < file_count; i++) {
                    long long delta = (long long)files[i].current_comp_size - (long long)files[i].best_comp_size;
                    
                    if (delta < 0) {
                        fprintf(stderr, "   [IMPROVED]  %s: %zu bytes (%lld bytes)\n", files[i].filename, files[i].current_comp_size, delta);
                    } else if (delta > 0) {
                        fprintf(stderr, "   [REGRESSED] %s: %zu bytes (+%lld bytes)\n", files[i].filename, files[i].current_comp_size, delta);
                    } else {
                        fprintf(stderr, "   [UNCHANGED] %s: %zu bytes\n", files[i].filename, files[i].current_comp_size);
                    }
                    
                    // Sync the best known size for this file
                    files[i].best_comp_size = files[i].current_comp_size;
                }
                fprintf(stderr, "*/\n");
                
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
