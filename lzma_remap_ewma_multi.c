#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <lzma.h>

/**
 * ==============================================================================
 * ALGORITHM OVERVIEW: EWMA-Driven Cohabitation Optimizer with Chunk Normalization
 * ==============================================================================
 * 
 * 0. Initialization & Optional State Randomization
 *    The algorithm begins with a predefined 'seed_remap'. If the '--reshuffle' 
 *    flag is provided, this initial seed is completely scrambled using a 
 *    Fisher-Yates shuffle across the entire 256-byte domain.
 * 
 * 1. The Global Objective (Multi-File Optimization)
 *    The objective function calculates the SUM of the compressed sizes of all 
 *    input files to guide the search toward a universal byte permutation.
 * 
 * 2. LZMA-Aligned Table Partitioning (The Invariant)
 *    The 256-element remap table is continuously partitioned into 2^lc equally 
 *    sized chunks. The algorithm enforces a strict normalization invariant:
 *      - Values inside each chunk are sorted in ascending order.
 *      - The chunks themselves are sorted by the average (sum) of their elements.
 * 
 * 3. EWMA "Badness" Cohabitation Predictor
 *    Tracks a 256x256 matrix representing the Exponentially Weighted Moving 
 *    Average (EWMA) "badness" of having two specific byte values reside in the 
 *    same chunk (0.0 = Favorable, 1.0 = Unfavorable).
 * 
 * 4. Variable Perturbation Magnitude (1-pair vs 2-pair swaps)
 *    To prevent getting trapped in local minima, the algorithm stochastically 
 *    decides whether to swap a single pair (2 elements) or two distinct pairs 
 *    (4 elements). Each pair is strictly selected from two *different* chunks, 
 *    though a double-swap might bridge the same two chunks twice.
 * 
 * 5. Greedy Acceptance & Balanced EWMA Feedback Loop (15-Neighbor Subsampling)
 *    Once a mutation is proposed and evaluated, the feedback loop updates exactly 
 *    15 randomly chosen neighbors for each swapped element. Expanding from 8 to 15 
 *    neighbors provides a much more robust gradient estimation of the search space.
 * ==============================================================================
 */

#define MAX_PICKS 15 // Increased neighborhood sampling size

/* 
 * REPLACE THIS BLOCK with the output from stderr to use the best mapping 
 * as the new seed for the next run. Initialized here to the Identity Mapping.
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
    size_t baseline_comp_size; 
    size_t initial_comp_size;   
    size_t current_comp_size;   
    size_t milestone_comp_size; 
    size_t temp_comp_size;      
} FileData;

// EWMA predictor matrix [byte_val1][byte_val2]
double ewma_badness[256][256];

/**
 * Algorithm: Aggressive Exponential Smoothing
 * -------------------------------------------
 * Set highly reactive (0.60). This ensures the badness matrix updates rapidly 
 * based on recent accept/reject evaluations, preventing the algorithm from 
 * stagnating on deeply entrenched but outdated historical data.
 */
const double EWMA_ALPHA = 0.60; 

void print_help(const char *prog_name) {
    printf("Usage: %s [OPTIONS] <input_file_1> [input_file_2 ...]\n\n", prog_name);
    printf("Options:\n");
    printf("  --help          Show this help message and exit.\n");
    printf("  --timeout=SEC   Set timeout in seconds (default: 600).\n");
    printf("  --dict=KB       Set LZMA dictionary size in kilobytes (default: 4).\n");
    printf("  --lc=BITS       Set LZMA literal context bits (1-4, default: 3).\n");
    printf("  --lp=BITS       Set LZMA literal position bits (0-4, default: 0).\n");
    printf("  --pb=BITS       Set LZMA position bits (0-4, default: 2).\n");
    printf("  --reshuffle     Randomly shuffle the initial mapping before optimization.\n");
}

// Helper: Select distinct random neighbor indices from within a chunk
void get_random_neighbors(int chunk_size, int elem_idx_in_chunk, int num_picks, int *picks) {
    int count = 0;
    while (count < num_picks) {
        int cand = rand() % chunk_size;
        
        // Cannot pick itself
        if (cand == elem_idx_in_chunk) continue;
        
        // Ensure distinct choices
        int duplicate = 0;
        for (int k = 0; k < count; k++) {
            if (picks[k] == cand) {
                duplicate = 1;
                break;
            }
        }
        
        if (!duplicate) {
            picks[count++] = cand;
        }
    }
}

// Enforces structural invariant on the remap table
void normalize_remap(uint8_t *remap, int lc) {
    int num_chunks = 1 << lc;
    int chunk_size = 256 / num_chunks;
    
    // 1. Sort the elements inside each chunk (Insertion Sort)
    for (int c = 0; c < num_chunks; c++) {
        int start = c * chunk_size;
        for (int i = 1; i < chunk_size; i++) {
            uint8_t key = remap[start + i];
            int j = i - 1;
            while (j >= 0 && remap[start + j] > key) {
                remap[start + j + 1] = remap[start + j];
                j = j - 1;
            }
            remap[start + j + 1] = key;
        }
    }
    
    // 2. Sort the chunks themselves by the sum of their elements
    typedef struct {
        int sum;
        uint8_t first;
        uint8_t elements[256]; 
    } ChunkData;
    
    ChunkData chunks[256]; 
    for (int c = 0; c < num_chunks; c++) {
        chunks[c].sum = 0;
        int start = c * chunk_size;
        chunks[c].first = remap[start];
        for (int i = 0; i < chunk_size; i++) {
            chunks[c].elements[i] = remap[start + i];
            chunks[c].sum += remap[start + i];
        }
    }
    
    // Bubble sort for sorting chunks (small N, stable enough)
    for (int i = 0; i < num_chunks - 1; i++) {
        for (int j = i + 1; j < num_chunks; j++) {
            int swap = 0;
            if (chunks[i].sum > chunks[j].sum) {
                swap = 1;
            } else if (chunks[i].sum == chunks[j].sum && chunks[i].first > chunks[j].first) {
                swap = 1;
            }
            if (swap) {
                ChunkData temp = chunks[i];
                chunks[i] = chunks[j];
                chunks[j] = temp;
            }
        }
    }
    
    // 3. Write back the normalized state to the remap table
    for (int c = 0; c < num_chunks; c++) {
        int start = c * chunk_size;
        for (int i = 0; i < chunk_size; i++) {
            remap[start + i] = chunks[c].elements[i];
        }
    }
}

size_t compress_buffer(const uint8_t *in_buf, size_t in_len, uint8_t *out_buf, size_t out_capacity, int dict_param_kb, int lc_param, int lp_param, int pb_param) {
    lzma_options_lzma opt;
    if (lzma_lzma_preset(&opt, 0)) {
        fprintf(stderr, "Error: Failed to set LZMA preset.\n");
        exit(EXIT_FAILURE);
    }
    
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

void print_byte_literal(FILE *out, uint8_t b) {
    if (b >= 32 && b <= 126) {
        if (b == '\'') fprintf(out, "'\\''"); 
        else if (b == '\\') fprintf(out, "'\\\\'"); 
        else fprintf(out, "'%c' ", b); 
    } else {
        fprintf(out, "0x%02x", b); 
    }
}

void print_remap_table_as_source(const uint8_t *remap) {
    fprintf(stderr, "unsigned char seed_remap[256] = {\n    ");
    for (int i = 0; i < 256; i++) {
        print_byte_literal(stderr, remap[i]);
        if (i < 255) fprintf(stderr, ", ");
        if ((i + 1) % 16 == 0 && i < 255) fprintf(stderr, "\n    ");
    }
    fprintf(stderr, "\n};\n\n");
    fflush(stderr); 
}

double random_double() {
    return (double)rand() / (double)RAND_MAX;
}

/**
 * Algorithm: Stochastic Weighted Selection (Roulette Wheel Filter)
 * -----------------------------------------------------------------
 * Safely executes roulette-wheel selection over the calculated EWMA 
 * 'desirability' weights. Includes strict exclusion parameters to guarantee:
 *   1. We can force selection from a distinct chunk (exclude_chunk).
 *   2. We can prevent overlaps/collisions with already selected elements 
 *      during multi-pair swaps (exclude_idx1 ... exclude_idx3).
 */
int roulette_select(double *weights, int chunk_size, int exclude_chunk, int exclude_idx1, int exclude_idx2, int exclude_idx3) {
    double total_weight = 0.0;
    for (int i = 0; i < 256; i++) {
        if (exclude_chunk >= 0 && (i / chunk_size) == exclude_chunk) continue;
        if (i == exclude_idx1 || i == exclude_idx2 || i == exclude_idx3) continue;
        total_weight += weights[i];
    }
    
    if (total_weight <= 0.0) return -1; 
    
    double r = random_double() * total_weight;
    double accum = 0.0;
    for (int i = 0; i < 256; i++) {
        if (exclude_chunk >= 0 && (i / chunk_size) == exclude_chunk) continue;
        if (i == exclude_idx1 || i == exclude_idx2 || i == exclude_idx3) continue;
        
        accum += weights[i];
        if (accum >= r) return i;
    }
    return -1;
}

int main(int argc, char **argv) {
    srand((unsigned int)time(NULL));

    int timeout = 600; 
    int dict_param_kb = 4;
    int lc_param = 3; 
    int lp_param = 0;
    int pb_param = 2;
    int do_reshuffle = 0;
    
    const char **filenames = malloc(argc * sizeof(char*));
    int file_count = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_help(argv[0]);
            free(filenames);
            return EXIT_SUCCESS;
        } else if (strncmp(argv[i], "--timeout=", 10) == 0) {
            timeout = atoi(argv[i] + 10);
        } else if (strncmp(argv[i], "--dict=", 7) == 0) {
            dict_param_kb = atoi(argv[i] + 7);
        } else if (strncmp(argv[i], "--lc=", 5) == 0) {
            lc_param = atoi(argv[i] + 5);
        } else if (strncmp(argv[i], "--lp=", 5) == 0) {
            lp_param = atoi(argv[i] + 5);
        } else if (strncmp(argv[i], "--pb=", 5) == 0) {
            pb_param = atoi(argv[i] + 5);
        } else if (strcmp(argv[i], "--reshuffle") == 0) {
            do_reshuffle = 1;
        } else {
            filenames[file_count++] = argv[i];
        }
    }

    if (file_count == 0) {
        fprintf(stderr, "Error: No input files specified.\n");
        free(filenames);
        return EXIT_FAILURE;
    }

    if (lc_param < 1) {
        fprintf(stderr, "Error: lc must be >= 1 for chunk-based cohabitation mutation to work.\n");
        free(filenames);
        return EXIT_FAILURE;
    }

    int num_chunks = 1 << lc_param;
    int chunk_size = 256 / num_chunks;
    
    // Bounds Check: Request MAX_PICKS (15) unless chunk is too small
    int num_picks = (chunk_size - 1 < MAX_PICKS) ? (chunk_size - 1) : MAX_PICKS;

    FileData *files = malloc(file_count * sizeof(FileData));
    size_t max_file_size = 0;
    int max_filename_len = 0;

    for (int i = 0; i < file_count; i++) {
        files[i].filename = filenames[i];
        
        int fn_len = (int)strlen(filenames[i]);
        if (fn_len > max_filename_len) max_filename_len = fn_len;

        FILE *f = fopen(filenames[i], "rb");
        if (!f) return EXIT_FAILURE;

        fseek(f, 0, SEEK_END);
        files[i].file_size = (size_t)ftell(f);
        fseek(f, 0, SEEK_SET);

        if (files[i].file_size > max_file_size) max_file_size = files[i].file_size;

        files[i].in_buf = malloc(files[i].file_size);
        if (fread(files[i].in_buf, 1, files[i].file_size, f) != files[i].file_size) return EXIT_FAILURE;
        fclose(f);
    }

    uint8_t *remapped_buf = malloc(max_file_size);
    size_t out_capacity = lzma_stream_buffer_bound(max_file_size);
    uint8_t *out_buf = malloc(out_capacity);

    // Initialize EWMA Matrix
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            ewma_badness[i][j] = 0.5;
        }
    }

    // 1. Evaluate True Baseline
    size_t baseline_sum = 0;
    for (int i = 0; i < file_count; i++) {
        for (size_t j = 0; j < files[i].file_size; j++) remapped_buf[j] = files[i].in_buf[j];
        size_t s = compress_buffer(remapped_buf, files[i].file_size, out_buf, out_capacity, dict_param_kb, lc_param, lp_param, pb_param);
        files[i].baseline_comp_size = s;
        baseline_sum += s;
    }
    fprintf(stderr, "Original Identity Mapping Total Size: %zu bytes\n\n", baseline_sum);

    // 2. Evaluate User Seed
    uint8_t current_remap[256];
    memcpy(current_remap, seed_remap, 256);

    if (do_reshuffle) {
        for (int i = 255; i > 0; i--) {
            int j = rand() % (i + 1);
            uint8_t temp = current_remap[i];
            current_remap[i] = current_remap[j];
            current_remap[j] = temp;
        }
    }

    normalize_remap(current_remap, lc_param); 

    size_t initial_sum = 0;
    for (int i = 0; i < file_count; i++) {
        for (size_t j = 0; j < files[i].file_size; j++) remapped_buf[j] = current_remap[files[i].in_buf[j]];
        size_t s = compress_buffer(remapped_buf, files[i].file_size, out_buf, out_capacity, dict_param_kb, lc_param, lp_param, pb_param);
        files[i].initial_comp_size = s;
        files[i].current_comp_size = s;
        files[i].milestone_comp_size = s; 
        initial_sum += s;
    }
    
    size_t current_sum = initial_sum;
    size_t best_sum = initial_sum;
    fprintf(stderr, "Normalized Starting Seed Total Size: %zu bytes\n\n", best_sum);

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

        // 3. Evaluate Displacement Desirability Weights (Using 15 Random Neighbors)
        double weights[256];
        double total_weight = 0.0;
        
        for (int i = 0; i < 256; i++) {
            int chunk_idx = i / chunk_size;
            int start = chunk_idx * chunk_size;
            double badness_sum = 0.0;
            uint8_t elem = current_remap[i];
            
            int picks[MAX_PICKS];
            get_random_neighbors(chunk_size, i - start, num_picks, picks);
            
            for (int k = 0; k < num_picks; k++) {
                uint8_t neighbor = current_remap[start + picks[k]];
                badness_sum += ewma_badness[elem][neighbor];
            }

            /**
             * Algorithm: Non-linear Weight Scaling (Polynomial Penalty)
             * ---------------------------------------------------------
             * To make higher badness result in *noticeably larger* random 
             * weights, we apply a cubic transformation. A linear scale allows 
             * for a 2x selection probability between a badness of 10 and 5. 
             * A cubic scale (10^3 = 1000 vs 5^3 = 125) drastically magnifies 
             * the gap to an 8x selection probability, forcing the roulette wheel 
             * to mercilessly evict the worst cohabiting byte values.
             */
            weights[i] = (badness_sum * badness_sum * badness_sum) + 0.01; 
            total_weight += weights[i];
        }

        // 4. Algorithm: Multi-Pair Extraction
        // -----------------------------------
        // Standard Operation: Swap exactly 1 pair (idx1, idx2) spanning different chunks.
        // Probabilistic Operation (30% chance): Swap 2 distinct pairs (idx1/idx2 AND idx3/idx4) 
        // to escape local minima via a larger spatial jump.
        int idx1 = roulette_select(weights, chunk_size, -1, -1, -1, -1);
        int idx2 = roulette_select(weights, chunk_size, (idx1 / chunk_size), idx1, -1, -1);
        
        int do_double_swap = 0;
        int idx3 = -1, idx4 = -1;
        uint8_t e3 = 0, e4 = 0;

        if (random_double() < 0.30) {
            idx3 = roulette_select(weights, chunk_size, -1, idx1, idx2, -1);
            if (idx3 != -1) {
                // Ensure idx4 is in a different chunk than idx3, and completely distinct from 1, 2, and 3
                idx4 = roulette_select(weights, chunk_size, (idx3 / chunk_size), idx1, idx2, idx3);
            }
            if (idx3 != -1 && idx4 != -1) {
                do_double_swap = 1;
            }
        }

        // 5. Propose the Mutation
        uint8_t proposed_remap[256];
        memcpy(proposed_remap, current_remap, 256);
        
        uint8_t e1 = proposed_remap[idx1];
        uint8_t e2 = proposed_remap[idx2];
        proposed_remap[idx1] = e2;
        proposed_remap[idx2] = e1;
        
        if (do_double_swap) {
            e3 = proposed_remap[idx3];
            e4 = proposed_remap[idx4];
            proposed_remap[idx3] = e4;
            proposed_remap[idx4] = e3;
        }
        
        normalize_remap(proposed_remap, lc_param);

        // 6. Evaluation Phase
        size_t new_sum = 0;
        for (int i = 0; i < file_count; i++) {
            for (size_t j = 0; j < files[i].file_size; j++) remapped_buf[j] = proposed_remap[files[i].in_buf[j]];
            files[i].temp_comp_size = compress_buffer(remapped_buf, files[i].file_size, out_buf, out_capacity, dict_param_kb, lc_param, lp_param, pb_param);
            new_sum += files[i].temp_comp_size;
        }

        // 7. Capture Original and Proposed Chunk Boundaries for Balanced EWMA Updates
        
        // PAIR 1 Boundaries
        int old_c1 = idx1 / chunk_size;
        int old_c2 = idx2 / chunk_size;
        int old_start1 = old_c1 * chunk_size;
        int old_start2 = old_c2 * chunk_size;
        int new_idx1 = -1, new_idx2 = -1, new_idx3 = -1, new_idx4 = -1;

        for (int i = 0; i < 256; i++) {
            if (proposed_remap[i] == e1) new_idx1 = i;
            if (proposed_remap[i] == e2) new_idx2 = i;
            if (do_double_swap) {
                if (proposed_remap[i] == e3) new_idx3 = i;
                if (proposed_remap[i] == e4) new_idx4 = i;
            }
        }
        
        int new_c1 = new_idx1 / chunk_size;
        int new_c2 = new_idx2 / chunk_size;
        int new_start1 = new_c1 * chunk_size;
        int new_start2 = new_c2 * chunk_size;

        int old_picks1[MAX_PICKS], old_picks2[MAX_PICKS], new_picks1[MAX_PICKS], new_picks2[MAX_PICKS];
        get_random_neighbors(chunk_size, idx1 - old_start1, num_picks, old_picks1);
        get_random_neighbors(chunk_size, idx2 - old_start2, num_picks, old_picks2);
        get_random_neighbors(chunk_size, new_idx1 - new_start1, num_picks, new_picks1);
        get_random_neighbors(chunk_size, new_idx2 - new_start2, num_picks, new_picks2);

        // PAIR 2 Boundaries (Conditionally generated)
        int old_c3 = -1, old_c4 = -1, new_c3 = -1, new_c4 = -1;
        int old_start3 = 0, old_start4 = 0, new_start3 = 0, new_start4 = 0;
        int old_picks3[MAX_PICKS], old_picks4[MAX_PICKS], new_picks3[MAX_PICKS], new_picks4[MAX_PICKS];
        
        if (do_double_swap) {
            old_c3 = idx3 / chunk_size;
            old_c4 = idx4 / chunk_size;
            new_c3 = new_idx3 / chunk_size;
            new_c4 = new_idx4 / chunk_size;
            
            old_start3 = old_c3 * chunk_size;
            old_start4 = old_c4 * chunk_size;
            new_start3 = new_c3 * chunk_size;
            new_start4 = new_c4 * chunk_size;
            
            get_random_neighbors(chunk_size, idx3 - old_start3, num_picks, old_picks3);
            get_random_neighbors(chunk_size, idx4 - old_start4, num_picks, old_picks4);
            get_random_neighbors(chunk_size, new_idx3 - new_start3, num_picks, new_picks3);
            get_random_neighbors(chunk_size, new_idx4 - new_start4, num_picks, new_picks4);
        }

        // 8. Acceptance Criterion & Balanced Push-Pull EWMA Weight Updating
        if (new_sum < current_sum) {
            // ACCEPT: The swap improves compressibility
            // -----------------------------------------------------------
            // OLD neighbors push -> Unfavorable (1.0) (leaving them helped)
            // NEW neighbors push -> Favorable (0.0)   (joining them helped)
            
            // --- PAIR 1 EWMA UPDATES ---
            for (int k = 0; k < num_picks; k++) {
                uint8_t old_n = current_remap[old_start1 + old_picks1[k]];
                ewma_badness[e1][old_n] = (1.0 - EWMA_ALPHA) * ewma_badness[e1][old_n] + EWMA_ALPHA * 1.0;
                ewma_badness[old_n][e1] = ewma_badness[e1][old_n]; 

                uint8_t old_n2 = current_remap[old_start2 + old_picks2[k]];
                ewma_badness[e2][old_n2] = (1.0 - EWMA_ALPHA) * ewma_badness[e2][old_n2] + EWMA_ALPHA * 1.0;
                ewma_badness[old_n2][e2] = ewma_badness[e2][old_n2];

                uint8_t new_n = proposed_remap[new_start1 + new_picks1[k]];
                ewma_badness[e1][new_n] = (1.0 - EWMA_ALPHA) * ewma_badness[e1][new_n] + EWMA_ALPHA * 0.0;
                ewma_badness[new_n][e1] = ewma_badness[e1][new_n]; 

                uint8_t new_n2 = proposed_remap[new_start2 + new_picks2[k]];
                ewma_badness[e2][new_n2] = (1.0 - EWMA_ALPHA) * ewma_badness[e2][new_n2] + EWMA_ALPHA * 0.0;
                ewma_badness[new_n2][e2] = ewma_badness[e2][new_n2];
            }

            // --- PAIR 2 EWMA UPDATES ---
            if (do_double_swap) {
                for (int k = 0; k < num_picks; k++) {
                    uint8_t old_n3 = current_remap[old_start3 + old_picks3[k]];
                    ewma_badness[e3][old_n3] = (1.0 - EWMA_ALPHA) * ewma_badness[e3][old_n3] + EWMA_ALPHA * 1.0;
                    ewma_badness[old_n3][e3] = ewma_badness[e3][old_n3]; 

                    uint8_t old_n4 = current_remap[old_start4 + old_picks4[k]];
                    ewma_badness[e4][old_n4] = (1.0 - EWMA_ALPHA) * ewma_badness[e4][old_n4] + EWMA_ALPHA * 1.0;
                    ewma_badness[old_n4][e4] = ewma_badness[e4][old_n4];

                    uint8_t new_n3 = proposed_remap[new_start3 + new_picks3[k]];
                    ewma_badness[e3][new_n3] = (1.0 - EWMA_ALPHA) * ewma_badness[e3][new_n3] + EWMA_ALPHA * 0.0;
                    ewma_badness[new_n3][e3] = ewma_badness[e3][new_n3]; 

                    uint8_t new_n4 = proposed_remap[new_start4 + new_picks4[k]];
                    ewma_badness[e4][new_n4] = (1.0 - EWMA_ALPHA) * ewma_badness[e4][new_n4] + EWMA_ALPHA * 0.0;
                    ewma_badness[new_n4][e4] = ewma_badness[e4][new_n4];
                }
            }

            memcpy(current_remap, proposed_remap, 256);
            current_sum = new_sum;
            for (int i = 0; i < file_count; i++) files[i].current_comp_size = files[i].temp_comp_size;

            if (new_sum < best_sum) {
                long long total_delta_base = (long long)new_sum - (long long)baseline_sum;
                double total_pct_base = ((double)total_delta_base / (double)baseline_sum) * 100.0;
                
                long long total_delta_init = (long long)new_sum - (long long)initial_sum;
                double total_pct_init = ((double)total_delta_init / (double)initial_sum) * 100.0;
                
                long long total_delta_prev = (long long)new_sum - (long long)best_sum;
                double total_pct_prev = ((double)total_delta_prev / (double)best_sum) * 100.0;

                fprintf(stderr, "/* NEW GLOBAL BEST: %zu bytes (Iter %llu, dict:%dK/lc:%d/lp:%d/pb:%d) */\n", 
                        new_sum, iterations, dict_param_kb, lc_param, lp_param, pb_param);
                
                // ---------------------------------------------------------------------------------
                // Algorithm Debugging: Intermediate Mutation & Probability Report
                // ---------------------------------------------------------------------------------
                // Outputs the exact structural displacement of the values alongside a 16x16 grid 
                // tracking the overall probability landscape mapped by the EWMA matrix.
                fprintf(stderr, "/* MUTATION DETAILS:\n");
                fprintf(stderr, "   Type: %s\n", do_double_swap ? "Double Swap (4 unique values)" : "Single Swap (2 unique values)");
                
                fprintf(stderr, "   Pair 1:\n");
                fprintf(stderr, "     - Value 0x%02x moved from Chunk %d to %d (Weight: %.4f, Base Prob: %5.2f%%)\n", 
                        e1, old_c1, new_c1, weights[idx1], (weights[idx1]/total_weight)*100.0);
                fprintf(stderr, "     - Value 0x%02x moved from Chunk %d to %d (Weight: %.4f, Base Prob: %5.2f%%)\n", 
                        e2, old_c2, new_c2, weights[idx2], (weights[idx2]/total_weight)*100.0);
                
                if (do_double_swap) {
                    fprintf(stderr, "   Pair 2:\n");
                    fprintf(stderr, "     - Value 0x%02x moved from Chunk %d to %d (Weight: %.4f, Base Prob: %5.2f%%)\n", 
                            e3, old_c3, new_c3, weights[idx3], (weights[idx3]/total_weight)*100.0);
                    fprintf(stderr, "     - Value 0x%02x moved from Chunk %d to %d (Weight: %.4f, Base Prob: %5.2f%%)\n", 
                            e4, old_c4, new_c4, weights[idx4], (weights[idx4]/total_weight)*100.0);
                }

                fprintf(stderr, "\n   BASE SELECTION PROBABILITIES (%%) [16x16 Mapping Array]:\n");
                for (int r = 0; r < 16; r++) {
                    fprintf(stderr, "   ");
                    for (int c = 0; c < 16; c++) {
                        fprintf(stderr, "%5.2f ", (weights[r*16+c] / total_weight) * 100.0);
                    }
                    fprintf(stderr, "\n");
                }
                fprintf(stderr, "*/\n");
                
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
                    
                    files[i].milestone_comp_size = files[i].temp_comp_size;
                }
                fprintf(stderr, "*/\n");
                
                best_sum = new_sum;
                print_remap_table_as_source(current_remap);
            }
        } else {
            // REJECT: Compressibility degraded
            // -----------------------------------------------------------
            // NEW neighbors push -> Unfavorable (1.0) (joining them hurt)
            // OLD neighbors push -> Favorable (0.0)   (staying was better)

            // --- PAIR 1 EWMA UPDATES ---
            for (int k = 0; k < num_picks; k++) {
                uint8_t new_n = proposed_remap[new_start1 + new_picks1[k]];
                ewma_badness[e1][new_n] = (1.0 - EWMA_ALPHA) * ewma_badness[e1][new_n] + EWMA_ALPHA * 1.0;
                ewma_badness[new_n][e1] = ewma_badness[e1][new_n];

                uint8_t new_n2 = proposed_remap[new_start2 + new_picks2[k]];
                ewma_badness[e2][new_n2] = (1.0 - EWMA_ALPHA) * ewma_badness[e2][new_n2] + EWMA_ALPHA * 1.0;
                ewma_badness[new_n2][e2] = ewma_badness[e2][new_n2];

                uint8_t old_n = current_remap[old_start1 + old_picks1[k]];
                ewma_badness[e1][old_n] = (1.0 - EWMA_ALPHA) * ewma_badness[e1][old_n] + EWMA_ALPHA * 0.0;
                ewma_badness[old_n][e1] = ewma_badness[e1][old_n];

                uint8_t old_n2 = current_remap[old_start2 + old_picks2[k]];
                ewma_badness[e2][old_n2] = (1.0 - EWMA_ALPHA) * ewma_badness[e2][old_n2] + EWMA_ALPHA * 0.0;
                ewma_badness[old_n2][e2] = ewma_badness[e2][old_n2];
            }

            // --- PAIR 2 EWMA UPDATES ---
            if (do_double_swap) {
                for (int k = 0; k < num_picks; k++) {
                    uint8_t new_n3 = proposed_remap[new_start3 + new_picks3[k]];
                    ewma_badness[e3][new_n3] = (1.0 - EWMA_ALPHA) * ewma_badness[e3][new_n3] + EWMA_ALPHA * 1.0;
                    ewma_badness[new_n3][e3] = ewma_badness[e3][new_n3];

                    uint8_t new_n4 = proposed_remap[new_start4 + new_picks4[k]];
                    ewma_badness[e4][new_n4] = (1.0 - EWMA_ALPHA) * ewma_badness[e4][new_n4] + EWMA_ALPHA * 1.0;
                    ewma_badness[new_n4][e4] = ewma_badness[e4][new_n4];

                    uint8_t old_n3 = current_remap[old_start3 + old_picks3[k]];
                    ewma_badness[e3][old_n3] = (1.0 - EWMA_ALPHA) * ewma_badness[e3][old_n3] + EWMA_ALPHA * 0.0;
                    ewma_badness[old_n3][e3] = ewma_badness[e3][old_n3];

                    uint8_t old_n4 = current_remap[old_start4 + old_picks4[k]];
                    ewma_badness[e4][old_n4] = (1.0 - EWMA_ALPHA) * ewma_badness[e4][old_n4] + EWMA_ALPHA * 0.0;
                    ewma_badness[old_n4][e4] = ewma_badness[e4][old_n4];
                }
            }
        }
    }

    fprintf(stderr, "/* Search finished. Iterations: %llu. Final Best Sum: %zu bytes */\n", iterations, best_sum);

    for (int i = 0; i < file_count; i++) free(files[i].in_buf);
    free(files);
    free(filenames);
    free(remapped_buf); 
    free(out_buf);
    
    return EXIT_SUCCESS;
}
