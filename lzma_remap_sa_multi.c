#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <lzma.h>

/**
 * ==============================================================================
 * ALGORITHM OVERVIEW: Chunk-Partitioned Greedy Local Search
 * ==============================================================================
 * 
 * 1. The Global Objective (Multi-File Optimization)
 *    The objective function calculates the SUM of the compressed sizes of all 
 *    input files. This guides the search toward a universal byte permutation 
 *    that minimizes entropy across the entire dataset.
 * 
 * 2. Tunable LZMA Markov Chain Modeling
 *    - dict (Dictionary Size): Configurable via '--dict=' in KB (default: 4).
 *      Defaults to the 4KB minimum to drastically increase evaluation speed.
 *    - lc (Literal Context): Configurable via '--lc=' (default: 3). 
 * 
 * 3. Chunk Partitioning and Strict Normalisation
 *    Depending on the `lc` value, the 256-byte table is partitioned into 
 *    pow(2, lc) equally sized chunks. To ensure search space structural consistency:
 *    - Elements inside each chunk are always strictly ascending.
 *    - The chunks themselves are sorted by the mathematical average (sum).
 * 
 * 4. Progressive Heuristic Pre-Pass Partitioning
 *    - Optional feature triggered by `--heuristic[=MAX_TESTS]`.
 *    - Automatically starts by dividing chunks into M=2 subchunks.
 *    - Exhaustively generates valid combinatorial unlabeled subsets.
 *    - If an improvement over the identity mapping is found, it recursively
 *      scales to M=4, M=8, etc., attempting finer structural permutations.
 *    - Duplicate Rejection: When M >= 4, the algorithm mathematically detects
 *      if a generated partition is structurally equivalent to one already tested 
 *      in the M/2 pass (where all paired adjacent elements fall into the same bin)
 *      and skips its evaluation to preserve testing allowance.
 * 
 * 5. Exhaustive Mutation Strategy
 *    - The algorithm picks 3 distinct chunks randomly (or 2 if lc=1).
 *    - Inside each picked chunk, it picks 1 random element.
 *    - It rapidly evaluates all permutations of these elements and greedily applies
 *      improvements.
 * ==============================================================================
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

// Context purely to support the recursive heuristic combination generator
typedef struct {
    int M;
    int C;
    int S;
    unsigned long long heur_tested;
    unsigned long long heur_limit;
    size_t best_heur_sum;
    uint8_t best_heur_remap[256];
    int file_count;
    FileData *files;
    uint8_t *remapped_buf;
    uint8_t *out_buf;
    size_t out_capacity;
    int dict_param_kb;
    int lc_param;
    int lp_param;
    int pb_param;
    time_t start_time;
    int timeout;
} HeurContext;

void print_help(const char *prog_name) {
    printf("Usage: %s [OPTIONS] <input_file_1> [input_file_2 ...]\n\n", prog_name);
    printf("Options:\n");
    printf("  --help            Show this help message and exit.\n");
    printf("  --timeout=SEC     Set timeout in seconds (default: 600).\n");
    printf("  --reshuffle       Randomize the initial remap table before starting.\n");
    printf("  --heuristic[=N]   Run exhaustive subchunk scaling pre-pass (default N=128).\n");
    printf("  --dict=KB         Set LZMA dictionary size in kilobytes (default: 4).\n");
    printf("  --lc=BITS         Set LZMA literal context bits (1-4, default: 3).\n");
    printf("  --lp=BITS         Set LZMA literal position bits (0-4, default: 0).\n");
    printf("  --pb=BITS         Set LZMA position bits (0-4, default: 2).\n");
    printf("\n");
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

int cmp_uint8(const void *a, const void *b) {
    return (*(const uint8_t *)a) - (*(const uint8_t *)b);
}

typedef struct {
    int sum;
    uint8_t first_val;
    uint8_t data[256];
} Chunk;

int cmp_chunk(const void *a, const void *b) {
    const Chunk *ca = (const Chunk *)a;
    const Chunk *cb = (const Chunk *)b;
    if (ca->sum < cb->sum) return -1;
    if (ca->sum > cb->sum) return 1;
    if (ca->first_val < cb->first_val) return -1;
    if (ca->first_val > cb->first_val) return 1;
    return 0;
}

void normalize_remap(uint8_t *remap, int lc) {
    int num_chunks = 1 << lc;
    int chunk_size = 256 / num_chunks;
    
    Chunk chunks[16]; 
    
    for (int c = 0; c < num_chunks; c++) {
        chunks[c].sum = 0;
        for (int i = 0; i < chunk_size; i++) {
            chunks[c].data[i] = remap[c * chunk_size + i];
        }
        qsort(chunks[c].data, chunk_size, sizeof(uint8_t), cmp_uint8);
        
        for (int i = 0; i < chunk_size; i++) {
            chunks[c].sum += chunks[c].data[i];
        }
        chunks[c].first_val = chunks[c].data[0];
    }
    
    qsort(chunks, num_chunks, sizeof(Chunk), cmp_chunk);
    
    for (int c = 0; c < num_chunks; c++) {
        for (int i = 0; i < chunk_size; i++) {
            remap[c * chunk_size + i] = chunks[c].data[i];
        }
    }
}

// Backtracking algorithm generates unlabeled subsets, guaranteeing mathematically distinct subchunk partitions 
void generate_partitions(int element_idx, int num_active_chunks, int *counts, int *assignment, HeurContext *ctx) {
    if (ctx->heur_tested >= ctx->heur_limit) return;
    time_t now = time(NULL);
    if (difftime(now, ctx->start_time) >= ctx->timeout) return;
    
    if (element_idx == ctx->C * ctx->M) {
        
        // Mathematical duplicate rejection:
        // A partition at step M >= 4 is logically identical to a partition evaluated
        // in the M/2 pass if every adjacent pair of subchunks falls into the same bin.
        if (ctx->M > 2) {
            int is_duplicate = 1;
            for (int i = 0; i < ctx->C * ctx->M; i += 2) {
                if (assignment[i] != assignment[i + 1]) {
                    is_duplicate = 0;
                    break;
                }
            }
            if (is_duplicate) return; 
        }

        uint8_t heur_remap[256];
        int fill[16] = {0};
        
        // Assemble subchunks mapped to their physical representation based on the partition logic
        for (int i = 0; i < ctx->C * ctx->M; i++) {
            int c = assignment[i];
            int offset = fill[c] * ctx->S;
            for (int j = 0; j < ctx->S; j++) {
                heur_remap[c * (ctx->M * ctx->S) + offset + j] = i * ctx->S + j; 
            }
            fill[c]++;
        }
        
        normalize_remap(heur_remap, ctx->lc_param);
        
        size_t test_sum = 0;
        int aborted = 0;
        for (int i = 0; i < ctx->file_count; i++) {
            for (size_t j = 0; j < ctx->files[i].file_size; j++) {
                ctx->remapped_buf[j] = heur_remap[ctx->files[i].in_buf[j]];
            }
            size_t s = compress_buffer(ctx->remapped_buf, ctx->files[i].file_size, ctx->out_buf, ctx->out_capacity, ctx->dict_param_kb, ctx->lc_param, ctx->lp_param, ctx->pb_param);
            test_sum += s;
            if (test_sum >= ctx->best_heur_sum) {
                aborted = 1;
                break;
            }
        }
        
        if (!aborted && test_sum < ctx->best_heur_sum) {
            ctx->best_heur_sum = test_sum;
            memcpy(ctx->best_heur_remap, heur_remap, 256);
        }
        ctx->heur_tested++;
        return;
    }
    
    // Assign to existing unlabelled bin if it has capacity
    for (int c = 0; c < num_active_chunks; c++) {
        if (counts[c] < ctx->M) {
            counts[c]++;
            assignment[element_idx] = c;
            generate_partitions(element_idx + 1, num_active_chunks, counts, assignment, ctx);
            counts[c]--;
            if (ctx->heur_tested >= ctx->heur_limit) return;
        }
    }
    // Allocate to a completely new bin
    if (num_active_chunks < ctx->C) {
        counts[num_active_chunks]++;
        assignment[element_idx] = num_active_chunks;
        generate_partitions(element_idx + 1, num_active_chunks + 1, counts, assignment, ctx);
        counts[num_active_chunks]--;
    }
}

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
    int dict_param_kb = 4; 
    int lc_param = 3; 
    int lp_param = 0;
    int pb_param = 2;
    int do_reshuffle = 0;
    
    int do_heuristic = 0;
    int heur_limit = 128;
    
    const char **filenames = malloc(argc * sizeof(char*));
    int file_count = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_help(argv[0]);
            free(filenames);
            return EXIT_SUCCESS;
        } else if (strncmp(argv[i], "--timeout=", 10) == 0) {
            timeout = atoi(argv[i] + 10);
        } else if (strcmp(argv[i], "--reshuffle") == 0) {
            do_reshuffle = 1;
        } else if (strcmp(argv[i], "--heuristic") == 0) {
            do_heuristic = 1;
        } else if (strncmp(argv[i], "--heuristic=", 12) == 0) {
            do_heuristic = 1;
            heur_limit = atoi(argv[i] + 12);
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

    if (lc_param == 0) {
        fprintf(stderr, "Error: lc=0 results in 1 chunk effectively forcing a rigid identity mapping.\n");
        free(filenames);
        return EXIT_FAILURE;
    }

    srand((unsigned int)time(NULL));

    FileData *files = malloc(file_count * sizeof(FileData));
    size_t max_file_size = 0;
    int max_filename_len = 0;

    for (int i = 0; i < file_count; i++) {
        files[i].filename = filenames[i];
        
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

        if (files[i].file_size > max_file_size) max_file_size = files[i].file_size;

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

    time_t start_time = time(NULL);

    uint8_t current_remap[256];
    memcpy(current_remap, seed_remap, 256);

    // 2. Pre-Configuration Branch (Heuristic, Reshuffle, or Base Seed)
    if (do_heuristic) {
        int C = 1 << lc_param;
        
        fprintf(stderr, "Heuristic Pre-pass Enabled:\n");
        fprintf(stderr, "  Total chunks (C):        %d\n", C);
        fprintf(stderr, "  Testing limit:           %d arrangements\n\n", heur_limit);
        
        HeurContext ctx;
        ctx.C = C;
        ctx.heur_tested = 0;
        ctx.heur_limit = heur_limit;
        ctx.best_heur_sum = baseline_sum; // Start against true identity
        for (int i = 0; i < 256; i++) ctx.best_heur_remap[i] = i; 
        
        ctx.file_count = file_count;
        ctx.files = files;
        ctx.remapped_buf = remapped_buf;
        ctx.out_buf = out_buf;
        ctx.out_capacity = out_capacity;
        ctx.dict_param_kb = dict_param_kb;
        ctx.lc_param = lc_param;
        ctx.lp_param = lp_param;
        ctx.pb_param = pb_param;
        ctx.start_time = start_time;
        ctx.timeout = timeout;
        
        int M = 2;
        while (M * C <= 256 && ctx.heur_tested < ctx.heur_limit) {
            ctx.M = M;
            ctx.S = 256 / (C * M);
            fprintf(stderr, "  Trying M=%d (Subchunk size: %d bytes)...\n", M, ctx.S);
            
            int counts[16] = {0};
            int assignment[256] = {0};
            
            size_t before_m_sum = ctx.best_heur_sum;
            
            generate_partitions(0, 0, counts, assignment, &ctx);
            
            // Progressive Scaling: Continue searching finer granularities if improvement occurs
            if (ctx.best_heur_sum < before_m_sum) {
                fprintf(stderr, "    -> Improved compression to %zu bytes.\n", ctx.best_heur_sum);
                M *= 2;
            } else {
                fprintf(stderr, "    -> No improvement found at M=%d. Stopping heuristic scale-up.\n", M);
                break;
            }
        }
        
        memcpy(current_remap, ctx.best_heur_remap, 256);
        fprintf(stderr, "\nHeuristic Pre-pass Finished. Seed replaced by Best Heuristic (tested %llu)\n\n", ctx.heur_tested);

    } else if (do_reshuffle) {
        // Fisher-Yates uniform shuffle
        for (int i = 255; i > 0; i--) {
            int j = rand() % (i + 1);
            uint8_t temp = current_remap[i];
            current_remap[i] = current_remap[j];
            current_remap[j] = temp;
        }
    }

    // Evaluate the resulting start-point configuration unconditionally 
    size_t initial_sum = 0;
    for (int i = 0; i < file_count; i++) {
        for (size_t j = 0; j < files[i].file_size; j++) {
            remapped_buf[j] = current_remap[files[i].in_buf[j]];
        }
        size_t s = compress_buffer(remapped_buf, files[i].file_size, out_buf, out_capacity, dict_param_kb, lc_param, lp_param, pb_param);
        files[i].initial_comp_size = s;
        initial_sum += s;
    }
    
    normalize_remap(current_remap, lc_param);
    
    size_t current_sum = 0;
    for (int i = 0; i < file_count; i++) {
        for (size_t j = 0; j < files[i].file_size; j++) {
            remapped_buf[j] = current_remap[files[i].in_buf[j]];
        }
        size_t s = compress_buffer(remapped_buf, files[i].file_size, out_buf, out_capacity, dict_param_kb, lc_param, lp_param, pb_param);
        files[i].current_comp_size = s;
        files[i].milestone_comp_size = s; 
        current_sum += s;
    }
    
    size_t best_sum = current_sum;
    if (do_heuristic) {
        fprintf(stderr, "Initial Seed Total Size (Heuristic & Normalised): %zu bytes\n\n", current_sum);
    } else if (do_reshuffle) {
        fprintf(stderr, "Initial Seed Total Size (Randomised & Normalised): %zu bytes\n\n", current_sum);
    } else {
        fprintf(stderr, "Initial Seed Total Size (Pure): %zu bytes\n", initial_sum);
        fprintf(stderr, "Initial Seed Total Size (Normalised): %zu bytes\n\n", current_sum);
    }

    unsigned long long iterations = 0;
    int num_chunks = 1 << lc_param;
    int chunk_size = 256 / num_chunks;
    
    int perms3[5][3] = { {1, 0, 2}, {2, 1, 0}, {0, 2, 1}, {1, 2, 0}, {2, 0, 1} };
    const char *perms3_desc[5] = {
        "2-way element swap (Chunks A <-> B)", "2-way element swap (Chunks A <-> C)",
        "2-way element swap (Chunks B <-> C)", "3-way element rotation (A <- B <- C <- A)",
        "3-way element rotation (A -> B -> C -> A)" 
    };
    
    int perms2[1][2] = { {1, 0} };
    const char *perms2_desc[1] = { "2-way element swap (Chunks A <-> B)" };

    // =========================================================
    // THE OPTIMIZATION LOOP
    // =========================================================
    while (1) {
        time_t now = time(NULL);
        double elapsed = difftime(now, start_time);
        if (elapsed >= timeout) break;

        iterations++;

        // 3. Selection Phase
        int c[3] = {-1, -1, -1};
        int num_picks = (lc_param == 1) ? 2 : 3;

        c[0] = rand() % num_chunks;
        do { c[1] = rand() % num_chunks; } while(c[1] == c[0]);
        if (num_picks == 3) {
            do { c[2] = rand() % num_chunks; } while(c[2] == c[0] || c[2] == c[1]);
        }

        int idx[3];
        uint8_t val[3];
        for (int i = 0; i < num_picks; i++) {
            int offset = rand() % chunk_size;
            idx[i] = c[i] * chunk_size + offset;
            val[i] = current_remap[idx[i]];
        }

        // 4. Exhaustive Search over Valid Permutations
        size_t best_local_sum = current_sum; 
        uint8_t best_local_remap[256];
        size_t best_local_file_sizes[256]; 
        const char *best_desc = NULL;
        int found_improvement = 0;
        
        int num_perms = (num_picks == 3) ? 5 : 1;
        
        for (int p = 0; p < num_perms; p++) {
            uint8_t test_remap[256];
            memcpy(test_remap, current_remap, 256);
            
            if (num_picks == 3) {
                test_remap[idx[0]] = val[perms3[p][0]];
                test_remap[idx[1]] = val[perms3[p][1]];
                test_remap[idx[2]] = val[perms3[p][2]];
            } else {
                test_remap[idx[0]] = val[perms2[p][0]];
                test_remap[idx[1]] = val[perms2[p][1]];
            }
            
            normalize_remap(test_remap, lc_param);
            
            size_t test_sum = 0;
            int aborted = 0;
            size_t local_file_sizes[256]; 
            
            for (int i = 0; i < file_count; i++) {
                for (size_t j = 0; j < files[i].file_size; j++) {
                    remapped_buf[j] = test_remap[files[i].in_buf[j]];
                }
                
                local_file_sizes[i] = compress_buffer(remapped_buf, files[i].file_size, out_buf, out_capacity, dict_param_kb, lc_param, lp_param, pb_param);
                test_sum += local_file_sizes[i];
                
                if (test_sum >= best_local_sum) {
                    aborted = 1;
                    break;
                }
            }
            
            if (!aborted && test_sum < best_local_sum) {
                best_local_sum = test_sum;
                memcpy(best_local_remap, test_remap, 256);
                best_desc = (num_picks == 3) ? perms3_desc[p] : perms2_desc[p];
                found_improvement = 1;
                for (int i = 0; i < file_count; i++) {
                    best_local_file_sizes[i] = local_file_sizes[i];
                }
            }
        }
        
        // 5. Greedy Application
        if (found_improvement && best_local_sum < current_sum) {
            current_sum = best_local_sum;
            memcpy(current_remap, best_local_remap, 256);
            for (int i = 0; i < file_count; i++) {
                files[i].current_comp_size = best_local_file_sizes[i];
            }
            
            if (current_sum < best_sum) {
                long long total_delta_base = (long long)current_sum - (long long)baseline_sum;
                double total_pct_base = ((double)total_delta_base / (double)baseline_sum) * 100.0;
                
                long long total_delta_init = (long long)current_sum - (long long)initial_sum;
                double total_pct_init = ((double)total_delta_init / (double)initial_sum) * 100.0;
                
                long long total_delta_prev = (long long)current_sum - (long long)best_sum;
                double total_pct_prev = ((double)total_delta_prev / (double)best_sum) * 100.0;

                fprintf(stderr, "/* NEW GLOBAL BEST: %zu bytes (Iter %llu, Elapsed: %.1fs, Op: %s, dict:%dK/lc:%d/lp:%d/pb:%d) */\n", 
                        current_sum, iterations, elapsed, best_desc, dict_param_kb, lc_param, lp_param, pb_param);
                
                fprintf(stderr, "/* TOTAL IMPROVEMENT:\n");
                fprintf(stderr, "      vs Nonremapped: %lld bytes (%+.2f%%)\n", total_delta_base, total_pct_base);
                fprintf(stderr, "      vs Initial:     %lld bytes (%+.2f%%)\n", total_delta_init, total_pct_init);
                fprintf(stderr, "      vs Previous:    %lld bytes (%+.2f%%)\n", total_delta_prev, total_pct_prev);
                fprintf(stderr, "*/\n");
                
                fprintf(stderr, "/* FILE METRICS [vs Nonremapped]:\n");
                for (int i = 0; i < file_count; i++) {
                    long long delta = (long long)files[i].current_comp_size - (long long)files[i].baseline_comp_size;
                    double pct = ((double)delta / (double)files[i].baseline_comp_size) * 100.0;
                    fprintf(stderr, "   - %-*s : %10zu bytes (%+10lld B, %+7.2f%%)\n", 
                            max_filename_len, files[i].filename, files[i].current_comp_size, delta, pct);
                }
                
                fprintf(stderr, "   FILE METRICS [vs Initial]:\n");
                for (int i = 0; i < file_count; i++) {
                    long long delta = (long long)files[i].current_comp_size - (long long)files[i].initial_comp_size;
                    double pct = ((double)delta / (double)files[i].initial_comp_size) * 100.0;
                    fprintf(stderr, "   - %-*s : %+10lld bytes (%+7.2f%%)\n", 
                            max_filename_len, files[i].filename, delta, pct);
                }
                
                fprintf(stderr, "   FILE METRICS [vs Previous]:\n");
                for (int i = 0; i < file_count; i++) {
                    long long delta = (long long)files[i].current_comp_size - (long long)files[i].milestone_comp_size;
                    double pct = ((double)delta / (double)files[i].milestone_comp_size) * 100.0;
                    fprintf(stderr, "   - %-*s : %+10lld bytes (%+7.2f%%)\n", 
                            max_filename_len, files[i].filename, delta, pct);
                    
                    files[i].milestone_comp_size = files[i].current_comp_size;
                }
                fprintf(stderr, "*/\n");
                
                best_sum = current_sum;
                print_remap_table_as_source(current_remap);
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
