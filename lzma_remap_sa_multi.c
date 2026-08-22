#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <lzma.h>

/**
 * ==============================================================================
 * ALGORITHM OVERVIEW: Chunk-Partitioned Variable Neighborhood Search
 * ==============================================================================
 * 
 * 1. Tunable LZMA Markov Chain Modeling (Fast Search Phase)
 *    - dict (Dictionary Size): Configurable via '--dict=' in KB (default: 4).
 *    - lc (Literal Context): Configurable via '--lc=' (default: 3). 
 * 
 * 2. Progressive Heuristic Pre-Pass Partitioning
 *    - Recursively evaluates subsets of increasing granularity (M=2, 4, 8) to
 *      find the most structurally compatible initial seed.
 * 
 * 3. Variable Neighborhood Search (Multi-Swap Algorithm)
 *    - Starts by applying N (default: 128) random pair swaps simultaneously.
 *    - Crucial Constraint: Swaps are only valid if the elements belong to 
 *      different chunks. This preserves intra-chunk entropy while optimizing 
 *      inter-chunk mappings.
 *    - If no improvement is found after K (default: 1000) iterations, the search 
 *      space is narrowed by halving N. This mimics simulated annealing.
 * 
 * 4. Exhaustive "Thorough" Fallback and Candidate Archiving
 *    - When N drops to 1, and K iterations pass without improvement, the algorithm
 *      enters Thorough Mode, systematically testing EVERY possible valid pair.
 *    - If an improvement is found, it greedily applies it and jumps back to N=128.
 *    - If NO improvement is found across all possible pairs, a local minima is 
 *      confirmed. The table is saved as a "Result Candidate".
 *    - The algorithm then reverts to the initial baseline (or a fresh random 
 *      reshuffle) and begins a completely new optimization path.
 * 
 * 5. Detailed Metric Reporting & Final Evaluation
 *    - Inter-iteration reports detail per-file and total aggregate delta 
 *      percentages compared to the Identity map, the Initial seed map, and 
 *      the Previous step map. It also exposes the active LZMA settings and 
 *      candidate count.
 *    - Final Evaluation runs an absolute data entropy check via LZMA2 Preset 7 
 *      (Extreme) on all discovered result candidates. Includes an absolute 
 *      baseline of Identity remap mapped explicitly at lc=3, lp=0, pb=2.
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
} FileData;

typedef struct {
    int M, C, S;
    unsigned long long heur_tested, heur_limit;
    size_t best_heur_sum;
    uint8_t best_heur_remap[256];
    int file_count;
    FileData *files;
    uint8_t *remapped_buf;
    uint8_t *out_buf;
    size_t out_capacity;
    int dict_param_kb, lc_param, lp_param, pb_param;
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
    printf("  --swaps=N         Initial number of simultaneous pair swaps (default: 128).\n");
    printf("  --stagnation=K    Iterations without improvement before halving swaps (default: 1000).\n");
    printf("  --dict=KB         Set LZMA dictionary size in kilobytes (default: 4).\n");
    printf("  --lc=BITS         Set LZMA literal context bits (1-4, default: 3).\n");
    printf("  --lp=BITS         Set LZMA literal position bits (0-4, default: 0).\n");
    printf("  --pb=BITS         Set LZMA position bits (0-4, default: 2).\n");
    printf("\n");
}

// ------------------------------------------------------------------------------------------
// Core Evaluation Helpers
// ------------------------------------------------------------------------------------------
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

size_t compress_buffer_extreme(const uint8_t *in_buf, size_t in_len, uint8_t *out_buf, size_t out_capacity, int lc_param, int lp_param, int pb_param) {
    lzma_options_lzma opt;
    if (lzma_lzma_preset(&opt, 7 | LZMA_PRESET_EXTREME)) return out_capacity + 1;
    opt.dict_size = 8 * 1024 * 1024; 
    opt.lc = lc_param; opt.lp = lp_param; opt.pb = pb_param;
    lzma_filter filters[2] = { { .id = LZMA_FILTER_LZMA2, .options = &opt }, { .id = LZMA_VLI_UNKNOWN, .options = NULL } };
    size_t out_pos = 0;
    lzma_ret ret = lzma_stream_buffer_encode(filters, LZMA_CHECK_CRC32, NULL, in_buf, in_len, out_buf, &out_pos, out_capacity);
    return (ret == LZMA_OK) ? out_pos : out_capacity + 1;
}

size_t evaluate_remap(const uint8_t *remap_table, int file_count, FileData *files, 
                      uint8_t *remapped_buf, uint8_t *out_buf, size_t out_capacity, 
                      int dict, int lc, int lp, int pb, size_t abort_threshold, 
                      size_t *individual_sizes) {
    size_t total = 0;
    for (int i = 0; i < file_count; i++) {
        for (size_t j = 0; j < files[i].file_size; j++) {
            remapped_buf[j] = remap_table[files[i].in_buf[j]];
        }
        size_t s = compress_buffer(remapped_buf, files[i].file_size, out_buf, out_capacity, dict, lc, lp, pb);
        if (individual_sizes) individual_sizes[i] = s;
        total += s;
        if (abort_threshold > 0 && total >= abort_threshold) return total;
    }
    return total;
}

size_t evaluate_extreme(const uint8_t *remap_table, size_t file_size, const uint8_t *in_buf,
                        uint8_t *remapped_buf, uint8_t *out_buf, size_t out_capacity, 
                        int lc, int lp, int pb) {
    for (size_t j = 0; j < file_size; j++) remapped_buf[j] = remap_table[in_buf[j]];
    return compress_buffer_extreme(remapped_buf, file_size, out_buf, out_capacity, lc, lp, pb);
}

// ------------------------------------------------------------------------------------------
// Reporting & Formatting Helpers
// ------------------------------------------------------------------------------------------
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
    fprintf(stderr, "\n};\n\n");
    fflush(stderr); 
}

void print_detailed_report(int file_count, FileData *files, size_t *new_sizes, 
                           unsigned long long iter, double elapsed, int swaps, 
                           const char* mode_name, int is_global_best,
                           int lc, int lp, int pb, int num_candidates) {
    size_t total_baseline = 0, total_initial = 0, total_prev = 0, total_new = 0;
    
    for (int i = 0; i < file_count; i++) {
        total_baseline += files[i].baseline_comp_size;
        total_initial += files[i].initial_comp_size;
        total_prev += files[i].current_comp_size;
        total_new += new_sizes[i];
    }
    
    fprintf(stderr, "\n/* %s: %zu bytes (Iter %llu, %.1fs, %s", 
            is_global_best ? "NEW GLOBAL BEST" : "Local Improvement", 
            total_new, iter, elapsed, mode_name);
    if (swaps > 0) fprintf(stderr, ", %d swaps", swaps);
    fprintf(stderr, ") [lc=%d, lp=%d, pb=%d | cands=%d] */\n", lc, lp, pb, num_candidates);
    
    fprintf(stderr, "/* %-20s | %-10s | %-12s | %-12s | %-12s\n", "File", "New Size", "vs Identity", "vs Seed", "vs Previous");
    for (int i = 0; i < file_count; i++) {
        double pct_id = files[i].baseline_comp_size ? ((double)((long long)new_sizes[i] - (long long)files[i].baseline_comp_size) / files[i].baseline_comp_size) * 100.0 : 0.0;
        double pct_seed = files[i].initial_comp_size ? ((double)((long long)new_sizes[i] - (long long)files[i].initial_comp_size) / files[i].initial_comp_size) * 100.0 : 0.0;
        double pct_prev = files[i].current_comp_size ? ((double)((long long)new_sizes[i] - (long long)files[i].current_comp_size) / files[i].current_comp_size) * 100.0 : 0.0;
        
        fprintf(stderr, " * %-20s | %10zu | %+11.2f%% | %+11.2f%% | %+11.2f%%\n", 
                files[i].filename, new_sizes[i], pct_id, pct_seed, pct_prev);
    }
    
    // Total aggregate reporting
    double tot_pct_id = total_baseline ? ((double)((long long)total_new - (long long)total_baseline) / total_baseline) * 100.0 : 0.0;
    double tot_pct_seed = total_initial ? ((double)((long long)total_new - (long long)total_initial) / total_initial) * 100.0 : 0.0;
    double tot_pct_prev = total_prev ? ((double)((long long)total_new - (long long)total_prev) / total_prev) * 100.0 : 0.0;
    
    fprintf(stderr, " * %-20s | %10zu | %+11.2f%% | %+11.2f%% | %+11.2f%%\n", 
            "TOTAL", total_new, tot_pct_id, tot_pct_seed, tot_pct_prev);
    
    fprintf(stderr, " */\n");
    fflush(stderr);
}

// ------------------------------------------------------------------------------------------
// Chunk Alignment Normalization
// ------------------------------------------------------------------------------------------
int cmp_uint8(const void *a, const void *b) { return (*(const uint8_t *)a) - (*(const uint8_t *)b); }

typedef struct { int sum; uint8_t first_val; uint8_t data[256]; } Chunk;

int cmp_chunk(const void *a, const void *b) {
    const Chunk *ca = (const Chunk *)a, *cb = (const Chunk *)b;
    if (ca->sum != cb->sum) return ca->sum - cb->sum;
    return ca->first_val - cb->first_val;
}

void normalize_remap(uint8_t *remap, int lc) {
    int num_chunks = 1 << lc;
    int chunk_size = 256 / num_chunks;
    Chunk chunks[16]; 
    for (int c = 0; c < num_chunks; c++) {
        chunks[c].sum = 0;
        for (int i = 0; i < chunk_size; i++) chunks[c].data[i] = remap[c * chunk_size + i];
        qsort(chunks[c].data, chunk_size, sizeof(uint8_t), cmp_uint8);
        for (int i = 0; i < chunk_size; i++) chunks[c].sum += chunks[c].data[i];
        chunks[c].first_val = chunks[c].data[0];
    }
    qsort(chunks, num_chunks, sizeof(Chunk), cmp_chunk);
    for (int c = 0; c < num_chunks; c++) {
        for (int i = 0; i < chunk_size; i++) remap[c * chunk_size + i] = chunks[c].data[i];
    }
}

void generate_partitions(int element_idx, int num_active_chunks, int *counts, int *assignment, HeurContext *ctx) {
    if (ctx->heur_tested >= ctx->heur_limit) return;
    if (difftime(time(NULL), ctx->start_time) >= ctx->timeout) return;
    
    if (element_idx == ctx->C * ctx->M) {
        if (ctx->M > 2) {
            int is_duplicate = 1;
            for (int i = 0; i < ctx->C * ctx->M; i += 2) {
                if (assignment[i] != assignment[i + 1]) { is_duplicate = 0; break; }
            }
            if (is_duplicate) return; 
        }
        uint8_t heur_remap[256];
        int fill[16] = {0};
        for (int i = 0; i < ctx->C * ctx->M; i++) {
            int c = assignment[i];
            int offset = fill[c] * ctx->S;
            for (int j = 0; j < ctx->S; j++) heur_remap[c * (ctx->M * ctx->S) + offset + j] = i * ctx->S + j; 
            fill[c]++;
        }
        normalize_remap(heur_remap, ctx->lc_param);
        size_t test_sum = evaluate_remap(heur_remap, ctx->file_count, ctx->files, ctx->remapped_buf, ctx->out_buf, ctx->out_capacity, ctx->dict_param_kb, ctx->lc_param, ctx->lp_param, ctx->pb_param, ctx->best_heur_sum, NULL);
        if (test_sum < ctx->best_heur_sum) {
            ctx->best_heur_sum = test_sum;
            memcpy(ctx->best_heur_remap, heur_remap, 256);
        }
        ctx->heur_tested++;
        return;
    }
    
    for (int c = 0; c < num_active_chunks; c++) {
        if (counts[c] < ctx->M) {
            counts[c]++; assignment[element_idx] = c;
            generate_partitions(element_idx + 1, num_active_chunks, counts, assignment, ctx);
            counts[c]--;
            if (ctx->heur_tested >= ctx->heur_limit) return;
        }
    }
    if (num_active_chunks < ctx->C) {
        counts[num_active_chunks]++; assignment[element_idx] = num_active_chunks;
        generate_partitions(element_idx + 1, num_active_chunks + 1, counts, assignment, ctx);
        counts[num_active_chunks]--;
    }
}

int main(int argc, char **argv) {
    int timeout = 600; 
    int dict_param_kb = 4, lc_param = 3, lp_param = 0, pb_param = 2;
    int do_reshuffle = 0, do_heuristic = 0, heur_limit = 128;
    
    // Parameters for Variable Neighborhood Search
    int initial_swaps = 128;
    int max_stagnation = 1000;
    
    const char **filenames = malloc(argc * sizeof(char*));
    int file_count = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) { print_help(argv[0]); free(filenames); return EXIT_SUCCESS; }
        else if (strncmp(argv[i], "--timeout=", 10) == 0) timeout = atoi(argv[i] + 10);
        else if (strcmp(argv[i], "--reshuffle") == 0) do_reshuffle = 1;
        else if (strcmp(argv[i], "--heuristic") == 0) do_heuristic = 1;
        else if (strncmp(argv[i], "--heuristic=", 12) == 0) { do_heuristic = 1; heur_limit = atoi(argv[i] + 12); }
        else if (strncmp(argv[i], "--swaps=", 8) == 0) initial_swaps = atoi(argv[i] + 8);
        else if (strncmp(argv[i], "--stagnation=", 13) == 0) max_stagnation = atoi(argv[i] + 13);
        else if (strncmp(argv[i], "--dict=", 7) == 0) dict_param_kb = atoi(argv[i] + 7);
        else if (strncmp(argv[i], "--lc=", 5) == 0) lc_param = atoi(argv[i] + 5);
        else if (strncmp(argv[i], "--lp=", 5) == 0) lp_param = atoi(argv[i] + 5);
        else if (strncmp(argv[i], "--pb=", 5) == 0) pb_param = atoi(argv[i] + 5);
        else filenames[file_count++] = argv[i];
    }

    if (file_count == 0 || lc_param < 1 || lc_param > 4 || lp_param < 0 || lp_param > 4 || pb_param < 0 || pb_param > 4) {
        fprintf(stderr, "Error: Invalid arguments or missing input files.\n");
        free(filenames);
        return EXIT_FAILURE;
    }

    srand((unsigned int)time(NULL));

    FileData *files = malloc(file_count * sizeof(FileData));
    size_t max_file_size = 0; int max_filename_len = 0;
    for (int i = 0; i < file_count; i++) {
        files[i].filename = filenames[i];
        if ((int)strlen(filenames[i]) > max_filename_len) max_filename_len = (int)strlen(filenames[i]);
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

    // 1. Evaluate True Baseline
    uint8_t identity_remap[256];
    for (int i = 0; i < 256; i++) identity_remap[i] = i;
    size_t baseline_sum = evaluate_remap(identity_remap, file_count, files, remapped_buf, out_buf, out_capacity, dict_param_kb, lc_param, lp_param, pb_param, 0, NULL);
    for (int i = 0; i < file_count; i++) {
        files[i].baseline_comp_size = evaluate_remap(identity_remap, 1, &files[i], remapped_buf, out_buf, out_capacity, dict_param_kb, lc_param, lp_param, pb_param, 0, NULL);
    }
    
    time_t start_time = time(NULL);
    uint8_t current_remap[256];
    memcpy(current_remap, seed_remap, 256);

    // 2. Pre-Configuration Branch
    if (do_heuristic) {
        HeurContext ctx = { .C = 1 << lc_param, .heur_tested = 0, .heur_limit = heur_limit, .best_heur_sum = baseline_sum, 
                            .file_count = file_count, .files = files, .remapped_buf = remapped_buf, .out_buf = out_buf, 
                            .out_capacity = out_capacity, .dict_param_kb = dict_param_kb, .lc_param = lc_param, 
                            .lp_param = lp_param, .pb_param = pb_param, .start_time = start_time, .timeout = timeout };
        for (int i = 0; i < 256; i++) ctx.best_heur_remap[i] = i; 
        int M = 2;
        while (M * ctx.C <= 256 && ctx.heur_tested < ctx.heur_limit) {
            ctx.M = M; ctx.S = 256 / (ctx.C * M);
            int counts[16] = {0}, assignment[256] = {0};
            size_t before_m_sum = ctx.best_heur_sum;
            generate_partitions(0, 0, counts, assignment, &ctx);
            if (ctx.best_heur_sum < before_m_sum) M *= 2; else break;
        }
        memcpy(current_remap, ctx.best_heur_remap, 256);
    } else if (do_reshuffle) {
        for (int i = 255; i > 0; i--) {
            int j = rand() % (i + 1); uint8_t temp = current_remap[i]; current_remap[i] = current_remap[j]; current_remap[j] = temp;
        }
    }

    normalize_remap(current_remap, lc_param);
    
    // Save the finalized Initial Seed
    uint8_t loop_start_remap[256];
    memcpy(loop_start_remap, current_remap, 256);

    size_t *initial_sizes = malloc(file_count * sizeof(size_t));
    size_t current_sum = evaluate_remap(current_remap, file_count, files, remapped_buf, out_buf, out_capacity, dict_param_kb, lc_param, lp_param, pb_param, 0, NULL);
    for (int i = 0; i < file_count; i++) {
        files[i].initial_comp_size = evaluate_remap(current_remap, 1, &files[i], remapped_buf, out_buf, out_capacity, dict_param_kb, lc_param, lp_param, pb_param, 0, NULL);
        files[i].current_comp_size = files[i].initial_comp_size;
        initial_sizes[i] = files[i].initial_comp_size;
    }
    size_t best_sum = current_sum;

    // Report 0 (Initial Baseline State)
    print_detailed_report(file_count, files, initial_sizes, 0, 0.0, 0, "Initial Seed Setup", 1, lc_param, lp_param, pb_param, 0);
    free(initial_sizes);

    unsigned long long iterations = 0;
    int chunk_size = 256 / (1 << lc_param);
    
    // Candidate Tracking Setup
    int candidate_capacity = 16;
    uint8_t (*candidates)[256] = malloc(candidate_capacity * 256);
    int num_candidates = 0;
    
    // Variable Neighborhood Search State Tracking
    int num_swaps = initial_swaps;
    int stagnation_counter = 0;
    int search_mode = 0; // 0 = Random Swaps, 1 = Thorough Evaluation

    // =========================================================
    // THE OPTIMIZATION LOOP
    // =========================================================
    while (1) {
        time_t now = time(NULL);
        double elapsed = difftime(now, start_time);
        if (elapsed >= timeout) break;
        iterations++;

        if (search_mode == 0) {
            // A. Random Multi-Swap Strategy
            uint8_t test_remap[256];
            memcpy(test_remap, current_remap, 256);
            
            for (int k = 0; k < num_swaps; k++) {
                int a = rand() % 256, b;
                do { b = rand() % 256; } while ((a / chunk_size) == (b / chunk_size)); // Ensure distinct chunks
                uint8_t tmp = test_remap[a]; test_remap[a] = test_remap[b]; test_remap[b] = tmp;
            }
            
            normalize_remap(test_remap, lc_param);
            size_t *local_file_sizes = malloc(file_count * sizeof(size_t));
            size_t test_sum = evaluate_remap(test_remap, file_count, files, remapped_buf, out_buf, out_capacity, dict_param_kb, lc_param, lp_param, pb_param, current_sum, local_file_sizes);
            
            if (test_sum < current_sum) {
                int is_global = (test_sum < best_sum);
                
                // Detailed Reporting Output
                print_detailed_report(file_count, files, local_file_sizes, iterations, elapsed, num_swaps, "Random Swaps", is_global, lc_param, lp_param, pb_param, num_candidates);
                
                if (is_global) best_sum = test_sum;
                current_sum = test_sum;
                memcpy(current_remap, test_remap, 256);
                for (int i = 0; i < file_count; i++) files[i].current_comp_size = local_file_sizes[i];
                
                stagnation_counter = 0;
                num_swaps = initial_swaps; 
                
                // Only print the C source code structure when breaking a global record to avoid clutter
                if (is_global) print_remap_table_as_source("seed_remap", current_remap);
            } else {
                stagnation_counter++;
                if (stagnation_counter >= max_stagnation) {
                    if (num_swaps > 1) {
                        num_swaps /= 2; // Annealing: Constrict search radius
                        stagnation_counter = 0;
                    } else {
                        search_mode = 1; // Exhausted random combinations at N=1, switch to thorough
                        stagnation_counter = 0;
                    }
                }
            }
            free(local_file_sizes);
        } 
        else if (search_mode == 1) {
            // B. Exhaustive (Thorough) Pair Check
            size_t best_thorough_sum = current_sum;
            uint8_t best_thorough_remap[256];
            size_t *best_thorough_files = malloc(file_count * sizeof(size_t));
            int found_improvement = 0;
            
            for (int i = 0; i < 256 && !found_improvement && difftime(time(NULL), start_time) < timeout; i++) {
                for (int j = i + 1; j < 256; j++) {
                    if ((i / chunk_size) == (j / chunk_size)) continue;
                    
                    uint8_t test_remap[256];
                    memcpy(test_remap, current_remap, 256);
                    uint8_t tmp = test_remap[i]; test_remap[i] = test_remap[j]; test_remap[j] = tmp;
                    normalize_remap(test_remap, lc_param);
                    
                    size_t *local_file_sizes = malloc(file_count * sizeof(size_t));
                    size_t test_sum = evaluate_remap(test_remap, file_count, files, remapped_buf, out_buf, out_capacity, dict_param_kb, lc_param, lp_param, pb_param, best_thorough_sum, local_file_sizes);
                    
                    if (test_sum < best_thorough_sum) {
                        best_thorough_sum = test_sum;
                        memcpy(best_thorough_remap, test_remap, 256);
                        for(int f = 0; f < file_count; f++) best_thorough_files[f] = local_file_sizes[f];
                        found_improvement = 1;
                        free(local_file_sizes);
                        break; // Greedily capture the first discovered improvement
                    }
                    free(local_file_sizes);
                }
            }
            
            if (found_improvement) {
                int is_global = (best_thorough_sum < best_sum);
                
                // Detailed Reporting Output
                print_detailed_report(file_count, files, best_thorough_files, iterations, elapsed, 0, "Thorough Evaluation", is_global, lc_param, lp_param, pb_param, num_candidates);
                
                if (is_global) best_sum = best_thorough_sum;
                current_sum = best_thorough_sum;
                memcpy(current_remap, best_thorough_remap, 256);
                for (int i = 0; i < file_count; i++) files[i].current_comp_size = best_thorough_files[i];
                
                search_mode = 0;
                num_swaps = initial_swaps;
                stagnation_counter = 0;
                
                if (is_global) print_remap_table_as_source("seed_remap", current_remap);
            } else {
                // C. Local Minimum Confirmed - Save Candidate & Restart
                fprintf(stderr, "\n/* Local Minimum hit at %zu. Saving Candidate %d and Reverting... */\n\n", current_sum, num_candidates + 1);
                
                if (num_candidates == candidate_capacity) {
                    candidate_capacity *= 2;
                    candidates = realloc(candidates, candidate_capacity * 256);
                }
                memcpy(candidates[num_candidates++], current_remap, 256);
                
                // Revert to start or random baseline
                if (do_reshuffle) {
                    for (int i = 0; i < 256; i++) current_remap[i] = seed_remap[i]; 
                    for (int i = 255; i > 0; i--) {
                        int j = rand() % (i + 1); uint8_t tmp = current_remap[i]; current_remap[i] = current_remap[j]; current_remap[j] = tmp;
                    }
                } else {
                    memcpy(current_remap, loop_start_remap, 256);
                }
                normalize_remap(current_remap, lc_param);
                
                // CRITICAL: We must re-evaluate the reset state and update the file sizes 
                // so the "vs Previous" reporting starts fresh from the new baseline peak.
                size_t *reset_sizes = malloc(file_count * sizeof(size_t));
                current_sum = evaluate_remap(current_remap, file_count, files, remapped_buf, out_buf, out_capacity, dict_param_kb, lc_param, lp_param, pb_param, 0, reset_sizes);
                for (int f = 0; f < file_count; f++) files[f].current_comp_size = reset_sizes[f];
                free(reset_sizes);
                
                search_mode = 0;
                num_swaps = initial_swaps;
                stagnation_counter = 0;
            }
            free(best_thorough_files);
        }
    }

    fprintf(stderr, "\n/* Search finished. Iterations: %llu. Extracted %d Result Candidates. */\n\n", iterations, num_candidates);

    // =========================================================
    // DUMP SAVED CANDIDATES
    // =========================================================
    for (int k = 0; k < num_candidates; k++) {
        char cand_name[32];
        snprintf(cand_name, sizeof(cand_name), "candidate_%d_remap", k + 1);
        print_remap_table_as_source(cand_name, candidates[k]);
    }

    // =========================================================
    // FINAL EXTREME COMPRESSION EVALUATION & DYNAMIC TABLE
    // =========================================================
    fprintf(stderr, "### Final Extreme Compression Evaluation (8MB Dict, Extreme Match Finder)\n\n");
    
    // Total Columns = 4 standard + candidates 
    int cols = 4 + num_candidates; 
    int *w_cols = calloc(cols + 2, sizeof(int)); 
    
    int w_file = max_filename_len < 4 ? 4 : max_filename_len;
    w_cols[0] = 22; // Absolute Baseline
    w_cols[1] = 16; // Current Identity
    w_cols[2] = 20; // Current Seed
    for (int k = 0; k < num_candidates; k++) w_cols[3 + k] = 14; 
    w_cols[cols - 1] = 19; 
    
    int w_sav_abs = 18, w_sav_id = 21, w_sav_sd = 17;
    
    // Header Row Generation
    fprintf(stderr, "| %-*s | %-*s | %-*s | %-*s | ", 
            w_file, "File", 
            w_cols[0], "Abs Baseline (3/0/2)", 
            w_cols[1], "Identity (Bytes)", 
            w_cols[2], "Initial Seed (Bytes)");
            
    for (int k = 0; k < num_candidates; k++) {
        char cand_head[32]; snprintf(cand_head, sizeof(cand_head), "Candidate %d", k + 1);
        fprintf(stderr, "%-*s | ", w_cols[3 + k], cand_head);
    }
    
    fprintf(stderr, "%-*s | %-*s | %-*s | %-*s |\n", 
            w_cols[cols - 1], "Final Remap (Bytes)", 
            w_sav_abs, "Savings (vs Abs)", 
            w_sav_id, "Savings (vs Identity)", 
            w_sav_sd, "Savings (vs Seed)");

    // Strict Markdown Separator Padding (Aligned text rendering support)
    fprintf(stderr, "|-"); for(int k = 0; k < w_file; k++) fputc('-', stderr);
    for (int c = 0; c < cols; c++) { fprintf(stderr, "-|-"); for(int k = 0; k < w_cols[c]; k++) fputc('-', stderr); }
    fprintf(stderr, "-|-"); for(int k = 0; k < w_sav_abs; k++) fputc('-', stderr);
    fprintf(stderr, "-|-"); for(int k = 0; k < w_sav_id; k++) fputc('-', stderr);
    fprintf(stderr, "-|-"); for(int k = 0; k < w_sav_sd; k++) fputc('-', stderr);
    fprintf(stderr, "-|\n");

    for (int i = 0; i < file_count; i++) {
        size_t *eval_sizes = malloc(cols * sizeof(size_t));
        
        // Col 0: Absolute Baseline (Identity map, forcibly evaluated with lc=3, lp=0, pb=2)
        eval_sizes[0] = evaluate_extreme(identity_remap, files[i].file_size, files[i].in_buf, remapped_buf, out_buf, out_capacity, 3, 0, 2);
        
        // Col 1: Current Parameter Identity map
        eval_sizes[1] = evaluate_extreme(identity_remap, files[i].file_size, files[i].in_buf, remapped_buf, out_buf, out_capacity, lc_param, lp_param, pb_param);
        
        // Col 2: Current Parameter Initial Seed
        eval_sizes[2] = evaluate_extreme(loop_start_remap, files[i].file_size, files[i].in_buf, remapped_buf, out_buf, out_capacity, lc_param, lp_param, pb_param);
        
        for (int k = 0; k < num_candidates; k++) {
            eval_sizes[3 + k] = evaluate_extreme(candidates[k], files[i].file_size, files[i].in_buf, remapped_buf, out_buf, out_capacity, lc_param, lp_param, pb_param);
        }
        
        eval_sizes[cols - 1] = evaluate_extreme(current_remap, files[i].file_size, files[i].in_buf, remapped_buf, out_buf, out_capacity, lc_param, lp_param, pb_param);

        double pct_abs  = eval_sizes[0] ? ((double)((long long)eval_sizes[cols - 1] - (long long)eval_sizes[0]) / (double)eval_sizes[0]) * 100.0 : 0.0;
        double pct_id   = eval_sizes[1] ? ((double)((long long)eval_sizes[cols - 1] - (long long)eval_sizes[1]) / (double)eval_sizes[1]) * 100.0 : 0.0;
        double pct_seed = eval_sizes[2] ? ((double)((long long)eval_sizes[cols - 1] - (long long)eval_sizes[2]) / (double)eval_sizes[2]) * 100.0 : 0.0;

        char sav_abs_str[32], sav_id_str[32], sav_sd_str[32];
        snprintf(sav_abs_str, sizeof(sav_abs_str), "%+.2f%%", pct_abs);
        snprintf(sav_id_str, sizeof(sav_id_str), "%+.2f%%", pct_id);
        snprintf(sav_sd_str, sizeof(sav_sd_str), "%+.2f%%", pct_seed);

        fprintf(stderr, "| %-*s | ", w_file, files[i].filename);
        for (int c = 0; c < cols; c++) fprintf(stderr, "%*zu | ", w_cols[c], eval_sizes[c]);
        fprintf(stderr, "%*s | %*s | %*s |\n", w_sav_abs, sav_abs_str, w_sav_id, sav_id_str, w_sav_sd, sav_sd_str);
        
        free(eval_sizes);
    }
    fprintf(stderr, "\n");

    for (int i = 0; i < file_count; i++) free(files[i].in_buf);
    free(files);
    free(filenames);
    free(remapped_buf); 
    free(out_buf);
    free(w_cols);
    free(candidates);
    
    return EXIT_SUCCESS;
}
