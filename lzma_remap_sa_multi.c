#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <lzma.h>

/**
 * ==============================================================================
 * ALGORITHM OVERVIEW: Variable Neighborhood Search & Bit-Heuristics
 * ==============================================================================
 * 
 * 1. Tunable LZMA Markov Chain Modeling (Fast Search Phase)
 *    - dict (Dictionary Size): Configurable via '--dict=' in KB (default: 4).
 *    - lc (Literal Context): Configurable via '--lc=' (default: 3). 
 * 
 * 2. External Seed Parsing & Validation
 *    - Dynamically scans a target file for the exact seed declaration using the 
 *      lc/lp/pb signature.
 *    - Implements a validation pass using a frequency mapping array to guarantee
 *      the parsed result is a strict 1:1 permutation of the 0-255 byte space.
 * 
 * 3. Exhaustive Bit-Permutation Heuristic Pre-Pass
 *    - When '--heuristic' is active, the system generates all 40,320 (8!) 
 *      possible remaps derived from swapping bits within a byte of the identity 
 *      remap using Heap's Algorithm.
 *    - Each candidate is rigorously evaluated using LZMA Extreme (Preset 6, 8MB dict).
 *    - The best permutation anchors the starting state for the multi-swap phase.
 * 
 * 4. Variable Neighborhood Search (Multi-Swap Algorithm)
 *    - Starts by applying N (default: 256) random pair swaps simultaneously.
 *    - If '--normalize' is active, swaps are ONLY valid if elements belong to 
 *      different chunks. Otherwise, global swaps are permitted.
 *    - If no improvement is found after K (default: 1000) iterations, the search 
 *      space is narrowed by halving N (simulated annealing).
 * 
 * 5. N=1 Optimization and Tested Pair Tracking (Bit Array Collision Avoidance)
 *    - When N drops to 1, a bit array (tested_pairs) is utilized to memorize 
 *      the previously checked swap combinations.
 *    - Random swaps query this array to immediately skip already-tested pairs. 
 *    - If the random generator encounters too many tested pairs (dense space), 
 *      the system preemptively breaks into Exhaustive Thorough mode.
 * 
 * 6. Exhaustive "Thorough" Fallback and Candidate Archiving
 *    - Systematically tests EVERY possible valid pair. It consults the same 
 *      bit array to skip previously evaluated pairs, vastly accelerating the pass.
 *    - If an improvement is found, it greedily applies it, wipes the bit array 
 *      clean (since the landscape changed), and jumps back to N=256.
 *    - If NO improvement is found across all possible pairs, a local minima is 
 *      confirmed. The table is archived as a "Result Candidate", the bit array 
 *      is reset, and a new search path begins.
 * 
 * 7. Final Evaluation and Winner Selection
 *    - Evaluates LZMA2 Preset 6 (Extreme, 8MB dict) for all archived candidates 
 *      and the final unfinished remap.
 *    - Sorts all candidates based on their total compressed size.
 * 
 * 8. Result Persistence & Seed Evolution
 *    - If the final extreme evaluation determines the absolute best candidate 
 *      surpasses the initial loaded seed, the system physically patches the 
 *      original seed file, substituting the new permutation under a '.new' file.
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
    int original_idx;
    uint8_t remap[256];
    size_t *file_s_rem;
    size_t total_rem;
} FinalEval;

void print_help(const char *prog_name) {
    printf("Usage: %s [OPTIONS] <input_file_1> [input_file_2 ...]\n\n", prog_name);
    printf("Options:\n");
    printf("  --help            Show this help message and exit.\n");
    printf("  --timeout=SEC     Set timeout in seconds (default: 600).\n");
    printf("  --reshuffle       Randomize the initial remap table before starting.\n");
    printf("  --seedfile=FILE   Load the seed remap from an external file.\n");
    printf("  --heuristic       Run exhaustive 8-bit permutation heuristic (40,320 evals).\n");
    printf("  --normalize       Enable normalization (sorting) of the remap tables (disabled by default).\n");
    printf("  --swaps=N         Initial number of simultaneous pair swaps (default: 256).\n");
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
    // Algorithm: Apply LZMA2 Preset 6 (Extreme) for maximum exhaustive evaluation.
    if (lzma_lzma_preset(&opt, 6 | LZMA_PRESET_EXTREME)) return out_capacity + 1;
    opt.dict_size = 8 * 1024 * 1024; // 8MB dictionary as specified
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

int cmp_eval(const void *a, const void *b) {
    size_t sum_a = ((const FinalEval*)a)->total_rem;
    size_t sum_b = ((const FinalEval*)b)->total_rem;
    if (sum_a < sum_b) return -1;
    if (sum_a > sum_b) return 1;
    return 0;
}

// ------------------------------------------------------------------------------------------
// Rigid Seed File Parser & Permutation Validator
// ------------------------------------------------------------------------------------------
int load_seed_from_file(const char *filename, int lc, int lp, int pb, unsigned char *out_remap) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open seed file '%s'.\n", filename);
        return 0;
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
        fprintf(stderr, "Error: Configuration '%s' not found in seed file.\n", target_decl);
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
 * Streams the original file to a new file character by character. 
 * When the target declaration signature is matched, it safely skips the old 
 * array initialization payload and injects the new optimized permutation, 
 * preserving all surrounding code, comments, and file structure.
 */
void save_new_seed_file(const char *orig_path, int lc, int lp, int pb, const uint8_t *best_remap) {
    char new_path[1024];
    snprintf(new_path, sizeof(new_path), "%s.new", orig_path);
    
    FILE *in = fopen(orig_path, "r");
    FILE *out = fopen(new_path, "w");
    if (!in || !out) {
        if (in) fclose(in);
        if (out) fclose(out);
        return;
    }

    char target_decl[64];
    snprintf(target_decl, sizeof(target_decl), "remap_seed_%d%d%d", lc, lp, pb);
    int target_len = strlen(target_decl);
    
    int match_idx = 0;
    int in_target = 0;
    int c;

    while ((c = fgetc(in)) != EOF) {
        if (!in_target) {
            fputc(c, out);
            if (c == target_decl[match_idx]) {
                match_idx++;
                if (match_idx == target_len) {
                    in_target = 1; // Discovered the specific C array declaration block!
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
    
    fclose(in);
    fclose(out);
    fprintf(stderr, "\n/* Successfully saved optimized seed configuration to: %s */\n\n", new_path);
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

// ------------------------------------------------------------------------------------------
// Bit-Permutation Heuristic Evaluator
// ------------------------------------------------------------------------------------------
typedef struct {
    uint8_t best_remap[256];
    int best_perm[8];
    size_t best_size;
    int file_count;
    FileData *files;
    uint8_t *remapped_buf;
    uint8_t *out_buf;
    size_t out_capacity;
    int lc, lp, pb;
    unsigned long long count;
} HeuristicBitCtx;

/**
 * Algorithm: Heap's Algorithm for Exhaustive Bit-Permutation
 * Iteratively constructs every unique permutation of an 8-element set (40,320 combinations).
 * For each layout, it fabricates a full 256-byte translation table where every byte 
 * has its individual bits strictly reordered according to the active permutation.
 * Evaluates candidates aggressively via LZMA Extreme 6 preset.
 */
void generate_bit_permutations(int k, int *perm, HeuristicBitCtx *ctx) {
    if (k == 1) {
        uint8_t test_remap[256];
        for (int i = 0; i < 256; i++) {
            uint8_t mapped = 0;
            for (int bit = 0; bit < 8; bit++) {
                if (i & (1 << bit)) {
                    mapped |= (1 << perm[bit]);
                }
            }
            test_remap[i] = mapped;
        }

        size_t total_size = 0;
        for (int f = 0; f < ctx->file_count; f++) {
            // Mandated evaluation constraints: Extreme LZMA 6 preset utilizing 8MB Dictionary
            size_t s = evaluate_extreme(test_remap, ctx->files[f].file_size, ctx->files[f].in_buf, 
                                        ctx->remapped_buf, ctx->out_buf, ctx->out_capacity, 
                                        ctx->lc, ctx->lp, ctx->pb);
            total_size += s;
            if (total_size >= ctx->best_size) break; 
        }

        if (total_size < ctx->best_size) {
            ctx->best_size = total_size;
            memcpy(ctx->best_remap, test_remap, 256);
            memcpy(ctx->best_perm, perm, 8 * sizeof(int));
        }
        
        ctx->count++;
        if (ctx->count % 1000 == 0) {
            fprintf(stderr, "   ... evaluated %llu / 40320 candidates (best compression: %zu bytes)\r", ctx->count, ctx->best_size);
            fflush(stderr);
        }
        return;
    }

    for (int i = 0; i < k; i++) {
        generate_bit_permutations(k - 1, perm, ctx);
        int swap_idx = (k % 2 == 0) ? i : 0;
        int tmp = perm[swap_idx];
        perm[swap_idx] = perm[k - 1];
        perm[k - 1] = tmp;
    }
}

int main(int argc, char **argv) {
    int timeout = 600; 
    int dict_param_kb = 4, lc_param = 3, lp_param = 0, pb_param = 2;
    int do_reshuffle = 0, do_heuristic = 0;
    int do_normalize = 0;
    const char *seedfile_path = NULL;
    
    // Parameters for Variable Neighborhood Search
    int initial_swaps = 256;
    int max_stagnation = 1000;
    
    const char **filenames = malloc(argc * sizeof(char*));
    int file_count = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) { print_help(argv[0]); free(filenames); return EXIT_SUCCESS; }
        else if (strncmp(argv[i], "--timeout=", 10) == 0) timeout = atoi(argv[i] + 10);
        else if (strcmp(argv[i], "--reshuffle") == 0) do_reshuffle = 1;
        else if (strncmp(argv[i], "--seedfile=", 11) == 0) seedfile_path = argv[i] + 11;
        else if (strcmp(argv[i], "--heuristic") == 0) do_heuristic = 1;
        else if (strcmp(argv[i], "--normalize") == 0) do_normalize = 1;
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
    size_t max_file_size = 0; int max_filename_len = 5; 
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

    // Process external seed logic before main loop
    if (seedfile_path) {
        if (!load_seed_from_file(seedfile_path, lc_param, lp_param, pb_param, seed_remap)) {
            return EXIT_FAILURE; 
        }
    }
    
    time_t start_time = time(NULL);
    uint8_t current_remap[256];
    memcpy(current_remap, seed_remap, 256);

    // 2. Pre-Configuration Branch
    if (do_heuristic) {
        fprintf(stderr, "/* Starting Exhaustive Bit-Permutation Heuristic (40,320 candidates) */\n");
        HeuristicBitCtx ctx = { 
            .best_size = SIZE_MAX, 
            .file_count = file_count, .files = files, 
            .remapped_buf = remapped_buf, .out_buf = out_buf, 
            .out_capacity = out_capacity, 
            .lc = lc_param, .lp = lp_param, .pb = pb_param, 
            .count = 0 
        };
        int perm[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        
        generate_bit_permutations(8, perm, &ctx);
        
        fprintf(stderr, "\n/* Heuristic Selection Complete. Chosen size: %zu bytes. */\n", ctx.best_size);
        fprintf(stderr, "/* Optimal Bit Permutation Mapping: */\n/* ");
        for (int b = 0; b < 8; b++) {
            fprintf(stderr, "[Bit %d -> Bit %d] ", b, ctx.best_perm[b]);
        }
        fprintf(stderr, "*/\n\n");
        
        memcpy(current_remap, ctx.best_remap, 256);
    } else if (do_reshuffle) {
        for (int i = 255; i > 0; i--) {
            int j = rand() % (i + 1); uint8_t temp = current_remap[i]; current_remap[i] = current_remap[j]; current_remap[j] = temp;
        }
    }

    if (do_normalize) normalize_remap(current_remap, lc_param);
    
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
    int search_mode = 0; 
    
    // Memory mapping to track evaluated node pairings.
    // Dimensions: [256 elements] x [32 bytes]. 
    // Bit mapping algorithm enables fast bitwise lookup to prune redundant evaluations.
    uint8_t tested_pairs[256][32];
    memset(tested_pairs, 0, sizeof(tested_pairs));

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
            
            if (num_swaps == 1) {
                int valid_pair = 0;
                // Algorithm: Cap random pair guessing to prevent CPU stalls
                for (int retries = 0; retries < 1000; retries++) {
                    int a = rand() % 256, b = rand() % 256;
                    if (a == b) continue; 
                    
                    /**
                     * Algorithm: Chunk Compatibility
                     * Evaluates if we are in normalize mode. If so, swaps are ONLY 
                     * valid if the elements belong to different LZMA chunks.
                     */
                    if (do_normalize && ((a / chunk_size) == (b / chunk_size))) continue;
                    
                    int min_idx = a < b ? a : b;
                    int max_idx = a > b ? a : b;
                    
                    // Algorithm: Bit matrix logic for O(1) collision detection
                    if (!(tested_pairs[min_idx][max_idx >> 3] & (1 << (max_idx & 7)))) {
                        uint8_t tmp = test_remap[min_idx]; test_remap[min_idx] = test_remap[max_idx]; test_remap[max_idx] = tmp;
                        // Log pairing as tested so we never duplicate this calculation in the current era
                        tested_pairs[min_idx][max_idx >> 3] |= (1 << (max_idx & 7));
                        valid_pair = 1;
                        break;
                    }
                }
                if (!valid_pair) {
                    // Search terrain is overly saturated. Abandon random walk and default to deterministic search
                    search_mode = 1;
                    stagnation_counter = 0;
                    continue; 
                }
            } else {
                for (int k = 0; k < num_swaps; k++) {
                    int a = rand() % 256, b;
                    do { 
                        b = rand() % 256; 
                    } while (a == b || (do_normalize && ((a / chunk_size) == (b / chunk_size))));
                    uint8_t tmp = test_remap[a]; test_remap[a] = test_remap[b]; test_remap[b] = tmp;
                }
            }
            
            if (do_normalize) normalize_remap(test_remap, lc_param);
            
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
                
                // Clear the tested pairs collision mapping because the global landscape just shifted
                memset(tested_pairs, 0, sizeof(tested_pairs));
                
                // Only print the C source code structure when breaking a global record to avoid clutter
                if (is_global) print_remap_table_as_source("seed_remap", current_remap);
            } else {
                stagnation_counter++;
                if (stagnation_counter >= max_stagnation) {
                    if (num_swaps > 1) {
                        num_swaps /= 2; // Algorithm: Simulated Annealing search constriction
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
                    if (do_normalize && ((i / chunk_size) == (j / chunk_size))) continue;
                    
                    // Pruning phase utilizes bit array established during random swaps
                    if (tested_pairs[i][j >> 3] & (1 << (j & 7))) continue;
                    tested_pairs[i][j >> 3] |= (1 << (j & 7)); // Log pair state as processed
                    
                    uint8_t test_remap[256];
                    memcpy(test_remap, current_remap, 256);
                    uint8_t tmp = test_remap[i]; test_remap[i] = test_remap[j]; test_remap[j] = tmp;
                    if (do_normalize) normalize_remap(test_remap, lc_param);
                    
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
                
                memset(tested_pairs, 0, sizeof(tested_pairs)); // Reset the array on improvement jump
                
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
                if (do_normalize) normalize_remap(current_remap, lc_param);
                
                // CRITICAL: We must re-evaluate the reset state and update the file sizes 
                // so the "vs Previous" reporting starts fresh from the new baseline peak.
                size_t *reset_sizes = malloc(file_count * sizeof(size_t));
                current_sum = evaluate_remap(current_remap, file_count, files, remapped_buf, out_buf, out_capacity, dict_param_kb, lc_param, lp_param, pb_param, 0, reset_sizes);
                for (int f = 0; f < file_count; f++) files[f].current_comp_size = reset_sizes[f];
                free(reset_sizes);
                
                search_mode = 0;
                num_swaps = initial_swaps;
                stagnation_counter = 0;
                
                memset(tested_pairs, 0, sizeof(tested_pairs)); // Fresh environment clears bit state memory
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
    // FINAL EXTREME COMPRESSION EVALUATION
    // =========================================================
    fprintf(stderr, "==========================================================================\n");
    fprintf(stderr, "FINAL EXTREME COMPRESSION EVALUATION (8MB Dict, Extreme Match Finder)\n");
    fprintf(stderr, "==========================================================================\n\n");
    
    // Determine static widths for rigid table alignment 
    int w_file = max_filename_len < 7 ? 7 : max_filename_len;
    int w_abs = 20, w_id = 16, w_sd = 20, w_rem = 19;
    int w_s_abs = 16, w_s_id = 21, w_s_sd = 17;

    // Cache common baseline evaluations to prevent N * Candidates extreme compression cycles
    size_t *file_s_abs = malloc(file_count * sizeof(size_t));
    size_t *file_s_id = malloc(file_count * sizeof(size_t));
    size_t *file_s_sd = malloc(file_count * sizeof(size_t));
    size_t total_abs = 0, total_id = 0, total_sd = 0;

    for (int i = 0; i < file_count; i++) {
        file_s_abs[i] = evaluate_extreme(identity_remap, files[i].file_size, files[i].in_buf, remapped_buf, out_buf, out_capacity, 3, 0, 2);
        file_s_id[i]  = evaluate_extreme(identity_remap, files[i].file_size, files[i].in_buf, remapped_buf, out_buf, out_capacity, lc_param, lp_param, pb_param);
        
        // Critically: Evaluates strictly against the origin seed mapping (seed_remap) not intermediate heuristic states.
        file_s_sd[i]  = evaluate_extreme(seed_remap, files[i].file_size, files[i].in_buf, remapped_buf, out_buf, out_capacity, lc_param, lp_param, pb_param);
        
        total_abs += file_s_abs[i];
        total_id += file_s_id[i];
        total_sd += file_s_sd[i];
    }

    // Evaluate and sort all candidates
    FinalEval *evals = malloc((num_candidates + 1) * sizeof(FinalEval));
    for (int cand_idx = 0; cand_idx <= num_candidates; cand_idx++) {
        evals[cand_idx].original_idx = cand_idx;
        uint8_t *target_remap = (cand_idx == num_candidates) ? current_remap : candidates[cand_idx];
        memcpy(evals[cand_idx].remap, target_remap, 256);
        evals[cand_idx].file_s_rem = malloc(file_count * sizeof(size_t));
        evals[cand_idx].total_rem = 0;

        for (int i = 0; i < file_count; i++) {
            size_t s_rem = evaluate_extreme(target_remap, files[i].file_size, files[i].in_buf, remapped_buf, out_buf, out_capacity, lc_param, lp_param, pb_param);
            evals[cand_idx].file_s_rem[i] = s_rem;
            evals[cand_idx].total_rem += s_rem;
        }
    }

    // Sort evaluation block based on best overall compression (lowest total size)
    qsort(evals, num_candidates + 1, sizeof(FinalEval), cmp_eval);

    // Print sorted output
    for (int rank = 0; rank <= num_candidates; rank++) {
        int original_idx = evals[rank].original_idx;
        
        if (original_idx == num_candidates) {
            fprintf(stderr, "### Evaluation: Last Unfinished Remap");
        } else {
            fprintf(stderr, "### Evaluation: Candidate %d", original_idx + 1);
        }
        
        // Emphasize the top performing configuration
        if (rank == 0) {
            fprintf(stderr, "  [WINNER - BEST COMPRESSION RATIO]");
        }
        fprintf(stderr, "\n\n");

        fprintf(stderr, "| %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s |\n", 
                w_file, "File", w_abs, "Abs Baseline (3/0/2)", w_id, "Identity (Bytes)", 
                w_sd, "Initial Seed (Bytes)", w_rem, "Final Remap (Bytes)", 
                w_s_abs, "Savings (vs Abs)", w_s_id, "Savings (vs Identity)", w_s_sd, "Savings (vs Seed)");

        fprintf(stderr, "|-"); for(int k = 0; k < w_file; k++) fputc('-', stderr);
        fprintf(stderr, "-|-"); for(int k = 0; k < w_abs; k++) fputc('-', stderr);
        fprintf(stderr, "-|-"); for(int k = 0; k < w_id; k++) fputc('-', stderr);
        fprintf(stderr, "-|-"); for(int k = 0; k < w_sd; k++) fputc('-', stderr);
        fprintf(stderr, "-|-"); for(int k = 0; k < w_rem; k++) fputc('-', stderr);
        fprintf(stderr, "-|-"); for(int k = 0; k < w_s_abs; k++) fputc('-', stderr);
        fprintf(stderr, "-|-"); for(int k = 0; k < w_s_id; k++) fputc('-', stderr);
        fprintf(stderr, "-|-"); for(int k = 0; k < w_s_sd; k++) fputc('-', stderr);
        fprintf(stderr, "-|\n");

        size_t total_rem = evals[rank].total_rem;

        for (int i = 0; i < file_count; i++) {
            size_t s_abs = file_s_abs[i];
            size_t s_id  = file_s_id[i];
            size_t s_sd  = file_s_sd[i];
            size_t s_rem = evals[rank].file_s_rem[i];
            
            double p_abs = s_abs ? ((double)((long long)s_rem - (long long)s_abs) / (double)s_abs) * 100.0 : 0.0;
            double p_id  = s_id  ? ((double)((long long)s_rem - (long long)s_id)  / (double)s_id)  * 100.0 : 0.0;
            double p_sd  = s_sd  ? ((double)((long long)s_rem - (long long)s_sd)  / (double)s_sd)  * 100.0 : 0.0;
            
            char sav_abs_str[32], sav_id_str[32], sav_sd_str[32];
            snprintf(sav_abs_str, sizeof(sav_abs_str), "%+.2f%%", p_abs);
            snprintf(sav_id_str, sizeof(sav_id_str), "%+.2f%%", p_id);
            snprintf(sav_sd_str, sizeof(sav_sd_str), "%+.2f%%", p_sd);
            
            fprintf(stderr, "| %-*s | %*zu | %*zu | %*zu | %*zu | %*s | %*s | %*s |\n", 
                    w_file, files[i].filename, w_abs, s_abs, w_id, s_id, w_sd, s_sd, w_rem, s_rem, 
                    w_s_abs, sav_abs_str, w_s_id, sav_id_str, w_s_sd, sav_sd_str);
        }
        
        // Output Aggregate Total Data
        double tot_p_abs = total_abs ? ((double)((long long)total_rem - (long long)total_abs) / (double)total_abs) * 100.0 : 0.0;
        double tot_p_id  = total_id  ? ((double)((long long)total_rem - (long long)total_id)  / (double)total_id)  * 100.0 : 0.0;
        double tot_p_sd  = total_sd  ? ((double)((long long)total_rem - (long long)total_sd)  / (double)total_sd)  * 100.0 : 0.0;
        
        char tot_sav_abs[32], tot_sav_id[32], tot_sav_sd[32];
        snprintf(tot_sav_abs, sizeof(tot_sav_abs), "%+.2f%%", tot_p_abs);
        snprintf(tot_sav_id, sizeof(tot_sav_id), "%+.2f%%", tot_p_id);
        snprintf(tot_sav_sd, sizeof(tot_sav_sd), "%+.2f%%", tot_p_sd);

        fprintf(stderr, "| %-*s | %*zu | %*zu | %*zu | %*zu | %*s | %*s | %*s |\n\n", 
                w_file, "TOTAL", w_abs, total_abs, w_id, total_id, w_sd, total_sd, w_rem, total_rem, 
                w_s_abs, tot_sav_abs, w_s_id, tot_sav_id, w_s_sd, tot_sav_sd);
    }

    // Determine Seed Overwrite Logic Based on Final Analytics
    if (seedfile_path != NULL) {
        if (evals[0].total_rem < total_sd) {
            fprintf(stderr, "/* Best candidate (%zu bytes) empirically outperforms loaded seed (%zu bytes). Upgrading seed file... */\n", 
                    evals[0].total_rem, total_sd);
            save_new_seed_file(seedfile_path, lc_param, lp_param, pb_param, evals[0].remap);
        } else {
            fprintf(stderr, "/* Initial loaded seed remains structurally optimal against candidates. No new seed file generated. */\n");
        }
    }

    // Memory Cleanup
    for (int cand_idx = 0; cand_idx <= num_candidates; cand_idx++) free(evals[cand_idx].file_s_rem);
    free(evals);
    free(file_s_abs);
    free(file_s_id);
    free(file_s_sd);
    
    for (int i = 0; i < file_count; i++) free(files[i].in_buf);
    free(files);
    free(filenames);
    free(remapped_buf); 
    free(out_buf);
    free(candidates);
    
    return EXIT_SUCCESS;
}
