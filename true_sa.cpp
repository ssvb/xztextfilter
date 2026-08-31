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
#include <fstream>
#include <sstream>
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
 *    - Extracts a 3-byte "Data Fingerprint" from the highest frequency bytes
 *      to accurately tag saved/loaded seed states.
 * 
 * 2. State Initialization & Persistent Seeding
 *    - Establishes a true Baseline using the Identity Mapping.
 *    - Attempts to load a persistent seed matching the LZMA context parameters
 *      and the Data Fingerprint.
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
 * 
 * 6. Final Extreme Evaluation & Seed Retention
 *    - Benchmarks the final SA best state and Greedy best state using the LZMA2 
 *      "6" (or "6e" if --extreme) preset, an 8MB dictionary, and context parameters for maximum compression.
 *    - EXCLUDES the un-mutated identity baseline as a final contender.
 *    - Commits the optimal mapping back to the persistent seed file if it improves 
 *      upon the initial loaded state, prepended with comparative LZMA ratio statistics.
 * ==============================================================================
 */

// Thread-safe stdout/stderr output synchronization to prevent interleaved logs
static std::recursive_mutex stderr_mtx;

// Holds the contents of the files we are optimizing the compression for
struct FileData {
    std::string filename;
    std::vector<uint8_t> in_buf;
    size_t fast_eval_size = 0; // The calculated byte limit specifically used during the fast search phase
};

/**
 * ==============================================================================
 * InputFilesManager
 * ==============================================================================
 * Responsible for loading the source files from disk, partitioning the data for
 * fast evaluation phases, and performing all required statistical analysis.
 */
class InputFilesManager {
public:
    InputFilesManager(const std::vector<std::string>& filenames, size_t initial_limit)
        : filenames_(filenames), datasizelimit_(initial_limit), max_file_size_(0), victim_index_(0) {}

    bool loadFiles() {
        files_.resize(filenames_.size());
        for (size_t i = 0; i < filenames_.size(); i++) {
            files_[i].filename = filenames_[i];
            FILE* f = fopen(filenames_[i].c_str(), "rb");
            if (!f) return false;
            fseek(f, 0, SEEK_END);
            files_[i].in_buf.resize(ftell(f));
            fseek(f, 0, SEEK_SET);
            if (files_[i].in_buf.size() > max_file_size_) max_file_size_ = files_[i].in_buf.size();
            if (!files_[i].in_buf.empty() && fread(files_[i].in_buf.data(), 1, files_[i].in_buf.size(), f) != files_[i].in_buf.size()) {
                fclose(f);
                return false;
            }
            fclose(f);
        }
        return true;
    }

    void analyzeAndAdjustLimit() {
        apply_water_filling(datasizelimit_);

        /**
         * Algorithm: Frequency Analysis & Dynamic Victim Index Validation
         * Iterates over the byte layout of the files to construct a global frequency 
         * histogram. The index mapping to the lowest frequency becomes the "victim", 
         * serving as the continuous focal point for swaps in the SA loop.
         * 
         * Note: This analyzes the data capped by the `--datasizelimit` option. If the
         * resulting data fingerprint doesn't match the full untruncated data fingerprint,
         * it gradually increases the limit to ensure global accuracy of the core mapping.
         */
        std::array<unsigned long long, 256> full_freq{};
        for (const auto& file : files_) {
            for (uint8_t byte_val : file.in_buf) full_freq[byte_val]++;
        }
        std::string full_fingerprint = get_fingerprint(full_freq);

        size_t effective_increase = 0;

        if (datasizelimit_ > 0) {
            while (true) {
                byte_freq_.fill(0);
                for (const auto& file : files_) {
                    for (size_t i = 0; i < file.fast_eval_size; i++) {
                        byte_freq_[file.in_buf[i]]++;
                    }
                }
                std::string capped_fingerprint = get_fingerprint(byte_freq_);
                
                if (capped_fingerprint == full_fingerprint) {
                    break; // Match found
                }
                
                if (effective_increase == 0) {
                    std::lock_guard<std::recursive_mutex> lock(stderr_mtx);
                    fprintf(stderr, "/* Warning: Capped data fingerprint (%s) does not match full data fingerprint (%s). Increasing limit... */\n", capped_fingerprint.c_str(), full_fingerprint.c_str());
                }
                
                size_t total_size = 0;
                for (const auto& file : files_) total_size += file.in_buf.size();
                if (datasizelimit_ >= total_size) {
                    break; // Hard cap at maximum total file size to prevent runaway allocation
                }
                
                datasizelimit_++;
                effective_increase++;
                apply_water_filling(datasizelimit_);
            }
            
            if (effective_increase > 0) {
                std::lock_guard<std::recursive_mutex> lock(stderr_mtx);
                fprintf(stderr, "/* Effective cap increase: %zu bytes (New limit: %zu bytes) */\n", effective_increase, datasizelimit_);
            }
        } else {
            byte_freq_ = full_freq;
        }

        /**
         * Algorithm: Global Frequency Ranking, Top-10 Hex, & Fingerprinting
         * Sorts all byte values (0-255) by occurrence frequency in descending order.
         * Extracts the 10 most frequent byte values and formats them as a concatenated
         * two-digit hexadecimal string reported to stderr.
         * Also extracts the top 3 most frequent bytes to formulate the Data Fingerprint,
         * resolving and tagging the persistence entries properly.
         */
        std::vector<ByteCount> freq_ranked(256);
        for (int i = 0; i < 256; i++) {
            freq_ranked[i] = { static_cast<uint8_t>(i), byte_freq_[i] };
        }
        std::stable_sort(freq_ranked.begin(), freq_ranked.end(), [](const ByteCount& a, const ByteCount& b) {
            if (a.count != b.count) return a.count > b.count;
            return a.byte_val < b.byte_val;
        });

        char top10_hex_buf[21] = {0};
        for (int i = 0; i < 10; i++) snprintf(top10_hex_buf + (i * 2), 3, "%02X", freq_ranked[i].byte_val);

        top10_hex_ = std::string(top10_hex_buf);
        fingerprint_ = full_fingerprint;

        // Identify lowest frequency byte to act as the primary rotational pivot ("Victim Index")
        victim_index_ = 0;
        for (int i = 1; i < 256; i++) {
            if (byte_freq_[i] < byte_freq_[victim_index_]) victim_index_ = i;
        }
    }

    // Accessors for utilizing the fully prepped evaluation buffers
    const std::vector<FileData>& getFiles() const { return files_; }
    size_t getMaxFileSize() const { return max_file_size_; }
    size_t getDatasizeLimit() const { return datasizelimit_; }
    const std::string& getFingerprint() const { return fingerprint_; }
    const std::string& getTop10Hex() const { return top10_hex_; }
    int getVictimIndex() const { return victim_index_; }
    const std::array<unsigned long long, 256>& getByteFreq() const { return byte_freq_; }

private:
    struct ByteCount {
        uint8_t byte_val;
        unsigned long long count;
    };

    std::vector<std::string> filenames_;
    std::vector<FileData> files_;
    size_t datasizelimit_;
    size_t max_file_size_;

    std::array<unsigned long long, 256> byte_freq_{};
    std::string fingerprint_;
    std::string top10_hex_;
    int victim_index_;

    /**
     * ==============================================================================
     * Algorithm: Fair-Share Data Truncation (Water-Filling)
     * ==============================================================================
     * To accelerate the evaluation phases (SA and BFS), an optional data size limit 
     * can be specified. This algorithm fairly distributes that total limit across 
     * all input files. If a file is naturally smaller than its allocated fair share, 
     * its unused quota is mathematically cascaded/redistributed to remaining files.
     * 
     * 1. Sort files by their actual sizes in ascending order.
     * 2. Iteratively divide `remaining_limit` by `remaining_files` for the fair share.
     * 3. Take `min(actual_size, fair_share)` as the current file's allocation.
     * 4. Subtract the allocation from the remaining limit and proceed.
     */
    void apply_water_filling(size_t limit) {
        if (limit == 0) {
            for (auto& f : files_) f.fast_eval_size = f.in_buf.size();
            return;
        }
        std::vector<size_t> indices(files_.size());
        for (size_t i = 0; i < files_.size(); i++) indices[i] = i;
        
        // Sort indices by file buffer size in ascending order
        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            return files_[a].in_buf.size() < files_[b].in_buf.size();
        });
        
        size_t remaining_limit = limit;
        size_t remaining_files = files_.size();
        
        for (size_t idx : indices) {
            size_t fair_share = remaining_limit / remaining_files;
            size_t allocation = std::min(files_[idx].in_buf.size(), fair_share);
            
            files_[idx].fast_eval_size = allocation;
            remaining_limit -= allocation;
            remaining_files--;
        }
    }

    static std::string get_fingerprint(const std::array<unsigned long long, 256>& freqs) {
        std::vector<ByteCount> ranked(256);
        for (int i = 0; i < 256; i++) {
            ranked[i] = { static_cast<uint8_t>(i), freqs[i] };
        }
        std::stable_sort(ranked.begin(), ranked.end(), [](const ByteCount& a, const ByteCount& b) {
            if (a.count != b.count) return a.count > b.count;
            return a.byte_val < b.byte_val;
        });
        char fp[7] = {0};
        for (int i = 0; i < 3; i++) snprintf(fp + (i * 2), 3, "%02X", ranked[i].byte_val);
        return std::string(fp);
    }
};

// Thread-local evaluation workspace to prevent buffer reallocation and race conditions
struct LZMAWorkspace {
    std::vector<uint8_t> remapped_buf;
    std::vector<uint8_t> out_buf;

    explicit LZMAWorkspace(size_t max_file_size) {
        // Ensure minimum size of 1 to avoid zero-allocation edge cases
        remapped_buf.resize(std::max<size_t>(max_file_size, 1));
        
        // Calculate maximum possible output size for LZMA to avoid buffer overruns
        size_t out_capacity = lzma_stream_buffer_bound(max_file_size);
        out_buf.resize(std::max<size_t>(out_capacity, 1));
    }
};

// Typedef for our 256-byte permutation array
using RemapTable = std::array<uint8_t, 256>;

/**
 * ==============================================================================
 * Self-Sufficient DB Handler
 * ==============================================================================
 * Handles loading, saving, updating, retrieving, and formatting remap tables.
 */
class RemapDatabase {
public:
    explicit RemapDatabase(const std::string& filepath) : filepath_(filepath) {}

    // Formulate the expected array name based on LZMA parameters, extreme flag, file fingerprint, and type
    static std::string get_declaration_name(int lc, int lp, int pb, bool use_extreme, const std::string& fingerprint, bool is_decoder) {
        char target_decl[128];
        if (is_decoder) {
            snprintf(target_decl, sizeof(target_decl), "lzma_decoder_byte_remap_%d%d%d%s_%s", lc, lp, pb, use_extreme ? "e" : "", fingerprint.c_str());
        } else {
            snprintf(target_decl, sizeof(target_decl), "lzma_byte_remap_%d%d%d%s_%s", lc, lp, pb, use_extreme ? "e" : "", fingerprint.c_str());
        }
        return std::string(target_decl);
    }

    /**
     * String formatter for robustly handling unified syntax for byte declarations.
     * Prefers character literals for readable ASCII, falls back to hex for unprintable/control bytes.
     */
    static std::string format_byte(uint8_t byte) {
        char buf[16];
        if (byte >= 32 && byte <= 126) {
            if (byte == '\'') return "'\\''";
            else if (byte == '\\') return "'\\\\'";
            else { snprintf(buf, sizeof(buf), "'%c' ", byte); return buf; }
        } else {
            snprintf(buf, sizeof(buf), "0x%02x", byte);
            return buf;
        }
    }

    /**
     * Formats the remap table as a valid C source string buffer.
     * Important: We format dynamically as inverted, grouping them by calculated buckets.
     */
    static std::string format_table(const std::string& var_name, const RemapTable& remap, int lc, bool print_stats = false, double pct_a = 0.0, double pct_b = 0.0) {
        std::ostringstream oss;
        if (print_stats) {
            // Output factual accuracy update: the new percentage metrics
            char stat_buf[128];
            snprintf(stat_buf, sizeof(stat_buf), "/* %+.3f%% %+.3f%% */\n", pct_a, pct_b);
            oss << stat_buf;
        }
        oss << "unsigned char " << var_name << "[256] = {\n    ";

        /**
         * Algorithm: Output Inversion
         * Values and indexes are swapped to formulate the decoder array format 
         * for visualization and serialization.
         */
        RemapTable inverted;
        for (int i = 0; i < 256; i++) inverted[remap[i]] = i;

        for (int i = 0; i < 256; i++) {
            oss << format_byte(inverted[i]);
            
            // Append bucket identification strictly at 16-byte boundaries
            if ((i + 1) % 16 == 0) {
                int bucket = i >> (8 - lc);
                if (i < 255) oss << ",  /* bucket " << bucket << " */\n    ";
                else oss << "   /* bucket " << bucket << " */\n";
            } else {
                if (i < 255) oss << ", ";
            }
        }
        oss << "};\n";
        return oss.str();
    }

    // Rigid Seed File Parser & Permutation Validator (Supports both normal and decoder targets)
    bool load(int lc, int lp, int pb, bool use_extreme, const std::string& fingerprint, RemapTable& out_remap) const {
        if (filepath_.empty()) return false;
        FILE *file = fopen(filepath_.c_str(), "r");
        if (!file) return false; // Silently return if file does not exist

        std::string target_decl_new = get_declaration_name(lc, lp, pb, use_extreme, fingerprint, true);
        std::string target_decl_old = get_declaration_name(lc, lp, pb, use_extreme, fingerprint, false);

        char line[512];
        bool found_decl = false;
        bool is_inverted = false;

        // Scan for dynamic target array declaration (Ignores preceding comments)
        while (fgets(line, sizeof(line), file)) {
            if (strstr(line, target_decl_new.c_str()) != NULL) {
                found_decl = true;
                is_inverted = true;
                break;
            }
            if (strstr(line, target_decl_old.c_str()) != NULL) {
                found_decl = true;
                is_inverted = false;
                break;
            }
        }

        if (!found_decl) {
            fclose(file);
            return false; // Target configuration not found
        }

        // Locate the opening brace for the array payload
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
            return false; // Malformed declaration
        }

        RemapTable temp_remap;

        // Robust parsing of byte literals, supporting both hex (0x00) and char ('a', '\n') formats
        int count = 0, c;
        while (count < 256 && (c = fgetc(file)) != EOF) {
            if (c == '0') {
                int next = fgetc(file);
                if (next == 'x' || next == 'X') {
                    unsigned int val;
                    if (fscanf(file, "%2x", &val) == 1) temp_remap[count++] = (unsigned char)val;
                } else ungetc(next, file); 
            } else if (c == '\'') {
                int char_val = fgetc(file);
                if (char_val == '\\') { 
                    int escaped = fgetc(file);
                    if (escaped == '\'') char_val = '\'';
                    else if (escaped == '\\') char_val = '\\';
                }
                temp_remap[count++] = (unsigned char)char_val;
                // Consume characters until the closing quote
                while ((c = fgetc(file)) != EOF && c != '\'') { }
            } else if (c == '}') break; // End of array detected
        }
        fclose(file);

        // Validation Phase 1: Ensure exactly 256 bytes were read
        if (count != 256) {
            std::lock_guard<std::recursive_mutex> lock(stderr_mtx);
            fprintf(stderr, "Error: Parsed %d bytes, expected 256.\n", count);
            return false;
        }

        /**
         * Algorithm: Validating Permutation
         * We iterate through the parsed 256-byte array and use a frequency map (boolean array)
         * to guarantee that all numbers are in the valid range 0-255 with strictly no 
         * duplications and no skips (a complete bijective permutation of 0-255).
         */
        int seen[256] = {0};
        for (int i = 0; i < 256; i++) {
            if (seen[temp_remap[i]]) {
                std::lock_guard<std::recursive_mutex> lock(stderr_mtx);
                fprintf(stderr, "Error: DB file array contains duplicate value 0x%02X.\n", temp_remap[i]);
                return false; 
            }
            seen[temp_remap[i]] = 1;
        }

        for (int i = 0; i < 256; i++) {
            if (!seen[i]) {
                std::lock_guard<std::recursive_mutex> lock(stderr_mtx);
                fprintf(stderr, "Error: DB file array is missing value 0x%02X (skips present).\n", i);
                return false;
            }
        }

        /**
         * Algorithm: Decoding and Reverting
         * If the loaded array was in the new inverted "decoder" format, we invert 
         * it back here to strictly maintain the standard non-inverted operations 
         * for the remainder of the application.
         */
        if (is_inverted) {
            for (int i = 0; i < 256; i++) {
                out_remap[temp_remap[i]] = i;
            }
        } else {
            for (int i = 0; i < 256; i++) {
                out_remap[i] = temp_remap[i];
            }
        }

        return true;
    }

    /**
     * ==============================================================================
     * Algorithm: String-Based Replacement & Ratio Annotation (DB Upgrade)
     * ==============================================================================
     * This function updates the remap db file in-place, modifying the remap table and 
     * updating the factual accuracy of the preceding percentage block comment.
     * 
     * Crucially, it preserves ALL existing comments outside of the targeted percentage
     * block and the target array itself. 
     * 
     * 1. Scans forward to find the `lzma_byte_remap_XYZ` declaration.
     * 2. Scans backwards from the declaration to specifically identify and consume 
     *    ONLY the old / * %+XX.XXX% %+YY.YYY% * / block. It avoids touching other comments.
     * 3. Utilizes a literal-aware brace search to skip and replace the old array.
     * 4. Injects the updated percentage comment and new mapping array atomically.
     * 5. Strictly preserves all surrounding whitespace and newlines, ensuring the database
     *    format does not vertically collapse upon multiple rewrites.
     * 6. Ensures exactly one blank line is inserted between independent entries if appending.
     */
    void save(int lc, int lp, int pb, bool use_extreme, const std::string& fingerprint, 
              const RemapTable& best_remap, size_t baseline_size, size_t initial_size, size_t after_size,
              double pct_a, double pct_b) const {
        if (filepath_.empty()) return;

        std::string new_path = filepath_ + ".new";
        
        // Load entire file into a std::string to facilitate safe substring manipulation
        std::ifstream ifs(filepath_);
        std::string in_str;
        if (ifs) {
            in_str.assign((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
            ifs.close();
        }

        std::string target_decl_new = get_declaration_name(lc, lp, pb, use_extreme, fingerprint, true);
        std::string target_decl_old = get_declaration_name(lc, lp, pb, use_extreme, fingerprint, false);

        std::string new_content;
        char header[256];
        
        // Construct the updated factual representation of the percentage comment
        snprintf(header, sizeof(header), "/* %+.3f%% %+.3f%% */\nunsigned char %s[256] = {\n    ", pct_a, pct_b, target_decl_new.c_str());
        new_content += header;
        
        /**
         * Algorithm: Invert Mapping for Persistence
         * Values and indexes are swapped to formulate the decoder array format.
         */
        RemapTable inverted_remap;
        for (int i = 0; i < 256; i++) inverted_remap[best_remap[i]] = i;

        for (int i = 0; i < 256; i++) {
            new_content += format_byte(inverted_remap[i]);
            if ((i + 1) % 16 == 0) {
                int bucket = i >> (8 - lc);
                if (i < 255) new_content += ",  /* bucket " + std::to_string(bucket) + " */\n    ";
                else new_content += "   /* bucket " + std::to_string(bucket) + " */";
            } else {
                if (i < 255) new_content += ", ";
            }
        }
        
        /**
         * Algorithm: Strict Replacement Bounds
         * Prevent vertical collapse by NOT including a trailing newline in the replacement 
         * payload. We are performing an exact surgical swap from the comment's opening slash
         * to the array's closing semicolon.
         */
        new_content += "\n};";

        // Favor seeking the newer inverted declaration, fallback to resolving the older syntax
        size_t decl_pos = in_str.find(target_decl_new);
        if (decl_pos == std::string::npos) {
            decl_pos = in_str.find(target_decl_old);
        }
        
        if (decl_pos != std::string::npos) {
            // Step backwards to find the "unsigned" keyword
            size_t unsigned_pos = in_str.rfind("unsigned", decl_pos);
            if (unsigned_pos == std::string::npos) unsigned_pos = decl_pos;
            
            size_t start_replace = unsigned_pos;
            size_t search_pos = unsigned_pos;
            
            // Scrub trailing spaces before the keyword safely
            while (search_pos > 0 && std::isspace(in_str[search_pos - 1])) search_pos--;
            
            // Strictly target the `/* %+f%% %+f%% */` or older `/* float float */` pattern for replacement
            // This guarantees we only update factual accuracy of the metrics and never
            // remove existing descriptive comments the user may have left.
            if (search_pos >= 2 && in_str[search_pos - 1] == '/' && in_str[search_pos - 2] == '*') {
                size_t comment_start = in_str.rfind("/*", search_pos - 2);
                if (comment_start != std::string::npos) {
                    std::string comment_body = in_str.substr(comment_start, search_pos - comment_start);
                    
                    // Heuristic: Ensure it looks like the specific percentage or legacy float block before absorbing it
                    if (comment_body.find('.') != std::string::npos || comment_body.find('%') != std::string::npos) {
                        start_replace = comment_start;
                        // Algorithm Note: Deliberately avoided scouring preceding whitespaces/newlines 
                        // here to preserve exact block separation.
                    }
                }
            }

            // Fast-forward Literal-aware parser to find proper closing bracket of the array
            size_t end_replace = decl_pos;
            size_t open_brace = in_str.find('{', end_replace);
            if (open_brace != std::string::npos) {
                bool in_char_literal = false;
                bool in_escape = false;
                
                // Iterate safely to avoid matching braces inside character literals (e.g. '{')
                for (size_t i = open_brace; i < in_str.size(); i++) {
                    char c = in_str[i];
                    if (!in_char_literal) {
                        if (c == '\'') in_char_literal = true;
                        else if (c == '}') {
                            end_replace = i;
                            break;
                        }
                    } else {
                        if (in_escape) in_escape = false;
                        else if (c == '\\') in_escape = true;
                        else if (c == '\'') in_char_literal = false;
                    }
                }
                
                // Consume the trailing semicolon if present
                size_t semi = in_str.find(';', end_replace);
                if (semi != std::string::npos && semi - end_replace < 10) end_replace = semi + 1;
                else end_replace++;
                
                // Algorithm Note: Deliberately avoided cleaning up trailing newlines here
                // to maintain original formatting and whitespace boundaries.
            }
            
            // Execute the atomic replacement of the targeted float comment and array
            in_str.replace(start_replace, end_replace - start_replace, new_content);
        } else {
            // If the declaration wasn't found, safely append to the end of the file
            // Ensure an empty line gap between different entries.
            if (!in_str.empty()) {
                size_t trailing_newlines = 0;
                for (auto it = in_str.rbegin(); it != in_str.rend() && *it == '\n'; ++it) {
                    trailing_newlines++;
                }
                if (trailing_newlines == 0) in_str += "\n\n";
                else if (trailing_newlines == 1) in_str += "\n";
            }
            // Safely re-attach the newline during standard appending operations
            in_str += new_content + "\n";
        }

        std::ofstream out(new_path);
        if (!out) {
            std::lock_guard<std::recursive_mutex> lock(stderr_mtx);
            fprintf(stderr, "/* Error: Could not open temp file '%s' for writing. */\n", new_path.c_str());
            return;
        }
        out << in_str;
        out.close();

        // Atomic rename ensures file integrity even if the process is interrupted
        if (rename(new_path.c_str(), filepath_.c_str()) != 0) {
            std::lock_guard<std::recursive_mutex> lock(stderr_mtx);
            fprintf(stderr, "\n/* Error: Failed to atomically rename '%s' to '%s'. */\n\n", new_path.c_str(), filepath_.c_str());
        } else {
            std::lock_guard<std::recursive_mutex> lock(stderr_mtx);
            
            // Log detailed metrics to console upon successful persistence
            double pct_change_baseline = baseline_size ? ((double)((long long)after_size - (long long)baseline_size) / baseline_size) * 100.0 : 0.0;
            double pct_change_initial = initial_size ? ((double)((long long)after_size - (long long)initial_size) / initial_size) * 100.0 : 0.0;
            
            fprintf(stderr, "\n/* Successfully saved optimized configuration to DB: %s */\n", filepath_.c_str());
            fprintf(stderr, "/* Compression Improvement vs Identity Mapping : %zu bytes -> %zu bytes (%+10.3f%%) */\n", baseline_size, after_size, pct_change_baseline);
            fprintf(stderr, "/* Compression Improvement vs Initial State    : %zu bytes -> %zu bytes (%+10.3f%%) */\n\n", initial_size, after_size, pct_change_initial);
        }

        // Dump final payload mapping to console for user visibility
        std::lock_guard<std::recursive_mutex> lock(stderr_mtx);
        fprintf(stderr, "/* Saved Improved Remap Table Dump: */\n%s", 
                format_table(target_decl_new, best_remap, lc, true, pct_a, pct_b).c_str());
    }

private:
    std::string filepath_;
};

void print_help(const char* prog_name) {
    std::lock_guard<std::recursive_mutex> lock(stderr_mtx);
    std::cout << "Usage: " << prog_name << " [OPTIONS] <input_file_1> [input_file_2 ...]\n\n"
              << "Options:\n"
              << "  --help                 Show this help message and exit.\n"
              << "  --timeout=SEC          Set timeout in seconds for annealing (default: 600).\n"
              << "  --temperature=FLOAT    Set initial temperature for annealing (default: 50.0).\n"
              << "  --bfs-depth=INT        Set initial max BFS depth for Greedy Refinement (default: 1).\n"
              << "  --dict=KB              Set LZMA dictionary size in kilobytes (default: 4).\n"
              << "  --lc=BITS              Set LZMA literal context bits (1-4, default: 3).\n"
              << "  --lp=BITS              Set LZMA literal position bits (0-4, default: 0).\n"
              << "  --pb=BITS              Set LZMA position bits (0-4, default: 2).\n"
              << "  --extreme              Enable LZMA extreme compression preset during all phases.\n"
              << "  --datasizelimit=BYTES  Set data size limit for fast eval phases (distributed among files).\n"
              << "  --tablepenalty         Add a storage penalty for the remap table to the evaluation.\n"
              << "  --remapdb=FILE         Set persistent file to load/save remap database entries.\n\n";
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
    // Initialize LZMA2 options with the provided preset (usually 0 for fast eval, 6e for extreme eval)
    if (lzma_lzma_preset(&opt, preset)) return out_capacity + 1;

    // Apply strict parameter overrides for lc, lp, pb, and dictionary size
    opt.dict_size = ((uint32_t)dict_param_kb * 1024 < LZMA_DICT_SIZE_MIN) ? LZMA_DICT_SIZE_MIN : (uint32_t)dict_param_kb * 1024; 
    opt.lc = lc_param; 
    opt.lp = lp_param; 
    opt.pb = pb_param;

    lzma_filter filters[2] = { { LZMA_FILTER_LZMA2, &opt }, { LZMA_VLI_UNKNOWN, nullptr } };
    size_t out_pos = 0;
    
    // Execute encoding
    lzma_ret ret = lzma_stream_buffer_encode(filters, LZMA_CHECK_CRC32, nullptr, in_buf, in_len, out_buf, &out_pos, out_capacity);
    return (ret == LZMA_OK) ? out_pos : out_capacity + 1;
}

/**
 * Algorithm: Holistic State Evaluation & Remap Penalty
 * Translates file buffers using the provided RemapTable and evaluates overall 
 * LZMA compression size. The 'use_fast_limit' toggle dictates whether only the 
 * fair-share allocated front-chunk (fast_eval_size) of each file is evaluated, 
 * or the entirety of the file during extreme final evaluations.
 */
size_t evaluate_remap(const RemapTable& remap_table, const std::vector<FileData>& files, 
                      LZMAWorkspace& ws, uint32_t preset, int dict, int lc, int lp, int pb, 
                      bool use_table_penalty, bool use_fast_limit = false) {
    size_t total = 0;
    
    // Evaluate across all provided input files
    for (const auto& file : files) {
        size_t eval_size = use_fast_limit ? file.fast_eval_size : file.in_buf.size();
        if (eval_size == 0) continue; // Safety skip if empty allocation
        
        // Apply permutation O(N) mapping
        for (size_t j = 0; j < eval_size; j++) {
            ws.remapped_buf[j] = remap_table[file.in_buf[j]];
        }
        total += compress_buffer(ws.remapped_buf.data(), eval_size, ws.out_buf.data(), ws.out_buf.size(), preset, dict, lc, lp, pb);
    }
    
    // Optionally apply heuristic penalty to fragmented mappings
    if (use_table_penalty) {
        size_t penalty = 0;
        for (int i = 0; i < 255; i++) {
            // Adds 1 byte of penalty for every non-contiguous byte sequence
            if ((int)remap_table[i+1] - (int)remap_table[i] != 1) penalty++;
        }
        total += penalty;
    }
    
    return total;
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

    fprintf(stderr, "\nCurrent Remap Table Dump:\n%s", 
            RemapDatabase::format_table("current_decoder_remap", remap, lc).c_str());
    fprintf(stderr, "========================================================================\n\n");
}

/**
 * Algorithm: Asynchronous Greedy Worker Thread (BFS)
 * Manages concurrent local greedy exploration. Runs Breadth-First Search. 
 * Evaluates all victim-index swaps, queuing states that maintain 
 * or decrease total size. Instantly aborts and resets state when a new Annealing
 * best is found. Note: Evaluations run within this thread automatically adhere to 
 * the 'datasizelimit' (use_fast_limit = true) setting for hyper-speed searching.
 */
class GreedyWorker {
public:
    struct Task { RemapTable remap; size_t size; };
    struct BFSNode { uint64_t state; size_t size; int depth; };

    GreedyWorker(const std::vector<FileData>& files, size_t max_file_size, int victim_idx,
                 uint32_t preset, int dict, int lc, int lp, int pb, int start_bfs_depth, bool use_table_penalty)
        : files_(files), max_file_size_(max_file_size), victim_index_(victim_idx),
          preset_(preset), dict_(dict), lc_(lc), lp_(lp), pb_(pb), start_bfs_depth_(start_bfs_depth),
          use_table_penalty_(use_table_penalty),
          stop_flag_(false), new_task_flag_(false), has_best_(false),
          best_size_(std::numeric_limits<size_t>::max()) 
    {
        worker_thread_ = std::thread(&GreedyWorker::run, this);
    }

    ~GreedyWorker() { stop(); }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            stop_flag_ = true;
        }
        cv_.notify_one();
        if (worker_thread_.joinable()) worker_thread_.join();
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

    // Allows the Main SA thread to interrupt the Greedy Worker and provide it a new, 
    // better starting point to refine.
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
    // Efficiently unpacks the 64-bit state ID back into a fully realized RemapTable
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

    // Main loop for the asynchronous worker thread
    void run() {
        LZMAWorkspace ws(max_file_size_);
        while (true) {
            Task task;
            {
                std::unique_lock<std::mutex> lock(mtx_);
                // Sleep until a new best is provided or a shutdown is requested
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
            
            // Execute bounded depth Breadth-First Search on the new baseline
            run_bfs(task, ws);
        }
    }

    void run_bfs(const Task& start_task, LZMAWorkspace& ws) {
        int cumulative_depth = 0;
        int current_max_depth = start_bfs_depth_;
        Task current_root_task = start_task;
        
        // Fast-path bit array for shallow state tracking (Depths <= 3)
        std::vector<uint64_t> visited_bits((1ULL << 24) / 64, 0);

        while (true) {
again:;
            uint64_t root_state = victim_index_;
            std::queue<BFSNode> q;
            // Slower hash map utilized only for deeper branches (Depths > 3)
            std::unordered_set<uint64_t> visited_hash;
            std::fill(visited_bits.begin(), visited_bits.end(), 0);

            q.push({root_state, current_root_task.size, 0});
            uint32_t root_idx = root_state & 0xFFFFFF;
            visited_bits[root_idx >> 6] |= (1ULL << (root_idx & 63));

            uint64_t best_found_state = root_state;
            int best_found_depth = 0;
            size_t best_found_size = current_root_task.size;

            RemapTable base_remap = current_root_task.remap;

            // Exhaustive Queue Iteration
            while (!q.empty()) {
                if (new_task_flag_ || stop_flag_) return; // Bail out if interrupted

                BFSNode curr = q.front();
                
                // If maximum allowable depth is reached, assess if progress was made
                if (curr.depth >= current_max_depth) {
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
                            return; // Stop searching entirely; space exhausted
                        } else {
                            fprintf(stderr, "/* [Greedy Worker] INCREASING BFS MAX DEPTH TO %d AND RESUMING */\n", current_max_depth);
                            continue;
                        }
                    }
                }
                q.pop();
                RemapTable curr_remap = reconstruct_remap(curr.state, base_remap, victim_index_);

                // Evaluate permutations involving the victim index against all 255 other bytes
                for (int i = 0; i < 256; i++) {
                    if (i == victim_index_) continue;
                    if (new_task_flag_ || stop_flag_) return;

                    // Compute bit-packed 64-bit state identifier for cycle detection
                    uint64_t mask = (curr.depth == 0) ? 0 : ((1ULL << (curr.depth * 8)) - 1);
                    uint64_t next_state = (curr.state & mask) 
                                        | ((uint64_t)i << (curr.depth * 8)) 
                                        | ((uint64_t)victim_index_ << ((curr.depth + 1) * 8));

                    int next_depth = curr.depth + 1;
                    bool is_visited = false;

                    // O(1) Check for Depths <= 3 via bits mapping
                    if (next_depth <= 3) {
                        uint32_t idx = next_state & 0xFFFFFF;
                        uint32_t word_idx = idx >> 6;
                        uint32_t bit_idx = idx & 63;
                        is_visited = (visited_bits[word_idx] & (1ULL << bit_idx)) != 0;
                        if (!is_visited) visited_bits[word_idx] |= (1ULL << bit_idx);
                    } 
                    // Set-based Check for Depths 4-7
                    else if (next_depth <= 7) {
                        is_visited = !visited_hash.insert(next_state).second;
                    } else {
                        continue;
                    }

                    if (is_visited) continue;

                    RemapTable next_remap = curr_remap;
                    std::swap(next_remap[victim_index_], next_remap[i]);

                    // Evaluate using the fast-limit flag set to `true`
                    size_t next_size = evaluate_remap(next_remap, files_, ws, preset_, dict_, lc_, lp_, pb_, use_table_penalty_, true);

                    // Strictly Monotonic Criteria (Greedy): Only follow paths <= current state
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

                            fprintf(stderr, "\nInitial State Worker Started From:\n%s", 
                                    RemapDatabase::format_table("initial_state_decoder_remap", start_task.remap, lc_).c_str());
                            if (start_task.remap != current_root_task.remap) {
                                fprintf(stderr, "\nCurrent BFS Checkpoint:\n%s", 
                                        RemapDatabase::format_table("bfs_checkpoint_decoder_remap", current_root_task.remap, lc_).c_str());
                            }
                            fprintf(stderr, "\nCurrent Greedy Best Remap Table Dump (%d steps from start):\n%s", 
                                    cumulative_depth + next_depth, 
                                    RemapDatabase::format_table("greedy_best_decoder_remap", next_remap, lc_).c_str());
                            fprintf(stderr, "========================================================================\n\n");
                        }

                        // Track the local optimum within this specific BFS branch
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
    uint32_t preset_;
    int dict_, lc_, lp_, pb_;
    int start_bfs_depth_;
    bool use_table_penalty_;

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
    // Default tuning configuration
    int timeout = 600;
    double T_start = 50.0;
    int bfs_depth = 1;
    int dict_param_kb = 4, lc_param = 3, lp_param = 0, pb_param = 2;
    size_t datasizelimit = 0;
    bool use_table_penalty = false;
    bool use_extreme = false;
    std::string remapdb_path = "";
    std::vector<std::string> filenames;

    // Command Line Argument Parsing
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
        else if (arg.starts_with("--datasizelimit=")) datasizelimit = std::stoull(argv[i] + 16);
        else if (arg == "--tablepenalty") use_table_penalty = true;
        else if (arg == "--extreme") use_extreme = true;
        else if (arg.starts_with("--remapdb=")) remapdb_path = arg.substr(10);
        else filenames.emplace_back(argv[i]);
    }

    if (filenames.empty()) {
        std::lock_guard<std::recursive_mutex> lock(stderr_mtx);
        std::cerr << "Error: Missing input files.\n";
        return EXIT_FAILURE;
    }

    std::mt19937 rng(std::random_device{}());

    // Centralize file loading and statistical analysis in the new manager class
    InputFilesManager fileManager(filenames, datasizelimit);
    if (!fileManager.loadFiles()) {
        std::lock_guard<std::recursive_mutex> lock(stderr_mtx);
        std::cerr << "Error: Failed to load input files.\n";
        return EXIT_FAILURE;
    }

    // Executes internal fast-eval water-filling, byte frequency counts, and fingerprint generation
    fileManager.analyzeAndAdjustLimit();

    const auto& files = fileManager.getFiles();
    size_t max_file_size = fileManager.getMaxFileSize();
    int victim_index = fileManager.getVictimIndex();
    std::string fingerprint = fileManager.getFingerprint();
    const auto& byte_freq = fileManager.getByteFreq();
    
    // Limits could be auto-increased during fingerprint validation; sync state
    datasizelimit = fileManager.getDatasizeLimit(); 

    LZMAWorkspace main_ws(max_file_size);

    {
        std::lock_guard<std::recursive_mutex> lock(stderr_mtx);
        fprintf(stderr, "/* Top 10 Most Frequent Bytes (Concatenated Hex): %s */\n", fileManager.getTop10Hex().c_str());
        fprintf(stderr, "/* Data Fingerprint (Top 3 Hex): %s */\n", fingerprint.c_str());
        if (use_table_penalty) fprintf(stderr, "/* Table Penalty Enabled: 1 byte added per broken continuous sequence */\n");
        if (datasizelimit > 0) fprintf(stderr, "/* Data Size Limit: Fast evaluation capped at total %zu bytes across files */\n", datasizelimit);
        if (use_extreme) fprintf(stderr, "/* Extreme Mode Enabled: Active during all phases */\n");
    }

    // Generate strict 1:1 Identity map benchmark
    RemapTable identity_remap;
    for (int i = 0; i < 256; i++) identity_remap[i] = static_cast<uint8_t>(i);

    // Setup Presets Based on User Input (--extreme flag)
    uint32_t eval_preset = use_extreme ? (6 | LZMA_PRESET_EXTREME) : 6;
    
    // Initial phase assessments run with the `use_fast_limit = true` flag
    size_t baseline_sum = evaluate_remap(identity_remap, files, main_ws, eval_preset, dict_param_kb, lc_param, lp_param, pb_param, use_table_penalty, true);
    double safe_baseline = baseline_sum > 0 ? static_cast<double>(baseline_sum) : 1.0;

    RemapTable current_remap;
    bool db_loaded = false;
    
    RemapDatabase remap_db(remapdb_path);

    // Priority Check: Can we continue from a pre-calculated mapping in the DB?
    if (!remapdb_path.empty()) {
        if (remap_db.load(lc_param, lp_param, pb_param, use_extreme, fingerprint, current_remap)) {
            db_loaded = true;
            std::lock_guard<std::recursive_mutex> lock(stderr_mtx);
            fprintf(stderr, "/* Configuration loaded successfully from %s for %d%d%d%s_%s */\n", remapdb_path.c_str(), lc_param, lp_param, pb_param, use_extreme ? "e" : "", fingerprint.c_str());
        }
    }

    // Fallback: Total random initialization
    if (!db_loaded) {
        current_remap = identity_remap;
        std::shuffle(current_remap.begin(), current_remap.end(), rng);
    }
    
    RemapTable initial_remap = current_remap;
    size_t initial_sum = evaluate_remap(current_remap, files, main_ws, eval_preset, dict_param_kb, lc_param, lp_param, pb_param, use_table_penalty, true);
    size_t current_sum = initial_sum;
    size_t sa_best_sum = initial_sum;
    RemapTable sa_best_remap = current_remap;

    {
        std::lock_guard<std::recursive_mutex> lock(stderr_mtx);
        fprintf(stderr, "/* Simulated Annealing Initialized */\n");
        fprintf(stderr, "Victim Index   : 0x%02X (Frequency: %llu bytes)\n", victim_index, byte_freq[victim_index]);
        fprintf(stderr, "Initial Temp   : %.2f\n", T_start);
        fprintf(stderr, "Baseline Size  : %zu bytes (Identity)\n", baseline_sum);
        fprintf(stderr, "Initial Size   : %zu bytes (%s)\n\n", initial_sum, db_loaded ? "DB Entry Loaded" : "Shuffled");
    }

    // Spin up Greedy BFS worker to operate asynchronously from the current baseline
    GreedyWorker greedy_worker(files, max_file_size, victim_index, eval_preset, dict_param_kb, lc_param, lp_param, pb_param, bfs_depth, use_table_penalty);
    greedy_worker.notify_new_annealing_best(sa_best_remap, sa_best_sum);

    auto start_time = std::chrono::steady_clock::now();
    auto last_report_time = start_time;
    unsigned long long iterations = 0;
    
    // Limits console IO by batched reporting
    double report_delay = static_cast<double>(timeout) / 100.0;

    bool has_pending_report = false;
    RemapTable pending_remap;
    unsigned long long pending_iter = 0;
    double pending_elapsed = 0.0;
    size_t pending_new_size = 0, pending_prev_size = 0;
    double pending_T = 0.0; 

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

        if (elapsed >= timeout) break; // Hard exit on timeout
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
        do { swap_idx = dist256(rng); } while (swap_idx == victim_index);

        RemapTable test_remap = current_remap;
        std::swap(test_remap[victim_index], test_remap[swap_idx]);

        // Assess candidate mapping logic (Fast Limit: Enabled)
        size_t test_sum = evaluate_remap(test_remap, files, main_ws, eval_preset, dict_param_kb, lc_param, lp_param, pb_param, use_table_penalty, true);

        /**
         * Algorithm: Normalized Energy Delta Calculation
         * Evaluates relative change in compressed size rather than absolute bytes.
         * Energy E = (compressed_size / baseline_size) * 100000
         */
        double delta = (((double)test_sum - (double)current_sum) / safe_baseline) * 100000.0;

        // Metropolis Criteria 1: Absolute Enhancement or Equal Tie (Accept Unconditionally)
        if (delta <= 0.0) {
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

            // Lock in progression
            current_sum = test_sum;
            current_remap = test_remap;

            // Track SA best and notify worker if new best is found
            if (current_sum < sa_best_sum) {
                sa_best_sum = current_sum;
                sa_best_remap = current_remap;
                
                // If this crushes the greedy worker's best, synchronize and reset the worker
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
    // FINAL EVALUATION (Rigorous Settings) & PERSISTENT SAVE
    // =========================================================
    /**
     * Algorithm: Absolute Performance Benchmarking
     * Benchmarks the last Annealing best, the last Greedy best, AND the unmodified 
     * identity baseline mappings through a high-cost LZMA compression sequence 
     * to guarantee that whichever output is committed fundamentally provides the 
     * strongest global compression.
     * 
     * Note: Fast Limit is FALSE. It will benchmark against complete untruncated files.
     */
    {
        std::lock_guard<std::recursive_mutex> lock(stderr_mtx);
        fprintf(stderr, "========================================================================\n");
        fprintf(stderr, "FINAL EVALUATION (Preset: %s, Dictionary: 8MB, Full Buffer Pass)\n", use_extreme ? "6e" : "6");
        fprintf(stderr, "========================================================================\n");
    }

    // Rigorous evaluation settings mimicking deployment scenarios
    uint32_t final_preset = use_extreme ? (6 | LZMA_PRESET_EXTREME) : 6;
    int final_dict_kb = 8192; // 8MB

    // Fetch the metrics specifically requested for the factual percentage comments
    size_t final_baseline = evaluate_remap(identity_remap, files, main_ws, final_preset, final_dict_kb, lc_param, lp_param, pb_param, use_table_penalty, false);
    size_t final_baseline_default = evaluate_remap(identity_remap, files, main_ws, final_preset, final_dict_kb, 3, 0, 2, use_table_penalty, false);
    
    size_t final_initial_seed = evaluate_remap(initial_remap, files, main_ws, final_preset, final_dict_kb, lc_param, lp_param, pb_param, use_table_penalty, false);
    size_t final_sa_best = evaluate_remap(sa_best_remap, files, main_ws, final_preset, final_dict_kb, lc_param, lp_param, pb_param, use_table_penalty, false);
    
    double sa_pct_baseline = final_baseline ? ((double)((long long)final_sa_best - (long long)final_baseline) / final_baseline) * 100.0 : 0.0;
    double sa_pct_initial = final_initial_seed ? ((double)((long long)final_sa_best - (long long)final_initial_seed) / final_initial_seed) * 100.0 : 0.0;

    {
        std::lock_guard<std::recursive_mutex> lock(stderr_mtx);
        fprintf(stderr, "Identity Remap Baseline Size : %zu bytes\n", final_baseline);
        fprintf(stderr, "Initial DB Entry Size        : %zu bytes\n", final_initial_seed);
        fprintf(stderr, "Last Annealing Best Size     : %zu bytes (%+10.3f%% vs Identity, %+10.3f%% vs DB Entry)\n", final_sa_best, sa_pct_baseline, sa_pct_initial);
    }

    // Query Asynchronous BFS results
    RemapTable greedy_best_remap;
    size_t greedy_fast_best = 0;
    bool has_greedy_best = greedy_worker.get_best(greedy_best_remap, greedy_fast_best);
    
    RemapTable final_optimal_remap = sa_best_remap;
    size_t final_optimal_size = final_sa_best;
    const char* optimal_source_name = "Annealing Best";

    // Compare SA vs Greedy yields 
    if (has_greedy_best) {
        size_t final_greedy_best = evaluate_remap(greedy_best_remap, files, main_ws, final_preset, final_dict_kb, lc_param, lp_param, pb_param, use_table_penalty, false);
        double greedy_pct_baseline = final_baseline ? ((double)((long long)final_greedy_best - (long long)final_baseline) / final_baseline) * 100.0 : 0.0;
        double greedy_pct_initial = final_initial_seed ? ((double)((long long)final_greedy_best - (long long)final_initial_seed) / final_initial_seed) * 100.0 : 0.0;

        std::lock_guard<std::recursive_mutex> lock(stderr_mtx);
        fprintf(stderr, "Greedy Worker Best Size      : %zu bytes (%+10.3f%% vs Identity, %+10.3f%% vs DB Entry)\n", final_greedy_best, greedy_pct_baseline, greedy_pct_initial);

        // Elect greedy mapping if strictly better
        if (final_greedy_best < final_optimal_size) {
            final_optimal_remap = greedy_best_remap;
            final_optimal_size = final_greedy_best;
            optimal_source_name = "Greedy Worker Best";
        }
    }

    // Note: Identity Mapping is explicitly REMOVED from contention here.
    // The optimal size MUST be the best valid configuration determined by the heuristic passes (SA or Greedy).

    // Calculate Final Performance Deltas
    double opt_pct_baseline = final_baseline ? ((double)((long long)final_optimal_size - (long long)final_baseline) / final_baseline) * 100.0 : 0.0;
    double opt_pct_initial = final_initial_seed ? ((double)((long long)final_optimal_size - (long long)final_initial_seed) / final_initial_seed) * 100.0 : 0.0;

    double pct_a = final_baseline > 0 ? (((double)final_optimal_size - final_baseline) / final_baseline * 100.0) : 0.0;
    double pct_b = final_baseline_default > 0 ? (((double)final_optimal_size - final_baseline_default) / final_baseline_default * 100.0) : 0.0;

    {
        std::lock_guard<std::recursive_mutex> lock(stderr_mtx);
        fprintf(stderr, "\n========================================================================\n");
        fprintf(stderr, "FINAL OPTIMAL REMAP SOURCE: %s\n", optimal_source_name);
        fprintf(stderr, "Final Size                  : %zu bytes\n", final_optimal_size);
        fprintf(stderr, "Total Reduction vs Identity : %+10.3f%%\n", opt_pct_baseline);
        fprintf(stderr, "Total Reduction vs Initial  : %+10.3f%%\n", opt_pct_initial);
        fprintf(stderr, "Difference vs Context Ident.: %+.3f%%\n", pct_a);
        fprintf(stderr, "Difference vs Default Ident.: %+.3f%%\n", pct_b);
        fprintf(stderr, "========================================================================\n\n");
    }

    // Gatekeeper metric: Only physically overwrite the previous save if we actively improved it
    bool improvement_found = (final_optimal_size < final_initial_seed);

    if (improvement_found && !remapdb_path.empty()) {
        remap_db.save(lc_param, lp_param, pb_param, use_extreme, fingerprint, 
                      final_optimal_remap, final_baseline, final_initial_seed, final_optimal_size,
                      pct_a, pct_b);
    } else {
        std::lock_guard<std::recursive_mutex> lock(stderr_mtx);
        if (!improvement_found) fprintf(stderr, "/* No meaningful improvement vs initial state found; skipping save. */\n");
        else fprintf(stderr, "/* No DB file provided; skipping save. */\n");
        
        std::string final_target_decl = RemapDatabase::get_declaration_name(lc_param, lp_param, pb_param, use_extreme, fingerprint, true);
        
        fprintf(stderr, "\nFinal Remap Table Dump (Optimal Configuration):\n%s", 
                RemapDatabase::format_table(final_target_decl, final_optimal_remap, lc_param, true, pct_a, pct_b).c_str());
    }

    return EXIT_SUCCESS;
}
