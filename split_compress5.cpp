#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <lzma.h>

int min_threshold = 35;

std::vector<int> findOptimalArrangementIndices(const std::vector<uint8_t>& arr, uint8_t M, int W) {
    int n = arr.size();
    if (n == 0) {
        return {};
    }

    // dp[i] stores {count, sum} assuming arr[i] is the LAST chosen element
    std::vector<std::pair<int, long long>> dp(n, {0, 0LL});
    
    // prev_chosen[i] stores the index of the previous element we transitioned from
    std::vector<int> prev_chosen(n, -1);
    
    // max_idx_up_to[i] stores the index (<= i) that has the best {count, sum} so far
    std::vector<int> max_idx_up_to(n, -1);

    for (int i = 0; i < n; ++i) {
        if (arr[i] >= M) {
            // Can we connect it to a previous valid sequence?
            if (i >= W && max_idx_up_to[i - W] != -1) {
                int prev_best_idx = max_idx_up_to[i - W];
                dp[i] = {dp[prev_best_idx].first + 1, dp[prev_best_idx].second + arr[i]};
                prev_chosen[i] = prev_best_idx; // Link backwards
            } else {
                // Start a new sequence
                dp[i] = {1, static_cast<long long>(arr[i])};
                prev_chosen[i] = -1;
            }
        }

        // Maintain the index of the best DP state seen up to index i
        if (i == 0) {
            max_idx_up_to[i] = (arr[i] >= M) ? 0 : -1;
        } else {
            if (max_idx_up_to[i - 1] == -1) {
                // If there were no valid elements before, check if current is valid
                max_idx_up_to[i] = (dp[i].first > 0) ? i : -1;
            } else {
                // Compare current dp state with the best seen previously
                // std::pair operator> compares .first (count), then .second (sum)
                if (dp[i] > dp[max_idx_up_to[i - 1]]) {
                    max_idx_up_to[i] = i;
                } else {
                    max_idx_up_to[i] = max_idx_up_to[i - 1];
                }
            }
        }
    }

    // The index containing the absolute optimal ending element
    int best_last_idx = max_idx_up_to.back();
    
    // If no elements met the threshold M
    if (best_last_idx == -1) {
        return {}; 
    }

    // Reconstruct the path by following the prev_chosen links backwards
    std::vector<int> optimal_indices;
    int curr = best_last_idx;
    while (curr != -1) {
        optimal_indices.push_back(curr);
        curr = prev_chosen[curr];
    }

    // The path was constructed backwards, so reverse it
    std::reverse(optimal_indices.begin(), optimal_indices.end());
    
    return optimal_indices;
}

// The maximum number of chunks that are assumed to be sufficient
// for the range coder to fully adjust the probability counters
#define MAX_MODEL_CHUNK_DEPTH 15

typedef struct { int lc, lp, pb; } lzma_params;

// All possible valid lc/lp/pb configurations
const lzma_params all_valid_lc_lp_pb[75] = {
    { 0, 0, 0 }, { 0, 0, 1 }, { 0, 0, 2 }, { 0, 0, 3 }, { 0, 0, 4 },
    { 0, 1, 0 }, { 0, 1, 1 }, { 0, 1, 2 }, { 0, 1, 3 }, { 0, 1, 4 },
    { 0, 2, 0 }, { 0, 2, 1 }, { 0, 2, 2 }, { 0, 2, 3 }, { 0, 2, 4 },
    { 0, 3, 0 }, { 0, 3, 1 }, { 0, 3, 2 }, { 0, 3, 3 }, { 0, 3, 4 },
    { 0, 4, 0 }, { 0, 4, 1 }, { 0, 4, 2 }, { 0, 4, 3 }, { 0, 4, 4 },
    { 1, 0, 0 }, { 1, 0, 1 }, { 1, 0, 2 }, { 1, 0, 3 }, { 1, 0, 4 },
    { 1, 1, 0 }, { 1, 1, 1 }, { 1, 1, 2 }, { 1, 1, 3 }, { 1, 1, 4 },
    { 1, 2, 0 }, { 1, 2, 1 }, { 1, 2, 2 }, { 1, 2, 3 }, { 1, 2, 4 },
    { 1, 3, 0 }, { 1, 3, 1 }, { 1, 3, 2 }, { 1, 3, 3 }, { 1, 3, 4 },
    { 2, 0, 0 }, { 2, 0, 1 }, { 2, 0, 2 }, { 2, 0, 3 }, { 2, 0, 4 },
    { 2, 1, 0 }, { 2, 1, 1 }, { 2, 1, 2 }, { 2, 1, 3 }, { 2, 1, 4 },
    { 2, 2, 0 }, { 2, 2, 1 }, { 2, 2, 2 }, { 2, 2, 3 }, { 2, 2, 4 },
    { 3, 0, 0 }, { 3, 0, 1 }, { 3, 0, 2 }, { 3, 0, 3 }, { 3, 0, 4 },
    { 3, 1, 0 }, { 3, 1, 1 }, { 3, 1, 2 }, { 3, 1, 3 }, { 3, 1, 4 },
    { 4, 0, 0 }, { 4, 0, 1 }, { 4, 0, 2 }, { 4, 0, 3 }, { 4, 0, 4 }
};

const lzma_params common_lc_lp_pb[] = {
    { 0, 0, 0 }, { 2, 2, 2 }, { 3, 0, 2 }, { 4, 0, 0 }, { 3, 1, 1 }
};

// O(1) amortized tracking for the sum of squares of differences
// Used to continuously recalculate rolling variance/distances without O(N) loops per step.
inline void update_diff(int& diff, long long& sum_sq, int delta) {
    sum_sq += 2LL * diff * delta + static_cast<long long>(delta) * delta;
    diff += delta;
}

// ---------------------------------------------------------
// Partitioner Data Structures & Classes
// ---------------------------------------------------------

struct AnnotatedChunk {
    size_t offset;
    size_t length;
    std::string annotation;
};

class Partitioner {
protected:
    const std::vector<uint8_t>& data;
    size_t window_size;
    size_t smallest_chunk;
    size_t filter_radius;
    bool do_debug;

    mutable std::vector<uint8_t> cached_dist_results;
    mutable bool has_cached_dist_results = false;

    // Removed the redundant base implementation of calculate_dist_results(). 
    // It was overridden by the child class and never called.
    virtual std::vector<uint8_t> calculate_dist_results() const = 0;

    // Applies a moving average filter to smooth out volatile byte distance spikes
    // over a specified radius, reducing over-segmentation.
    std::vector<uint8_t> smooth_distances(const std::vector<uint8_t>& dist_results) const {
        size_t len = dist_results.size();
        std::vector<uint8_t> smoothed_dist(len, 0);
        if (len == 0) return smoothed_dist;

        long long current_sum = 0;
        size_t current_count = 0;
        
        // Initial window population
        for (size_t j = 0; j <= filter_radius && j < len; ++j) {
            current_sum += dist_results[j];
            current_count++;
        }
        smoothed_dist[0] = static_cast<double>(current_sum) / current_count;

        // Slide the window across the data
        for (size_t i = 1; i < len; ++i) {
            if (i + filter_radius < len) {
                current_sum += dist_results[i + filter_radius];
                current_count++;
            }
            if (i > filter_radius) {
                current_sum -= dist_results[i - filter_radius - 1];
                current_count--;
            }
            smoothed_dist[i] = static_cast<double>(current_sum) / current_count;
        }

        return smoothed_dist;
    }

public:
    Partitioner(const std::vector<uint8_t>& input_data, size_t win_size, size_t min_chunk, size_t radius, bool debug)
        : data(input_data), window_size(win_size), smallest_chunk(min_chunk), filter_radius(radius), do_debug(debug) {}

    virtual ~Partitioner() = default;

    virtual std::vector<uint8_t> get_distances() const {
        return calculate_dist_results();
    }

    virtual std::vector<AnnotatedChunk> partition() const = 0;

    virtual std::string generate_annotation(size_t offset, size_t length) const = 0; 
};

class PartitionerFreq : public Partitioner {
private:
    size_t lc;
    size_t max_chunk;

    struct ByteFreq {
        int count;
        uint8_t val;
        // Sort primarily by descending frequency, then by ascending byte value
        bool operator<(const ByteFreq& other) const {
            if (count != other.count) return count > other.count;
            return val < other.val;
        }
    };

    // Computes heuristic "distances" marking shift boundaries in the file's data distribution.
    // Highly distinct blocks will peak in distance, creating prime split candidates.
    std::vector<uint8_t> calculate_dist_results() const override {
        if (has_cached_dist_results) return cached_dist_results;

        size_t N = data.size();
        std::vector<uint8_t> dist_results(N + 1, 0);

        if (N == 0) return dist_results;

        size_t W = window_size;
        size_t pad_len = W;
        size_t padded_size = N + 2 * pad_len;

        // Create a mirror-padded version of the data array to safely run the rolling 
        // window over edge boundaries without out-of-bounds checks inside the core loop.
        std::vector<uint8_t> P(padded_size);
        for (size_t i = 0; i < padded_size; ++i) {
            int64_t v = static_cast<int64_t>(i) - static_cast<int64_t>(pad_len);
            int64_t orig_idx;
            if (v < 0) {
                int64_t mirror = -1 - v;
                orig_idx = mirror % static_cast<int64_t>(N);
            } else if (v >= static_cast<int64_t>(N)) {
                int64_t mirror = 2 * static_cast<int64_t>(N) - 1 - v;
                if (mirror < 0) {
                    int64_t rem = (-mirror) % static_cast<int64_t>(N);
                    orig_idx = (rem == 0) ? 0 : (static_cast<int64_t>(N) - rem);
                } else {
                    orig_idx = mirror % static_cast<int64_t>(N);
                }
            } else {
                orig_idx = v;
            }
            P[i] = data[orig_idx];
        }

        // Pair combination index based on local compression (lc) configurations
        size_t num_pairs = 1ULL << (lc + 8);
        auto get_pair_idx = [&](size_t pos) -> size_t {
            uint8_t b1 = P[pos];
            uint8_t b2 = P[pos + 1];

            int64_t v = static_cast<int64_t>(pos) - static_cast<int64_t>(pad_len);
            bool is_reflected = (v < 0) || (v >= static_cast<int64_t>(N) - 1);

            uint8_t first_byte = is_reflected ? b2 : b1;
            uint8_t second_byte = is_reflected ? b1 : b2;

            uint32_t b1_hi = (lc == 0) ? 0 : (first_byte >> (8 - lc));
            return (b1_hi << 8) | second_byte;
        };

        std::vector<int> diff_p(num_pairs, 0);
        int diff_b[256] = {0};
        long long sum_sq_b = 0;
        long long sum_sq_p = 0;

        // Precalculate maximum possible distributions to scale outputs to [0, 255]
        double max_sq_sum_b = 2.0 * static_cast<double>(W) * W;
        double max_sq_sum_p = (W > 1) ? (2.0 * static_cast<double>(W - 1) * (W - 1)) : 1.0;

        double kb = 32768.0 / W;
        double kp = (W > 1) ? (32768.0 / (W - 1)) : 32768.0;
        double kb_sq = kb * kb;
        double kp_sq = kp * kp;

        double max_X = (max_sq_sum_b * kb_sq) + (max_sq_sum_p * kp_sq);
        double max_dist = std::sqrt(max_X);

        // Bootstrap the rolling window
        for (size_t i = 0; i < W; ++i) {
            uint8_t a_b = P[i];
            uint8_t b_b = P[pad_len + i];
            update_diff(diff_b[a_b], sum_sq_b, 1);
            update_diff(diff_b[b_b], sum_sq_b, -1);

            if (i + 1 < W) {
                size_t a_p = get_pair_idx(i);
                size_t b_p = get_pair_idx(pad_len + i);
                update_diff(diff_p[a_p], sum_sq_p, 1);
                update_diff(diff_p[b_p], sum_sq_p, -1);
            }
        }

        // Slide the window over the file to calculate exact differences in distribution
        for (size_t offset = pad_len; offset <= pad_len + N; ++offset) {
            double X_b = sum_sq_b * kb_sq;
            double X_p = sum_sq_p * kp_sq;
            double dist = std::sqrt(X_b + X_p);

            double normalized_dist = (dist / max_dist) * 255.0;
            dist_results[offset - pad_len] = static_cast<uint8_t>(std::min(255.0, std::max(0.0, std::round(normalized_dist))));

            if (offset < pad_len + N) {
                // Eject the trailing byte and inject the leading byte
                update_diff(diff_b[P[offset - W]], sum_sq_b, -1);
                update_diff(diff_b[P[offset]], sum_sq_b, 1);

                if (W > 1) {
                    size_t a_out_p = get_pair_idx(offset - W);
                    size_t a_in_p = get_pair_idx(offset - 1);
                    update_diff(diff_p[a_out_p], sum_sq_p, -1);
                    update_diff(diff_p[a_in_p], sum_sq_p, 1);
                }

                // Balance the rolling sum logic for the leading boundary
                update_diff(diff_b[P[offset]], sum_sq_b, 1);
                update_diff(diff_b[P[offset + W]], sum_sq_b, -1);

                if (W > 1) {
                    size_t b_out_p = get_pair_idx(offset);
                    size_t b_in_p = get_pair_idx(offset + W - 1);
                    update_diff(diff_p[b_out_p], sum_sq_p, 1);
                    update_diff(diff_p[b_in_p], sum_sq_p, -1);
                }
            }
        }

        cached_dist_results = dist_results;
        has_cached_dist_results = true;
        return dist_results;
    }

public:
    PartitionerFreq(const std::vector<uint8_t>& input_data,
                    size_t win_size,
                    size_t min_chunk,
                    size_t max_c,
                    size_t lc_bits,
                    size_t radius = 32,
                    bool debug = false)
        : Partitioner(input_data, win_size, min_chunk, radius, debug),
          lc(std::min<size_t>(lc_bits, 8)),
          max_chunk(std::max<size_t>(max_c, min_chunk)) {}

    // Segments the data at points of maximum statistical difference
    std::vector<AnnotatedChunk> partition() const override {
        std::vector<AnnotatedChunk> chunks;
        size_t N = data.size();
        if (N == 0) return chunks;

        auto dist_results = calculate_dist_results();
        auto smoothed_dist = smooth_distances(dist_results);

        auto part_results = findOptimalArrangementIndices(dist_results, min_threshold, smallest_chunk);
        part_results.push_back(dist_results.size());

        for (int i = 0; i < part_results.size(); i++) {
            AnnotatedChunk ch;
            ch.offset = part_results[i];
            ch.length = part_results[i] - (i > 0 ? part_results[i - 1] : 0);
            ch.annotation = generate_annotation(ch.offset, ch.length);
            chunks.push_back(ch);
        }
        return chunks;
    }

    // Creates an annotation based on the hex values of the 8 most frequent bytes in a chunk
    std::string generate_annotation(size_t offset, size_t length) const override {
        if (length == 0) return "";

        int b_counts[256] = {0};
        for (size_t i = 0; i < length; ++i) {
            b_counts[data[offset + i]]++;
        }

        std::vector<ByteFreq> freqs;
        freqs.reserve(256);
        for (int i = 0; i < 256; ++i) {
            if (b_counts[i] > 0) {
                freqs.push_back({b_counts[i], static_cast<uint8_t>(i)});
            }
        }

        size_t num_top = std::min<size_t>(8, freqs.size());
        if (num_top > 0) {
            std::partial_sort(freqs.begin(), freqs.begin() + num_top, freqs.end());
        }

        std::ostringstream oss;
        oss << std::hex << std::setfill('0');
        for (size_t i = 0; i < num_top; ++i) {
            oss << std::setw(2) << static_cast<int>(freqs[i].val);
        }
        return oss.str();
    }
};

struct ChunkDef {
    size_t size;
    std::string annotation;
    int lc;
    int lp;
    int pb;
};

// Buffer-to-buffer compression utilizing liblzma, updating stream filters incrementally
std::vector<size_t> compress_buffer(const std::vector<uint8_t>& in_data,
                                    std::vector<uint8_t>& out_data,
                                    const std::vector<ChunkDef>& chunks,
                                    bool is_raw) {
    lzma_options_lzma opt;
    lzma_lzma_preset(&opt, LZMA_PRESET_DEFAULT);
    
    // Configure initial parameters from the first chunk or fallback to default
    if (!chunks.empty()) {
        opt.lc = chunks[0].lc;
        opt.lp = chunks[0].lp;
        opt.pb = chunks[0].pb;
    } else {
        opt.lc = 4;
        opt.lp = 0;
        opt.pb = 0;
    }

    lzma_filter filters[2] = {
        { LZMA_FILTER_LZMA2, &opt },
        { LZMA_VLI_UNKNOWN, nullptr }
    };

    lzma_stream strm = LZMA_STREAM_INIT;
    lzma_ret ret;

    // Use raw or standard .xz formatting for encoding
    if (is_raw) {
        ret = lzma_raw_encoder(&strm, filters);
    } else {
        ret = lzma_stream_encoder(&strm, filters, LZMA_CHECK_CRC64);
    }

    if (ret != LZMA_OK) {
        std::cerr << "Error initializing lzma encoder: " << ret << "\n";
        return {};
    }

    std::vector<uint8_t> out_buf(16384);
    out_data.clear();
    out_data.reserve(in_data.size() / 2); // Pre-allocate with an assumed average compression ratio

    // Generic worker lambda to consume input and push to out_data
    auto run_lzma = [&](lzma_action action) -> bool {
        do {
            strm.next_out = out_buf.data();
            strm.avail_out = out_buf.size();
            ret = lzma_code(&strm, action);
            
            if (ret != LZMA_OK && ret != LZMA_STREAM_END) {
                std::cerr << "Compression error code: " << ret << "\n";
                return false;
            }
            
            size_t produced = out_buf.size() - strm.avail_out;
            if (produced > 0) {
                out_data.insert(out_data.end(), out_buf.begin(), out_buf.begin() + produced);
            }
        } while (strm.avail_out == 0);
        return true;
    };

    size_t in_pos = 0;
    std::vector<size_t> compressed_sizes;
    size_t current_chunk_start_size = out_data.size();

    // If no multi-chunk definitions exist, process the entire buffer natively
    if (chunks.empty()) {
        strm.next_in = in_data.data();
        strm.avail_in = in_data.size();
        if (!run_lzma(LZMA_RUN)) return {};
        in_pos = in_data.size();
        
        // Finalize stream
        strm.avail_in = 0;
        if (!run_lzma(LZMA_FINISH)) return {};
        compressed_sizes.push_back(out_data.size() - current_chunk_start_size);
    } else {
        for (size_t i = 0; i < chunks.size(); ++i) {
            if (i > 0) {
                // Must ensure previous chunks are thoroughly flushed before altering stream properties
                strm.avail_in = 0;
                if (!run_lzma(LZMA_SYNC_FLUSH)) return {};

                // Record resulting flushed size before beginning tracking on the new chunk
                compressed_sizes.push_back(out_data.size() - current_chunk_start_size);
                current_chunk_start_size = out_data.size();

                // Swap LZMA filter parameters to uniquely fit the approaching chunk
                opt.lc = chunks[i].lc;
                opt.lp = chunks[i].lp;
                opt.pb = chunks[i].pb;
                filters[0].options = &opt;

                ret = lzma_filters_update(&strm, filters);
                if (ret != LZMA_OK) {
                    std::cerr << "Failed to update filters mid-stream: " << ret << "\n";
                    return {};
                }
            }

            // Cap the input by either the remainder of the dataset, or the defined constraint 
            size_t chunk_size = chunks[i].size;
            size_t bytes_to_process = std::min(chunk_size, in_data.size() - in_pos);

            if (bytes_to_process > 0) {
                strm.next_in = in_data.data() + in_pos;
                strm.avail_in = bytes_to_process;
                if (!run_lzma(LZMA_RUN)) return {};
                in_pos += bytes_to_process;
            }

            if (in_pos >= in_data.size()) break; // End early if we exhaust the file buffer 
        }

        // For inputs outlasting defined configurations, process the remaining tail optimally
        if (in_pos < in_data.size()) {
            strm.next_in = in_data.data() + in_pos;
            strm.avail_in = in_data.size() - in_pos;
            if (!run_lzma(LZMA_RUN)) return {};
        }

        // Finalize stream natively 
        strm.avail_in = 0;
        if (!run_lzma(LZMA_FINISH)) return {};
        compressed_sizes.push_back(out_data.size() - current_chunk_start_size);
    }

    lzma_end(&strm);
    return compressed_sizes;
}

int main(int argc, char** argv) {
    bool is_raw = false;
    bool autochunks = true;
    std::string input_path;
    std::string output_path;
    std::vector<ChunkDef> chunks;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--raw") {
            is_raw = true;
        } else if (input_path.empty()) {
            input_path = arg;
        } else if (output_path.empty()) {
            output_path = arg;
        } else {
            std::cerr << "Unexpected argument: " << arg << "\n";
            return 1;
        }
    }

    if (input_path.empty() || output_path.empty()) {
        std::cerr << "Usage: " << argv[0] << " [--raw] <input_file> <output_file>\n";
        return 1;
    }

    // Read the entire file into an input buffer
    std::ifstream in(input_path, std::ios::binary | std::ios::ate);
    if (!in) {
        std::cerr << "Failed to open input file.\n";
        return 1;
    }
    std::streamsize file_size = in.tellg();
    in.seekg(0, std::ios::beg);
    
    std::vector<uint8_t> in_data;
    if (file_size > 0) {
        in_data.resize(file_size);
        in.read(reinterpret_cast<char*>(in_data.data()), file_size);
    }

    if (autochunks) {
      PartitionerFreq pf(in_data,
                    1024,
                    512,
                    1024 * 1024,
                    8,
                    4,
                    true);
      auto ac = pf.partition();
      chunks.clear();
      for (int i = 0; i < ac.size(); i++) {
        ChunkDef cd{};
        cd.size = ac[i].length;
        cd.annotation = ac[i].annotation;
        chunks.push_back(cd);
      }
    }

    std::vector<uint8_t> out_data;
    std::vector<size_t> chunk_sizes = compress_buffer(in_data, out_data, chunks, is_raw);
    
    if (chunk_sizes.empty() && !in_data.empty()) {
        std::cerr << "Compression failed.\n";
        return 1;
    }

    // Print resulting compressed chunk sizes directly to stderr
    for (size_t i = 0; i < chunk_sizes.size(); ++i) {
        std::cerr << "Chunk " << i << " fingerprint: " << chunks[i].annotation << ", original size: " << chunks[i].size << ", compressed size: " << chunk_sizes[i] << " bytes\n";
    }

    // Write resulting buffer natively to the requested output file 
    std::ofstream out(output_path, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open output file.\n";
        return 1;
    }
    if (!out_data.empty()) {
        out.write(reinterpret_cast<char*>(out_data.data()), out_data.size());
    }

    std::cout << "Successfully compressed (" << (is_raw ? "raw LZMA2" : ".xz container") << ")!\n";
    return 0;
}
