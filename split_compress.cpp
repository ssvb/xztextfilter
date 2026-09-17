#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <lzma.h>

int main(int argc, char** argv) {
    bool is_raw = false;
    std::string input_path;
    std::string output_path;

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

    std::ifstream in(input_path, std::ios::binary | std::ios::ate);
    if (!in) {
        std::cerr << "Failed to open input file.\n";
        return 1;
    }
    std::streamsize file_size = in.tellg();
    in.seekg(0, std::ios::beg);
    std::streamsize midpoint = file_size / 2;

    std::ofstream out(output_path, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open output file.\n";
        return 1;
    }

    // 1. Configure initial parameters: lc=4, lp=0, pb=0
    lzma_options_lzma opt1;
    lzma_lzma_preset(&opt1, LZMA_PRESET_DEFAULT);
    opt1.lc = 4;
    opt1.lp = 0;
    opt1.pb = 0;

    lzma_filter filters[2] = {
        { LZMA_FILTER_LZMA2, &opt1 },
        { LZMA_VLI_UNKNOWN, nullptr }
    };

    lzma_stream strm = LZMA_STREAM_INIT;
    lzma_ret ret;

    // Initialize raw LZMA2 stream encoder if --raw is requested
    if (is_raw) {
        ret = lzma_raw_encoder(&strm, filters);
    } else {
        ret = lzma_stream_encoder(&strm, filters, LZMA_CHECK_CRC64);
    }

    if (ret != LZMA_OK) {
        std::cerr << "Error initializing lzma encoder: " << ret << "\n";
        return 1;
    }

    const size_t BUF_SIZE = 16384;
    std::vector<uint8_t> in_buf(BUF_SIZE);
    std::vector<uint8_t> out_buf(BUF_SIZE);

    auto compress_chunk = [&](lzma_action action) {
        do {
            strm.next_out = out_buf.data();
            strm.avail_out = BUF_SIZE;
            ret = lzma_code(&strm, action);
            
            if (ret != LZMA_OK && ret != LZMA_STREAM_END) {
                std::cerr << "Compression error code: " << ret << "\n";
                exit(1);
            }
            
            size_t out_size = BUF_SIZE - strm.avail_out;
            if (out_size > 0) {
                out.write(reinterpret_cast<char*>(out_buf.data()), out_size);
            }
        } while (strm.avail_out == 0);
    };

    // 2. Compress the first half using LZMA_RUN
    std::streamsize bytes_read_total = 0;
    while (bytes_read_total < midpoint) {
        std::streamsize to_read = std::min<std::streamsize>(BUF_SIZE, midpoint - bytes_read_total);
        in.read(reinterpret_cast<char*>(in_buf.data()), to_read);
        std::streamsize read_count = in.gcount();
        if (read_count == 0) break;

        bytes_read_total += read_count;
        strm.next_in = in_buf.data();
        strm.avail_in = read_count;
        compress_chunk(LZMA_RUN);
    }

    // 3. Flush the current LZMA2 chunk
    strm.avail_in = 0;
    compress_chunk(LZMA_SYNC_FLUSH);

    // 4. Update parameters mid-stream to lc=3, lp=1, pb=1
    lzma_options_lzma opt2;
    lzma_lzma_preset(&opt2, LZMA_PRESET_DEFAULT);
    opt2.lc = 3;
    opt2.lp = 1;
    opt2.pb = 1;
    filters[0].options = &opt2;

    ret = lzma_filters_update(&strm, filters);
    if (ret != LZMA_OK) {
        std::cerr << "Failed to update filters mid-stream: " << ret << "\n";
        return 1;
    }

    // 5. Compress the remaining half of the file
    while (in) {
        in.read(reinterpret_cast<char*>(in_buf.data()), BUF_SIZE);
        std::streamsize read_count = in.gcount();
        if (read_count == 0) break;

        strm.next_in = in_buf.data();
        strm.avail_in = read_count;
        compress_chunk(LZMA_RUN);
    }

    // 6. Finish the stream
    strm.avail_in = 0;
    compress_chunk(LZMA_FINISH);

    lzma_end(&strm);
    std::cout << "Successfully compressed (" << (is_raw ? "raw LZMA2" : ".xz container") << ")!\n";
    return 0;
}
