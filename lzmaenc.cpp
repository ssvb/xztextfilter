#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cstdio>
#include <divsufsort.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

// --- LZMA Model Constants ---
constexpr int kNumStates = 12;
constexpr int kNumPosStates = 4; // pb = 2 -> 2^2 = 4
constexpr int MIN_MATCH = 2;     // Standard LZMA min match length
constexpr int MAX_MATCH = 273;   // Standard LZMA max match length (2 + 8 + 8 + 256 - 1)

// --- 1. LZMA Range Encoder ---
struct LZMAEncoder {
    uint64_t low = 0;
    uint32_t range = 0xFFFFFFFF;
    uint8_t cache = 0;
    uint32_t cache_size = 1;
    std::vector<uint8_t> out;

    void ShiftLow() {
        if ((uint32_t)low < 0xFF000000 || (int)(low >> 32) != 0) {
            uint8_t temp = cache;
            do {
                out.push_back(temp + (uint8_t)(low >> 32));
                temp = 0xFF;
            } while (--cache_size != 0);
            cache = (uint8_t)((uint32_t)low >> 24);
        }
        cache_size++;
        low = (uint32_t)low << 8;
    }

    void WriteHeader(uint64_t uncompressed_size) {
        // Header Byte: (pb * 5 + lp) * 9 + lc -> (2 * 5 + 0) * 9 + 3 = 93 (0x5D)
        out.push_back(0x5D);

        // Dictionary Size (e.g. max(4096, uncompressed_size), Little Endian)
        uint32_t dict_size = 4096;
        while (dict_size < uncompressed_size && dict_size < (1 << 23)) {
            dict_size <<= 1;
        }
        for (int i = 0; i < 4; i++) {
            out.push_back((dict_size >> (i * 8)) & 0xFF);
        }

        // Uncompressed Size (8 bytes, Little Endian)
        for (int i = 0; i < 8; i++) {
            out.push_back((uncompressed_size >> (i * 8)) & 0xFF);
        }
    }

    void EncodeBit(uint16_t& prob, int bit) {
        uint32_t newBound = (range >> 11) * prob;
        if (bit == 0) {
            range = newBound;
            prob += ((1 << 11) - prob) >> 5;
        } else {
            low += newBound;
            range -= newBound;
            prob -= prob >> 5;
        }
        while (range < (1 << 24)) {
            range <<= 8;
            ShiftLow();
        }
    }

    void EncodeDirectBits(uint32_t value, int numBits) {
        for (int i = numBits - 1; i >= 0; i--) {
            range >>= 1;
            if ((value >> i) & 1) {
                low += range;
            }
            while (range < (1 << 24)) {
                range <<= 8;
                ShiftLow();
            }
        }
    }

    void EncodeTree(uint16_t* probs, int numBits, uint32_t symbol) {
        uint32_t m = 1;
        for (int i = numBits - 1; i >= 0; i--) {
            uint32_t bit = (symbol >> i) & 1;
            EncodeBit(probs[m], bit);
            m = (m << 1) | bit;
        }
    }

    void EncodeReverseTree(uint16_t* probs, int numBits, uint32_t symbol) {
        uint32_t m = 1;
        for (int i = 0; i < numBits; i++) {
            uint32_t bit = (symbol >> i) & 1;
            EncodeBit(probs[m], bit);
            m = (m << 1) | bit;
        }
    }

    void Flush() {
        for (int i = 0; i < 5; i++) ShiftLow();
    }
};

// --- 2. Matched Literal Encoding ---
void EncodeLiteral(LZMAEncoder& rc, std::vector<uint16_t>& prob_literals, 
                   uint8_t symbol, uint8_t matchByte, uint32_t state, uint8_t prevByte) {
    // lc = 3, lp = 0
    uint32_t context = prevByte >> 5;
    uint16_t* probs = &prob_literals[context * 0x300];

    uint32_t m = 1;
    if (state >= 7) { // Matched Literal mode
        for (int i = 7; i >= 0; i--) {
            uint32_t matchBit = (matchByte >> i) & 1;
            uint32_t bit = (symbol >> i) & 1;
            uint32_t probIndex = (0x100 + (matchBit << 8)) + m;
            rc.EncodeBit(probs[probIndex], bit);
            m = (m << 1) | bit;
            if (matchBit != bit) {
                i--;
                for (; i >= 0; i--) {
                    uint32_t b = (symbol >> i) & 1;
                    rc.EncodeBit(probs[m], b);
                    m = (m << 1) | b;
                }
                break;
            }
        }
    } else { // Normal Literal mode
        for (int i = 7; i >= 0; i--) {
            uint32_t bit = (symbol >> i) & 1;
            rc.EncodeBit(probs[m], bit);
            m = (m << 1) | bit;
        }
    }
}

// Helper to compute LZMA Position Slot
uint32_t GetPosSlot(uint32_t v) {
    if (v < 4) return v;
    uint32_t temp = v;
    int numBits = 0;
    while (temp >= 2) {
        temp >>= 1;
        numBits++;
    }
    return (numBits << 1) | ((v >> (numBits - 1)) & 1);
}

// --- 3. Kasai LCP Array Computation ---
void compute_lcp(const unsigned char* T, const std::vector<saidx_t>& SA, int n, 
                 std::vector<int>& ISA, std::vector<int>& LCP) {
    for (int i = 0; i < n; i++) ISA[SA[i]] = i;
    int h = 0;
    for (int i = 0; i < n; i++) {
        if (ISA[i] > 0) {
            int j = SA[ISA[i] - 1];
            while (i + h < n && j + h < n && T[i + h] == T[j + h]) h++;
            LCP[ISA[i]] = h;
            if (h > 0) h--;
        } else {
            LCP[ISA[i]] = 0;
        }
    }
}

// --- 4. Main LZMA Encoder Engine ---
std::vector<uint8_t> StandardLZMAEncode(const std::vector<uint8_t>& data) {
    LZMAEncoder rc;
    uint64_t n = data.size();
    rc.WriteHeader(n);

    if (n == 0) {
        rc.Flush();
        return rc.out;
    }

    // Build Suffix Array & LCP using libdivsufsort
    std::vector<saidx_t> SA(n);
    std::vector<int> ISA(n), LCP(n);
    divsufsort(data.data(), SA.data(), n);
    compute_lcp(data.data(), SA, n, ISA, LCP);

    // LZMA Probability Tables
    std::vector<uint16_t> is_match(kNumStates * kNumPosStates, 1024);
    std::vector<uint16_t> is_rep(kNumStates, 1024);
    std::vector<uint16_t> pos_slot(4 * 64, 1024);
    std::vector<uint16_t> pos_spec_tree(10 * 64, 1024);
    std::vector<uint16_t> prob_align(16, 1024);

    // Length Coder Tables
    uint16_t len_choice1 = 1024, len_choice2 = 1024;
    std::vector<uint16_t> len_low(kNumPosStates * 16, 1024);
    std::vector<uint16_t> len_mid(kNumPosStates * 16, 1024);
    std::vector<uint16_t> len_high(256, 1024);

    // Literal Tables (lc = 3, lp = 0 -> 8 contexts * 768 entries)
    std::vector<uint16_t> prob_literals(8 * 0x300, 1024);

    uint32_t state = 0;
    uint32_t rep0 = 1; // Last match distance

    for (size_t i = 0; i < n; ) {
        uint32_t posState = i & (kNumPosStates - 1);
        uint8_t prevByte = (i > 0) ? data[i - 1] : 0;

        int best_len = 0;
        int best_dist = 0;
        int rank = ISA[i];

        // Search Suffix Array Left
        int current_lcp = n;
        for (int k = rank, depth = 0; k > 0 && depth < 64; k--, depth++) {
            current_lcp = std::min(current_lcp, LCP[k]);
            if (current_lcp < MIN_MATCH) break;
            if (SA[k - 1] < (saidx_t)i) {
                int match_len = std::min(current_lcp, MAX_MATCH);
                if (match_len > best_len) {
                    best_len = match_len;
                    best_dist = i - SA[k - 1];
                }
            }
        }

        // Search Suffix Array Right
        current_lcp = n;
        for (int k = rank, depth = 0; k < (int)n - 1 && depth < 64; k++, depth++) {
            current_lcp = std::min(current_lcp, LCP[k + 1]);
            if (current_lcp < MIN_MATCH) break;
            if (SA[k + 1] < (saidx_t)i) {
                int match_len = std::min(current_lcp, MAX_MATCH);
                if (match_len > best_len) {
                    best_len = match_len;
                    best_dist = i - SA[k + 1];
                }
            }
        }

        if (best_len >= MIN_MATCH) {
            // Encode: Token is Match
            rc.EncodeBit(is_match[state * kNumPosStates + posState], 1);
            rc.EncodeBit(is_rep[state], 0); // Normal match

            // 1. Encode Match Length
            int lenSymbol = best_len - MIN_MATCH;
            if (lenSymbol < 8) {
                rc.EncodeBit(len_choice1, 0);
                rc.EncodeTree(&len_low[posState * 16], 3, lenSymbol);
            } else if (lenSymbol < 16) {
                rc.EncodeBit(len_choice1, 1);
                rc.EncodeBit(len_choice2, 0);
                rc.EncodeTree(&len_mid[posState * 16], 3, lenSymbol - 8);
            } else {
                rc.EncodeBit(len_choice1, 1);
                rc.EncodeBit(len_choice2, 1);
                rc.EncodeTree(len_high.data(), 8, lenSymbol - 16);
            }

            // 2. Encode Match Distance
            uint32_t distSymbol = best_dist - 1;
            uint32_t lenState = std::min(best_len - MIN_MATCH, 3);
            uint32_t slot = GetPosSlot(distSymbol);
            
            rc.EncodeTree(&pos_slot[lenState * 64], 6, slot);

            if (slot >= 4) {
                int footerBits = (slot >> 1) - 1;
                uint32_t base = (2 | (slot & 1)) << footerBits;
                uint32_t directVal = distSymbol - base;

                if (slot < 14) {
                    rc.EncodeReverseTree(&pos_spec_tree[(slot - 4) * 64], footerBits, directVal);
                } else {
                    int numDirectBits = footerBits - 4;
                    rc.EncodeDirectBits(directVal >> 4, numDirectBits);
                    rc.EncodeReverseTree(prob_align.data(), 4, directVal & 15);
                }
            }

            // Update State Engine
            rep0 = best_dist;
            state = (state < 7) ? 7 : 10;
            i += best_len;
        } else {
            // Encode: Token is Literal
            rc.EncodeBit(is_match[state * kNumPosStates + posState], 0);

            uint8_t matchByte = (state >= 7) ? data[i - rep0] : 0;
            EncodeLiteral(rc, prob_literals, data[i], matchByte, state, prevByte);

            // Update State Engine
            if (state < 4) state = 0;
            else if (state < 10) state -= 3;
            else state -= 6;

            i++;
        }
    }

    rc.Flush();
    return rc.out;
}

// --- 5. CLI Interface ---
void print_help(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [OPTION]... [FILE]...\n"
              << "Compress files using standard LZMA encoding with libdivsufsort.\n\n"
              << "  -c, --stdout      write on standard output, keep original files unchanged\n"
              << "  -k, --keep        keep (don't delete) input files\n"
              << "  -h, --help        display this help and exit\n\n"
              << "With no FILE, or when FILE is -, read standard input.\n";
}

bool read_entire_stream(std::istream& in, std::vector<uint8_t>& buffer) {
    char temp[4096];
    while (in.read(temp, sizeof(temp))) {
        buffer.insert(buffer.end(), temp, temp + in.gcount());
    }
    buffer.insert(buffer.end(), temp, temp + in.gcount());
    return true;
}

int main(int argc, char** argv) {
    bool to_stdout = false;
    bool keep_files = false;
    std::vector<std::string> files;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-c" || arg == "--stdout") to_stdout = true;
        else if (arg == "-k" || arg == "--keep") keep_files = true;
        else if (arg == "-h" || arg == "--help") {
            print_help(argv[0]);
            return 0;
        }
        else if (arg[0] == '-' && arg != "-") {
            std::cerr << argv[0] << ": unrecognized option '" << arg << "'\n";
            return 1;
        }
        else files.push_back(arg);
    }

#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    if (files.empty()) {
        files.push_back("-");
        to_stdout = true;
    }

    for (const auto& file : files) {
        std::vector<uint8_t> input_data;
        bool is_stdin = (file == "-");

        if (is_stdin) {
            read_entire_stream(std::cin, input_data);
        } else {
            std::ifstream ifs(file, std::ios::binary);
            if (!ifs) {
                std::cerr << argv[0] << ": " << file << ": No such file or directory\n";
                continue;
            }
            read_entire_stream(ifs, input_data);
            ifs.close();
        }

        std::vector<uint8_t> compressed = StandardLZMAEncode(input_data);

        if (to_stdout || is_stdin) {
            std::cout.write(reinterpret_cast<const char*>(compressed.data()), compressed.size());
            std::cout.flush();
        } else {
            std::string out_file = file + ".lzma";
            std::ofstream ofs(out_file, std::ios::binary);
            if (!ofs) {
                std::cerr << argv[0] << ": " << out_file << ": Permission denied\n";
                continue;
            }
            ofs.write(reinterpret_cast<const char*>(compressed.data()), compressed.size());
            ofs.close();

            if (!keep_files) {
                std::remove(file.c_str());
            }
        }
    }

    return 0;
}
