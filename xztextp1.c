/*
 * A filter for making UTF-8 texts more compressible by LZMA.
 * 
 * Copyright (c) 2023 Siarhei Siamashka
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

int verbose = 0;

static inline unsigned int fxhash(unsigned int x) { return (0x9e3779b9U * x) >> (32 - 7); }

static void do_binary_copy(int len)
{
	while (len-- > 0) {
		int tmp = fgetc_unlocked(stdin);
		assert(tmp != EOF);
		fputc_unlocked(tmp, stdout);
	}
}

static void output_utf8_char(int ch)
{
	assert(ch >= 0x80);
	if (ch < 0x800) {
		// 2 bytes UTF-8 character
		fputc_unlocked(0xC0 | (ch >> 6), stdout);
		fputc_unlocked(0x80 | (ch & 0x3F), stdout);
	} else if (ch < 0x10000) {
		// 3 bytes UTF-8 character
		fputc_unlocked(0xE0 | (ch >> 12), stdout);
		fputc_unlocked(0x80 | ((ch >> 6) & 0x3F), stdout);
		fputc_unlocked(0x80 | (ch & 0x3F), stdout);
	} else {
		// 4 bytes UTF-8 character
		fputc_unlocked(0xF0 | (ch >> 18), stdout);
		fputc_unlocked(0x80 | ((ch >> 12) & 0x3F), stdout);
		fputc_unlocked(0x80 | ((ch >> 6) & 0x3F), stdout);
		fputc_unlocked(0x80 | (ch & 0x3F), stdout);
	}
}

void decode()
{
	int blockstart = 0x80; /* Assume Latin-1 by default */
	int hashtbl[128] = { 0 };

	int ch = fgetc_unlocked(stdin);
	int ch2 = fgetc_unlocked(stdin);
	if (ch != '~' || ch2 != '1') {
		fprintf(stderr, "The input data is not a valid proto1 preprocessed text.\n");
		exit(1);
	}

	while ((ch = fgetc_unlocked(stdin)) != EOF) {
		/* Escape a binary block. The encoding overhead is one extra
		   byte for block sizes up to 4 and two bytes for block sizes
		   up to 388 */
		if (ch >= 0x1C && ch <= 0x1F) {
			int len = (ch - 0x1B); /* 1, 2, 3 or 4 */
			int tmp = fgetc_unlocked(stdin);
			assert(tmp != EOF);
			if (tmp >= 0x20 && tmp <= 0x7F) {
				/* this isn't a part of binary data because
				   0x20-0x7F don't need escaping */
				len = (tmp - 0x1F) * 4 + len;
			} else {
				fputc_unlocked(tmp, stdout);
				len--;
			}
			do_binary_copy(len);
			continue;
		}

		if (ch >= 0x0E && ch <= 0x15) {
			/* Emit a 2 byte UTF-8 character */
			int p2 = fgetc_unlocked(stdin);
			assert(p2 != EOF);
			ch = ((ch - 0x0E) << 8) | p2;
			/* 0x00-0x7F would be invalid for a two byte UTF-8
			   encoding, so use it for hash lookups */
			if (ch < 0x80) {
				ch = hashtbl[ch];
				assert(ch != 0);
				assert(ch >= 0x80);
			}
			output_utf8_char(ch);
			continue;
		} else if (ch == 0x16) {
			/* Emit a 3 byte UTF-8 character *without* hash table update */
			int p2 = fgetc_unlocked(stdin);
			assert(p2 != EOF);
			int p3 = fgetc_unlocked(stdin);
			assert(p3 != EOF);
			int val = (p2 << 8) | p3;
			/* 0x000-0x7FF would be invalid for a three byte UTF-8
			   encoding, so use it for switching blockstart */
			if (val < 0x800) {
				blockstart = val * 0x80; // FIXME (not all Unicode blocks are aligned at 0x80): https://en.wikipedia.org/wiki/Armenian_(Unicode_block)
				continue;
			}
			output_utf8_char(val);
			continue;
		} else if (ch == 0x17) {
			/* Emit a 3 byte UTF-8 character *with* hash table update */
			int p2 = fgetc_unlocked(stdin);
			assert(p2 != EOF);
			int p3 = fgetc_unlocked(stdin);
			assert(p3 != EOF);
			int val = (p2 << 8) | p3;
			/* 0x000-0x7FF would be invalid for a three byte UTF-8
			   encoding, so use it for escaping larger binary blocks
			   (sizes from 389 to 2436) */
			if (val < 0x800) {
				do_binary_copy(389 + val);
				continue;
			}
			hashtbl[fxhash(val)] = val;
			output_utf8_char(val);
			continue;
		} else if (ch == 0x18 || ch == 0x19) {
			/* A 4 byte UTF-8 character */
			int p2 = fgetc_unlocked(stdin);
			assert(p2 != EOF);
			int p3 = fgetc_unlocked(stdin);
			assert(p3 != EOF);
			int p4 = fgetc_unlocked(stdin);
			assert(p4 != EOF);
			int val = (p2 << 16) | (p3 << 8) | p4;
			if (ch == 0x19)
				hashtbl[fxhash(val)] = val;
			output_utf8_char(val);
			continue;
		}

		if (ch >= 0x80) {
			/* FIXME: maybe do a table lookup here to support arbitrary
			   codepages instead of just adding 'blockstart'? */
			output_utf8_char(ch - 0x80 + blockstart);
		} else {
			fputc_unlocked(ch, stdout);
		}
	}
}

#ifndef DISABLE_ENCODER
void encode_char(int val, int blockstart, int hashtbl[128])
{
	int hash_idx;
	if (val < 0x80) {
		switch (val) {
			case 0x0E:
			case 0x0F:
			case 0x10:
			case 0x11:
			case 0x12:
			case 0x13:
			case 0x14:
			case 0x15:
			case 0x16:
			case 0x17:
			case 0x18:
			case 0x19:
			case 0x1C:
			case 0x1D:
			case 0x1E:
			case 0x1F:
				fputc_unlocked(0x1C, stdout);
			default:
				fputc_unlocked(val, stdout);
				break;
		}
	} else if (val >= blockstart && val < blockstart + 0x80) {
		fputc_unlocked(val - blockstart + 0x80, stdout);
	} else if (val < 0x800) {
		fputc_unlocked(0x0E + (val >> 8), stdout);
		fputc_unlocked(val & 0xFF, stdout);
	} else if (val < 0x10000) {
		hash_idx = fxhash(val);
		if (hashtbl[hash_idx] == val) {
			/* 2 byte encoding via hash */
			fputc_unlocked(0x0E, stdout);
			fputc_unlocked(hash_idx, stdout);
		} else if (hashtbl[hash_idx] == 0) {
			/* add to hash */
//			fprintf(stderr, "adding U+%04X to hash\n", val);
			hashtbl[hash_idx] = val;
			fputc_unlocked(0x17, stdout);
			fputc_unlocked(val >> 8, stdout);
			fputc_unlocked(val & 0xFF, stdout);
		} else {
//			fprintf(stderr, "hash collision between U+%04X and U+%04X at index %d\n", val, hashtbl[hash_idx], hash_idx);
			fputc_unlocked(0x16, stdout);
			fputc_unlocked(val >> 8, stdout);
			fputc_unlocked(val & 0xFF, stdout);
		}
	} else {
		hash_idx = fxhash(val);
		if (hashtbl[hash_idx] == val) {
			/* 2 byte encoding via hash */
			fputc_unlocked(0x0E, stdout);
			fputc_unlocked(hash_idx, stdout);
		} else if (hashtbl[hash_idx] == 0) {
			/* add to hash */
//			fprintf(stderr, "adding U+%04X to hash\n", val);
			hashtbl[hash_idx] = val;
			fputc_unlocked(0x19, stdout);
			fputc_unlocked(val >> 16, stdout);
			fputc_unlocked(val >> 8, stdout);
			fputc_unlocked(val & 0xFF, stdout);
		} else {
//			fprintf(stderr, "hash collision between U+%04X and U+%04X at index %d\n", val, hashtbl[hash_idx], hash_idx);
			fputc_unlocked(0x18, stdout);
			fputc_unlocked(val >> 16, stdout);
			fputc_unlocked(val >> 8, stdout);
			fputc_unlocked(val & 0xFF, stdout);
		}
	}
}

void encode_bin_length(int len)
{
	assert(len > 0);
	if (len <= 4) {
		fputc_unlocked(0x1B + len, stdout);
		return;
	}
	if (len <= 388) {
		len -= 5;
		int rem = len % 4;
		fputc_unlocked(0x1C + rem, stdout);
		fputc_unlocked(0x20 + (len - rem) / 4, stdout);
		return;
		
	}
	if (len <= 2436) {
		len -= 389;
		fputc_unlocked(0x17, stdout);
		fputc_unlocked(len >> 8, stdout);
		fputc_unlocked(len & 0xFF, stdout);
		return;
	}
	assert(0);
}

void encode_codepage_switch(int blockstart)
{
	fputc_unlocked(0x16, stdout);
	fputc_unlocked((blockstart / 0x80) >> 8, stdout);
	fputc_unlocked((blockstart / 0x80) & 0xFF, stdout);
}

void encode()
{
	int blockstart = 0x80; /* Latin-1 by default */
	int hashtbl[128] = { 0 };
	int b1;
	long offs = 0;

	int codepage_detected = 0;
	int prev_char_block = -1;

	unsigned char binbuf[2436];
	int binbufsize = 0;

	while ((b1 = fgetc_unlocked(stdin)) != EOF) {
		offs++;
		if (offs == 1) {
			/* */
			fputc_unlocked('~', stdout);
			fputc_unlocked('1', stdout);
		}
		if (binbufsize >= 2400) {
			/* flush the binary buffer before switching to text mode */
			encode_bin_length(binbufsize);
			fwrite(binbuf, 1, binbufsize, stdout);
			binbufsize = 0;
		}
		int binbufsize_snapshot = binbufsize;
		binbuf[binbufsize++] = b1;

		if ((b1 & 0xC0) == 0x80) {
//			fprintf(stderr, "Error: malformed UTF-8 (the first byte) 0x%08X\n", offs);
			continue;
		}

		int valid = 1;
		int len = 0;
		while (b1 & 0x80) {
			len++;
			b1 <<= 1;
		}
		int val = (b1 & 0xFF) >> len;
		if (len == 0)
			len = 1;
		if ((len < 1) || (len > 4)) {
//			fprintf(stderr, "Error: malformed UTF-8 (too long) 0x%08X\n", offs);
			continue;
		}
		const static int mincode_tbl[] = { 0, 0, 0x80, 0x800, 0x10000};
		int mincode = mincode_tbl[len];
		for (int i = 0; i < len - 1; i++) {
			int b2 = fgetc_unlocked(stdin);
			offs++;
			if (b2 == EOF) {
//				fprintf(stderr, "Error: input data abruptly terminated with incomplete character\n");
				valid = 0;
				break;
			}
			binbuf[binbufsize++] = b2;
			if ((b2 & 0xC0) != 0x80) {
//				fprintf(stderr, "Error: malformed UTF-8 (non-first bytes are badly encoded) 0x%08X\n", offs);
				valid = 0;
				break;
			}
			val <<= 6;
			val |= (b2 & 0x3F);
		}
		if (valid && val < mincode) {
//			fprintf(stderr, "Error: too many bytes for an UTF-8 character (U+%04X, len=%d) 0x%08X\n", val, len, offs);
			valid = 0;
		}

		if (!codepage_detected && valid && binbufsize_snapshot == 0) {
			int char_block = (val >> 7) << 7;
			if (char_block == prev_char_block && char_block != 0) {
				if (verbose)
					fprintf(stderr, "Codepage detected: U+%04X..U+%04X at offs 0x%08X\n",
						char_block, char_block + 0x7F, (int)offs);
				if (blockstart != char_block)
					encode_codepage_switch(char_block);
				blockstart = char_block;
				codepage_detected = 1;
			}
			prev_char_block = char_block;
		}

		/* just append single byte ASCII characters to the binary buffer
		   if the binary buffer is already non-empty */
		if (valid && (binbufsize_snapshot == 0 || (len > 1 && val >= blockstart && val < blockstart + 0x80))) {
			if (binbufsize_snapshot > 0) {
				/* flush the binary buffer before switching to text mode */
				encode_bin_length(binbufsize_snapshot);
//				fprintf(stderr, "binbuf: %02X..%02X, offs=0x%08X\n", binbuf[0], binbuf[binbufsize_snapshot - 1], offs);
				fwrite(binbuf, 1, binbufsize_snapshot, stdout);
				binbufsize = 0;
			}
			encode_char(val, blockstart, hashtbl);
			binbufsize = 0;
		}
	}

	if (binbufsize > 0) {
		/* flush the remaining part as binary data */
		encode_bin_length(binbufsize);
		fwrite(binbuf, 1, binbufsize, stdout);
	}
}
#endif

int main(int argc, char **argv)
{
	if (argc == 2 && strcmp(argv[1], "-d") == 0) {
		decode();
		return 0;
	}

#ifndef DISABLE_ENCODER
	if (argc == 1) {
		encode();
		return 0;
	}
	if (argc == 2 && strcmp(argv[1], "-v") == 0) {
		verbose = 1;
		encode();
		return 0;
	}
#endif

	fputs("Usage: xztextp1 [-d|-h]\n"
		"  -d          decode\n"
		"  -h          display this short help and exit\n\n"
		"An experimental prototype1 of the preprocessing filter for improving\n"
		"LZMA compression of non-English UTF-8 texts. Reads data from stdin\n"
		"and writes results to stdout.\n",
		stderr);
	return 0;
}
