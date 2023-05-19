# xztextfilter
A preprocessing filter for improving LZMA compression of non-English UTF-8 texts. Intended to be contributed to XZ Utils after it's ready and stable: https://github.com/tukaani-project/xz/issues/50

## Releases

### proto1

This is a stdin to stdout convertor. Hastily written and almost untested. May contain bugs. Exists just to get started
and provide some sort of baseline to compare against.

| Test sample                              | normal compression | compression after filter | delta  |
| ---------------------------------------- | ------------------ | ------------------------ | ------ |
| ja-nautilus.mo                           |              27332 |                    27340 | +0.03% |
| ja-nautilus.po                           |              33236 |                    32988 | -0.75% |
| ml-nautilus.mo                           |              20920 |                    20128 | -3.79% |
| ml-nautilus.po                           |              56728 |                    53892 | -5.00% |
| pl-nautilus.mo                           |              27572 |                    28176 | +2.19% |
| pl-nautilus.po                           |              31752 |                    31888 | +0.43% |
| vi-nautilus.mo                           |              26608 |                    27236 | +2.36% |
| vi-nautilus.po                           |              43544 |                    43444 | -0.23% |
| ru-en-wiktionary-stardict.idx            |             324556 |                   323868 | -0.21% |
| ru-en-wiktionary-stardict.syn            |            1962776 |                  1796368 | -8.48% |
| tatoeba-nonascii-CC0.csv                 |            1367080 |                  1331232 | -2.62% |
| tatoeba-nonascii-CC0.sqlite3             |            1798964 |                  1801348 | +0.13% |
| bnwiki-truncated-30k.xml                 |             678784 |                   630740 | -7.08% |
| elwiki-truncated-30k.xml                 |             950288 |                   902532 | -5.03% |
| frwiki-truncated-30k.xml                 |            1137748 |                  1133288 | -0.39% |
| hiwiki-truncated-30k.xml                 |             791936 |                   733800 | -7.34% |
| hywiki-truncated-30k.xml                 |             563796 |                   542012 | -3.86% |
| jawiki-truncated-30k.xml                 |             954144 |                   932860 | -2.23% |
| kowiki-truncated-30k.xml                 |             723140 |                   715900 | -1.00% |
| mrwiki-truncated-30k.xml                 |             462532 |                   424700 | -8.18% |
| plwiki-truncated-30k.xml                 |             956380 |                   951596 | -0.50% |
| ukwiki-truncated-30k.xml                 |             965724 |                   912344 | -5.53% |
| viwiki-truncated-30k.xml                 |             874024 |                   867492 | -0.75% |
| zhwiki-truncated-30k.xml                 |            1250460 |                  1251916 | +0.12% |

## License

The code of the filter is MIT licensed in this particular repository. But the official XZ Utils project
maintainers (https://tukaani.org/xz/) are allowed to take any parts of this code and relicense it as
public domain within their git repository. So the path to public domain is via
the https://github.com/tukaani-project/xz repository.

The data files from the "testfiles" subdirectory include various text and binary files collected
from the Internet. They have their own licenses and the MIT license does not cover them.
