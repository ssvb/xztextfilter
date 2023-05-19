## Truncated wikipedia backup xml dumps for various languages

Wikipedia is a great source of texts in various languages. The backup
xml dumps can be downloaded here: https://dumps.wikimedia.org/backup-index.html
https://dumps.wikimedia.org/legal.html

This repository imports some of wikipedia xml dumps in a **truncated** form, so
that their size is manageable and suitable for testing:
```sh
curl -L https://dumps.wikimedia.org/.../filename.xml.bz2 | bzip2 -dc | head -n 100000 | xz -c > filename-truncated.xml.xz
```

It's difficult to test everything, but some languages are more interesting than the others.

### Ukrainian

Cyrillic alphabet, letters "ґ" and "Ґ" don't fit into the U+0400..U+047F range.

https://dumps.wikimedia.org/ukwiki/20230501/ukwiki-20230501-pages-articles-multistream.xml.bz2

### Armenian

Armenian occuipes a weird range U+0530..U+058F, which is not 128 characters aligned: https://en.wikipedia.org/wiki/Armenian_(Unicode_block)

https://dumps.wikimedia.org/hywiki/20230501/hywiki-20230501-pages-articles-multistream.xml.bz2

### Vietnamese

Vietnamese writing requires 134 additional letters (between both cases) besides the 52 already present in ASCII: https://en.wikipedia.org/wiki/Vietnamese_language_and_computers 

https://vietunicode.sourceforge.net/charset/vietalphabet.html

https://dumps.wikimedia.org/viwiki/20230501/viwiki-20230501-pages-articles-multistream.xml.bz2

### Marathi

3 bytes per character

https://dumps.wikimedia.org/mrwiki/20230501/mrwiki-20230501-pages-articles-multistream.xml.bz2

## Wiktionary dictionaries in binary StarDict format

https://en.wiktionary.org/wiki/Wiktionary:Copyrights

Downloaded the .syn and .idx files of the Russian-English dictionary from https://github.com/Vuizur/Wiktionary-Dictionaries

## Tatoeba

Download the CC0 licensed sentences from https://downloads.tatoeba.org/exports/sentences_CC0.tar.bz2

```Ruby
```

```
$ sqlite3 tatoeba-nonascii-CC0.sqlite3
CREATE TABLE sentences(id INTEGER PRIMARY KEY, lng TEXT, txt TEXT);
.mode csv
.mode ascii
.separator "\t" "\n"
.import tatoeba-nonascii-CC0.csv sentences
```
