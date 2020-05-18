
import nltk
import pathlib

import os
import re
import sys
import argparse
import nltk
from nltk.corpus import ptb
import os
from pathlib import Path

base_path = pathlib.Path(".data/PTB/LDC99T42/treebank_3/parsed/mrg/wsj")
train_section = [
    "00", "01", '02', '03', '04', '05', '06', '07', '08', '09', '10',
    '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
]
valid_section = ['21', "22"]
test_section = ['23', "24"]

word_tags = [
    'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 
    'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 
    'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
    'WDT', 'WP', 'WP$', 'WRB',
]
currency_tags_words = ['#', '$', 'C$', 'A$']
ellipsis = ['*', '*?*', '0', '*T*', '*ICH*', '*U*', '*RNR*', '*EXP*', '*PPA*', '*NOT*']
punctuation_tags = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``']
punctuation_words = [
    '.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``', '--', ';', 
    '-', '?', '!', '...', '-LCB-', '-RCB-',
]

def get_pos(sections, base_path):
    return [
        sentence.pos()
        for d in sections
        for f in sorted((base_path / d).iterdir())
        for sentence in ptb.parsed_sents(str(f.resolve()))
    ]

def is_num(x):
    return re.match(r'^-?\d+(?:\.\d+)?$', x)

def write_txt_pos(sentags, ignore_tags, sentence_f, tag_f, lower=False):
    sentences = []
    tags = []
    for sentag in sentags:
        sentence, tag = zip(*[x for x in sentag if x[1] not in ignore_tags])
        sentences.append(" ".join([
            #x if not is_num(x) else "N"
            (x.lower() if lower else x) if not is_num(x) else "0"
            for x in sentence
        ]))
        tags.append(" ".join(tag))
    sentence_f.write_text("\n".join(sentences))
    tag_f.write_text("\n".join(tags))
    return sentences, tags

if __name__ == "__main__":
    ignore_tags = ["-NONE-"]

    train_pos = get_pos(train_section, base_path)
    valid_pos = get_pos(valid_section, base_path)
    test_pos = get_pos(test_section, base_path)
    all_sections = train_pos + valid_pos + test_pos

    out_path = pathlib.Path(".data/PTB")
    sentence_f = out_path / "ptb.txt"
    tag_f = out_path / "ptb.tags"

    sentences, tags = write_txt_pos(all_sections, ignore_tags, sentence_f, tag_f)

    ignore_tags = ["-NONE-"] + punctuation_tags
    sentence_f = out_path / "ptb.nopunct.txt"
    tag_f = out_path / "ptb.nopunct.tags"
    sentences, tags = write_txt_pos(
        all_sections, ignore_tags, sentence_f, tag_f, lower=True,
    )
