
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
    '11', '12', '13', '14', '15', '16', '17', '18',
]
valid_section = ["19","20", '21']
test_section = ["22", '23', "24"]

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

def write_txt_pos(sentags, ignore_tags, sentence_f, tag_f,
    lower=False,
    digits=False,
):
    sentences = []
    tags = []
    for sentag in sentags:
        sentence, tag = zip(*[x for x in sentag if x[1] not in ignore_tags])
        sentences.append(" ".join([
            #x if not is_num(x) else "N"
            (x.lower() if lower else x)
            if not is_num(x) else (
                "0" if not digits else "0" * len(x)
            )
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

    out_path = pathlib.Path(".data/PTB/sup")
    sentence_f = out_path / "ptb.train.txt"
    tag_f = out_path / "ptb.train.tags"
    sentences, tags = write_txt_pos(train_pos, ignore_tags, sentence_f, tag_f)

    sentence_f = out_path / "ptb.valid.txt"
    tag_f = out_path / "ptb.valid.tags"
    sentences, tags = write_txt_pos(valid_pos, ignore_tags, sentence_f, tag_f)

    sentence_f = out_path / "ptb.test.txt"
    tag_f = out_path / "ptb.test.tags"
    sentences, tags = write_txt_pos(test_pos, ignore_tags, sentence_f, tag_f)
