
from __future__ import division

import math
import random
import os

import logging

import torch as th
import numpy as np

from torchtext import data
import io

from torchtext.data.utils import RandomShuffler
from torchtext.data.dataset import Dataset

logger = logging.getLogger(__name__)


def process_lines(
    path, tag_path,
    encoding,
    text_field, tag_field,
    newline_eos,
    fields,
):
    examples = []
    with io.open(path, encoding=encoding) as f, io.open(tag_path, encoding=encoding) as g:
        for i, (line, tag_line) in enumerate(zip(f, g)):
            text = text_field.preprocess(line)
            tags = tag_field.preprocess(tag_line)
            #print(text)
            if newline_eos:
                text.append(u'<eos>')
                tags.append(u'<eos>')
            example = data.Example.fromlist([text, tags], fields)
            example.idx = i
            examples.append(example)
    return examples


class TaggedPennTreebank(Dataset):
    """The Penn Treebank dataset.
    A relatively small dataset originally created for POS tagging.

    References
    ----------
    Marcus, Mitchell P., Marcinkiewicz, Mary Ann & Santorini, Beatrice (1993).
    Building a Large Annotated Corpus of English: The Penn Treebank
    """

    urls = ['https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt',
            'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt',
            'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt']
    name = 'penn-treebank'
    dirname = ''

    def __init__(
        self,
        path,
        tag_path,
        text_field,
        tag_field,
        newline_eos=True,
        feature_path = None,
        encoding='utf-8',
        **kwargs,
    ):
        """Create a LanguageModelingDataset given a path and a field.

        Arguments:
            path: Path to the data file.
            text_field: The field that will be used for text data.
            newline_eos: Whether to add an <eos> token for every newline in the
                data file. Default: True.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('text', text_field), ("tags", tag_field)]
        examples = process_lines(
            path, tag_path,
            encoding,
            text_field, tag_field,
            newline_eos,
            fields,
        )

        super(TaggedPennTreebank, self).__init__(
            examples, fields, **kwargs)

    @classmethod
    def splits(cls,
        text_field,
        tag_field,
        path=".data/PTB/sup",
        root='.data', train='ptb.train.txt',
        validation='ptb.valid.txt', test='ptb.test.txt',
        train_tags = "ptb.train.tags",
        validation_tags = "ptb.valid.tags",
        test_tags = "ptb.test.tags",
        **kwargs,
    ):
        """Create dataset objects for splits of the Penn Treebank dataset.

        Arguments:
            text_field: The field that will be used for text data.
            root: The root directory where the data files will be stored.
            train: The filename of the train data. Default: 'ptb.train.txt'.
            validation: The filename of the validation data, or None to not
                load the validation set. Default: 'ptb.valid.txt'.
            test: The filename of the test data, or None to not load the test
                set. Default: 'ptb.test.txt'.
        """
        if path is None:
            path = cls.download(root)
        train_data = None if train is None else cls(
            os.path.join(path, train),
            os.path.join(path, train_tags),
            text_field,
            tag_field,
            **kwargs,
        )
        val_data = None if validation is None else cls(
            os.path.join(path, validation),
            os.path.join(path, validation_tags),
            text_field,
            tag_field,
            **kwargs,
        )
        test_data = None if test is None else cls(
            os.path.join(path, test),
            os.path.join(path, test_tags),
            text_field,
            tag_field,
            **kwargs,
        )
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)

    @classmethod
    def iters(cls, batch_size=32, bptt_len=35, device=0, root='.data',
              vectors=None, **kwargs):
        """Create iterator objects for splits of the Penn Treebank dataset.

        This is the simplest way to use the dataset, and assumes common
        defaults for field, vocabulary, and iterator parameters.

        Arguments:
            batch_size: Batch size.
            bptt_len: Length of sequences for backpropagation through time.
            device: Device to create batches on. Use -1 for CPU and None for
                the currently active GPU device.
            root: The root directory where the data files will be stored.
            wv_dir, wv_type, wv_dim: Passed to the Vocab constructor for the
                text field. The word vectors are accessible as
                train.dataset.fields['text'].vocab.vectors.
            Remaining keyword arguments: Passed to the splits method.
        """
        TEXT = data.Field()

        train, val, test = cls.splits(TEXT, root=root, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)

        return data.BPTTIterator.splits(
            (train, val, test), batch_size=batch_size, bptt_len=bptt_len,
            device=device)

