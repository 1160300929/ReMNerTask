# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """

import torch
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union
import re
import ObjectFeatureExtractor
import GridFeatureExtractor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import torch.nn as nn
logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    """
    A single training/test example for token classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """

    guid: str
    words: List[str]
    labels: Optional[List[str]]
    img_id: str


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"

def preprocess_word(word):
    """
    - Do lowercase
    - Regular expression (number, url, hashtag, user)
        - https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb

    :param word: str
    :return: word: str
    """
    number_re = r"[-+]?[.\d]*[\d]+[:,.\d]*"
    url_re = r"https?:\/\/\S+\b|www\.(\w+\.)+\S*"
    hashtag_re = r"#\S+"
    user_re = r"@\w+"

    if re.compile(number_re).match(word):
        word = '<NUMBER>'
    elif re.compile(url_re).match(word):
        word = '<URL>'
    elif re.compile(hashtag_re).match(word):
        word = word[1:]  # only erase `#` at the front
    elif re.compile(user_re).match(word):
        word = word[1:]  # only erase `@` at the front

    word = word.lower()

    return word

'''
read dataset and convert datasets to features for Multimodal NER
'''
class MMNerTask:
    def read_examples_from_file(self, data_dir, mode: Union[Split, str]) -> List[InputExample]:
        data_dir = os.path.join(data_dir, "{}.txt".format(mode))
        with open(data_dir, "r", encoding="utf-8") as f:
            sentences = []

            sentence = [[], []]  # [[words], [tags], img_id]
            for line in f:
                if line.strip() == "":
                    continue

                if line.startswith("IMGID:"):
                    if sentence[0]:
                        sentences.append(sentence)
                        sentence = [[], []]  # Flush

                    # Add img_id at last
                    img_id = line.replace("IMGID:", "").strip() + '.jpg'
                    sentence.append(img_id)
                else:
                    try:
                        word, tag = line.strip().split("\t")
                        word = preprocess_word(word)
                        sentence[0].append(word)
                        sentence[1].append(tag)
                    except:
                        logger.info("\"{}\" cannot be splitted".format(line.rstrip()))
            # Flush the last one
            if sentence[0]:
                sentences.append(sentence)
        examples = []

        for (i, sentence) in enumerate(sentences):
            words, labels, img_id = sentence[0], sentence[1], sentence[2]
            assert len(words) == len(labels)

            guid = "%s-%s" % (mode, i)
            if i % 10000 == 0:
                logger.info(sentence)
            examples.append(InputExample(guid=guid, img_id=img_id, words=words, labels=labels))

        return examples

    def get_labels(self, path: str) -> List[str]:
        if path:
            with open(path, "r") as f:
                labels = f.read().splitlines()
            if "O" not in labels:
                labels = ["O"] + labels
            return labels
        else:
            return ["O", "B-OTHER", "I-OTHER", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

    def convert_examples_to_features(
        self,
        examples,
        label_list,
        max_seq_length,
        tokenizer,
        data_dir
    ) -> List[InputFeatures]:
        raise NotImplementedError

class MMNerTask_Pixel(MMNerTask):

    def convert_examples_to_features(
        self,
        examples,
        label_list,
        max_seq_length,
        tokenizer,
        data_dir
    ) -> List[InputFeatures]:
        return None

class MMNerTask_Object(MMNerTask):

    def convert_examples_to_features(
        self,
        examples,
        label_list,
        max_seq_length,
        tokenizer,
        data_dir
    ) -> List[InputFeatures]:
        return ObjectFeatureExtractor.convert_mm_examples_to_features(examples, label_list, max_seq_length,
                                                                      tokenizer,
                                                                      data_dir)

class MMNerTask_Grid(MMNerTask):
    def convert_examples_to_features(
        self,
        examples,
        label_list,
        max_seq_length,
        tokenizer,
        data_dir,
        crop_size=224,
    ) -> List[InputFeatures]:
        return GridFeatureExtractor.convert_mm_examples_to_features(examples, label_list, max_seq_length, tokenizer, crop_size,
                                                                    data_dir)

class MMNerDataset(Dataset):

    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index

    def __init__(
            self,
            token_classification_task: MMNerTask,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            labels: List[str],
            model_type: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
                 ):
        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}_{}_{}".format(mode.value, tokenizer.__class__.__name__, model_type,str(max_seq_length)),
        )
        if os.path.exists(cached_features_file) and not overwrite_cache:
            self.features = torch.load(cached_features_file)
        else:
            examples = token_classification_task.read_examples_from_file(data_dir, mode)
            # TODO clean up all this to leverage built-in features of tokenizers
            self.features = token_classification_task.convert_examples_to_features(
                examples,
                labels,
                max_seq_length,
                tokenizer,
                data_dir,
            )
            torch.save(self.features,cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


def valid_sequence_output(sequence_output, valid_mask, attention_mask):
    batch_size, max_len, feat_dim = sequence_output.shape
    valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32,
                               device='cuda' if torch.cuda.is_available() else 'cpu')
    valid_attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long,
                                       device='cuda' if torch.cuda.is_available() else 'cpu')
    for i in range(batch_size):
        jj = -1
        for j in range(max_len):
            if valid_mask[i][j].item() == 1:
                jj += 1
                valid_output[i][jj] = sequence_output[i][j]
                valid_attention_mask[i][jj] = attention_mask[i][j]
    return valid_output, valid_attention_mask

