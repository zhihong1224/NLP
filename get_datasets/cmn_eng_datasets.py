import os
import numpy as np
import re
import json
import torch
import torch.utils.data as data
from utils.config import args


class LabelTransform(object):
    def __init__(self, size, pad):
        self.size = size
        self.pad = pad

    def __call__(self, label):
        label = np.pad(label, (0, (self.size - label.shape[0])), mode='constant', constant_values=self.pad)
        return label


class EN2CNDataset(data.Dataset):
    def __init__(self, root, max_output_len, set_name):
        self.root = root

        self.word2int_cn, self.int2word_cn = self.get_dictionary('cn')
        self.word2int_en, self.int2word_en = self.get_dictionary('en')

        # load data
        self.data = []
        with open(os.path.join(self.root, f'{set_name}.txt'), "r") as f:
            for line in f:
                self.data.append(line)
        print(f'{set_name} dataset size: {len(self.data)}')

        self.cn_vocab_size = len(self.word2int_cn)
        self.en_vocab_size = len(self.word2int_en)
        self.transform = LabelTransform(max_output_len, self.word2int_en['<PAD>'])

    def get_dictionary(self, language):
        # load dict
        with open(os.path.join(self.root, f'word2int_{language}.json'), "r") as f:
            word2int = json.load(f)
        with open(os.path.join(self.root, f'int2word_{language}.json'), "r") as f:
            int2word = json.load(f)
        return word2int, int2word

    def __len__(self):
        return len(self.data)

    def __getitem__(self, Index):
        # tokenize
        sentences = self.data[Index]
        sentences = re.split('[\t\n]', sentences)
        sentences = list(filter(None, sentences))
        # print (sentences)
        assert len(sentences) == 2

        # special words
        BOS = self.word2int_en['<BOS>']
        EOS = self.word2int_en['<EOS>']
        UNK = self.word2int_en['<UNK>']

        # add <EOS> and <BOS>
        en, cn = [BOS], [BOS]
        # subword
        sentence = re.split(' ', sentences[0])
        sentence = list(filter(None, sentence))
        # print (f'en: {sentence}')
        for word in sentence:
            en.append(self.word2int_en.get(word, UNK))
        en.append(EOS)

        # e.g. < BOS >, we, are, friends, < EOS > --> 1, 28, 29, 205, 2
        sentence = re.split(' ', sentences[1])
        sentence = list(filter(None, sentence))
        # print (f'cn: {sentence}')
        for word in sentence:
            cn.append(self.word2int_cn.get(word, UNK))
        cn.append(EOS)

        en, cn = np.asarray(en), np.asarray(cn)

        # <PAD> to same length
        en, cn = self.transform(en), self.transform(cn)
        en, cn = torch.LongTensor(en), torch.LongTensor(cn)

        return en, cn
