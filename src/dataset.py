import json

import torch
from torch.utils.data import IterableDataset
import torch.nn as nn


class KDDcupDataset(IterableDataset):
    def __init__(self, file_path):
        super(KDDcupDataset).__init__()
        self.file_path = file_path
        self.feature_names = ['target', 'display_url', 'ad_id', 'advertiser_id', 
                              'depth', 'position', 'user_id', 'gender', 'age',
                              'keyword', 'title', 'description', 'query']
        self.keyword_vocab = json.load(open('../data/keyword_vocab.json', 'r'))
        self.title_vocab = json.load(open('../data/title_vocab.json', 'r'))
        self.description_vocab = json.load(open('../data/description_vocab.json', 'r'))
        self.keyword_sequence_len = 16
        self.title_sequence_len = 32
        self.description_sequence_len = 50
        
    def preprocess_line(self, line):
        ret = dict()
        line['target'] = torch.tensor(list(map(float, line['target'])), dtype=torch.float32)
        line['depth'] = torch.tensor(list(map(float, line['depth'])), dtype=torch.float32)
        line['position'] = torch.tensor(list(map(float, line['position'])), dtype=torch.float32)
        line['gender'] = torch.nn.functional.one_hot(torch.tensor(list(map(int, line['gender'])), dtype=torch.int64), num_classes=3).view(-1)
        line['age'] = torch.nn.functional.one_hot(torch.tensor(list(map(int, line['age'])), dtype=torch.int64) - 1, num_classes=6).view(-1)
        ret['target'] = line['target']
        ret['numeric'] = torch.cat([line['depth'], line['position'], line['gender'], line['age']], dim=0)

        line['keyword'] = torch.tensor([self.keyword_vocab[str(el)] for el in line['keyword'].split('|') if str(el) in self.keyword_vocab], dtype=torch.int64)
        line['title'] = torch.tensor([self.title_vocab[str(el)] for el in line['title'].split('|') if str(el) in self.title_vocab], dtype=torch.int64)
        line['description'] = torch.tensor([self.description_vocab[str(el)] for el in line['description'].split('|') if str(el) in self.description_vocab], dtype=torch.int64)

        ret['keyword'] = nn.ConstantPad1d((0, self.keyword_sequence_len - len(line['keyword'])), 0)(line['keyword'])
        ret['title'] = nn.ConstantPad1d((0, self.title_sequence_len - len(line['title'])), 0)(line['title'])
        ret['description'] = nn.ConstantPad1d((0, self.description_sequence_len - len(line['description'])), 0)(line['description'])
        
        return ret

    def preprocess(self, line):
        line = line.strip()
        line = line.split('\t')
        line = dict(zip(self.feature_names, line))
        line = self.preprocess_line(line)
        return line
    
    def __iter__(self):
        file_iter = open(self.file_path, 'r')
        next(file_iter)
        mapped_iter = map(self.preprocess, file_iter)

        return mapped_iter
