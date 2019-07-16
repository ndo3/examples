import os
from io import open
import torch

import json

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
        # adding the perplexity data to evaluate perplexity of - the file would be json :P
        json_file_path = os.path.join(path, 'perplexity.json')
        self.perplexity = []
        self.id_to_text = {}
        txt_file_path = os.path.join(path, 'currenttxtfile.txt')
        with open(json_file_path, "rb", encoding="utf-8") as json_file:
            json_data = json.load(json_file)
            for id, item in enumerate(json_data):
                self.id_to_text[id] = item
                with open(txt_file_path, "w+") as current_txt_file:
                    current_txt_file.write(item['text'])
                    self.perplexity.append(self.tokenize(txt_file_path))
                    # and then after calculating the perplexity score I would delete the file
                    current_txt_file.truncate()
        # so by this point self.perplexity is going to be a list of token ids
        # write things to idtotext.json so that we can keep track of things
        with open("idtotext.json", "w+") as id_to_text_file:
            json.dump(self.id_to_text, id_to_text_file)

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        return ids
