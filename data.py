import os
import torch
from torch.autograd import Variable
import pickle

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.word2count = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, mode = '', rare_limit = 5):
        self.dictionary = Dictionary()
        self.rare_limit = rare_limit
        
        train_path = os.path.join(path, mode + 'train.txt')
        valid_path = os.path.join(path, mode + 'valid.txt')
        test_path = os.path.join(path, mode + 'test.txt')
        
        tokens = self.add_dictionary([train_path, valid_path, test_path])
        self.dictionary = self.delete_rare()
        
        self.train = self.tokenize(train_path, tokens[0])
        self.valid = self.tokenize(valid_path, tokens[1])
        self.test = self.tokenize(test_path, tokens[2])

    def tokenize(self, path, tokens):
        """Tokenizes a text file."""
        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    if word in self.dictionary.word2idx:
                        ids[token] = self.dictionary.word2idx[word]
                        token += 1

        return ids[:token]
    
    def add_dictionary(self, paths):        
        # Add words to the dictionary  
        tokens = []
        for path in paths:
            assert os.path.exists(path)
            with open(path, 'r') as f:
                token_len = 0
                for line in f:
                    words = line.split() + ['<eos>']
                    token_len += len(words)
                    for word in words:
                        self.dictionary.add_word(word)
            tokens.append(token_len)
        return tokens
    
    def delete_rare(self):
        new_dict = Dictionary()
        new_dict.add_word('<pad>')
        for word in self.dictionary.word2idx:
            if self.dictionary.word2count[word] > self.rare_limit:
                new_dict.add_word(word)
        return new_dict


def batchify(data, batch_size):
    """The output should have size [L x batch_size], where L could be a long sequence length"""
    # Work out how cleanly we can divide the dataset into batch_size parts (i.e. continuous seqs).
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the batch_size batches.
    data = data.view(batch_size, -1)
    data = data.cuda()
    return data


def get_batch(source, i, seq_len=None, evaluation=False):
    seq_len = min(seq_len, source.size(1) - 1 - i)
    data = Variable(source[:, i:i+seq_len], volatile=evaluation)
    target = Variable(source[:, i+1:i+1+seq_len])     # CAUTION: This is un-flattened!
    return data, target
