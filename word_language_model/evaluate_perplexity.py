# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

import data
import model

import json

parser = argparse.ArgumentParser(description='Parser to evaluate perplexity')
parser.add_argument('--orgdata', default='./data/wikitext-2', help='original dataset so that we know what the corpus is')
parser.add_argument('--model', default='model.pt', help='directory of the .pt file that is the best saved model')
parser.add_argument('--data', default='data/original_tweets.json', help='dir de json file no que the texts que model is evaluated')
args = parser.parse_args()

# open the model
with open(args.model, 'rb') as f, open(args.data, 'rb') as d:
    # use PyTorch to load the model
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()
    # Load the texts that we are going to evaluate perplexity with
    data = json.load(d)
    data = [t['text'] for t in data]
    # load corpus
    corpus = data.Corpus(args.orgdata)
    # the same original batch size like the original file
    eval_batch_size = 10
    # run evaluate perplexity against our list of texts
    evaluate_perplexity(data)
        

def evaluate_perplexity(input_texts):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    pass