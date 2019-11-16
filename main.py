#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 14:14:38 2019

@author: lena
"""

import os
import json
import numpy as np
import time
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.data import DataLoader

from configure import parse_args
from dataset import Dataset
from models import EncoderRNN, Attn, LuongAttnDecoderRNN
from train import train
from decode import decode_dataset, evaluate

args = parse_args()
lang = args.lang
print(lang, "\n")

decoded_words_unseen = {}
decoded_words = {}
wrong_preds = {}
wrong_preds_unseen = {}


if not os.path.exists("checkpoints/" + lang):
    os.mkdir("checkpoints/" + lang)

# Initialize dataset
dataset = Dataset("data/" + lang + "/" + lang + "_train.txt", train=True)  # toy?


# Initialize models
encoder = EncoderRNN(
    len(dataset.in_vocab[0]),
    args.embed_size,
    args.hidden_size,
    args.n_layers,
    args.dropout,
)
decoder = LuongAttnDecoderRNN(
    args.attn_model,
    args.hidden_size,
    len(dataset.out_vocab[0]),
    args.n_layers,
    args.dropout,
)

# Initialize optimizers and criterion
# encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate)
# decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.learning_rate * decoder_learning_ratio)
encoder_optimizer = optim.Adadelta(encoder.parameters())
decoder_optimizer = optim.Adadelta(decoder.parameters())
criterion = nn.CrossEntropyLoss()

# Move models to GPU
if args.USE_CUDA:
    encoder.cuda()
    decoder.cuda()

# train(dataset,
#        args.batch_size,
#        args.n_epochs,
#        encoder,
#        decoder,
#        encoder_optimizer,
#        decoder_optimizer,
#        criterion,
#        args.checkpoint_dir,
#        lang)

# evaluate

# find the last encoder state
encoder_last_state = sorted(
    [x for x in os.listdir("checkpoints/" + lang) if x.startswith("enc")]
)[-1]
print(encoder_last_state)

# find the last decoder state
decoder_last_state = sorted(
    [x for x in os.listdir("checkpoints/" + lang) if x.startswith("dec")]
)[-1]
print(decoder_last_state)

encoder.load_state_dict(torch.load("checkpoints/" + lang + "/" + encoder_last_state))
decoder.load_state_dict(torch.load("checkpoints/" + lang + "/" + decoder_last_state))


# predict for test

# predict for unseen
figs_path = "figs/" + lang + "/test"
if not os.path.exists(figs_path):
    os.makedirs(figs_path)

decoded_words_test = decode_dataset(
    "data/" + lang + "/" + lang + "_test.txt", encoder, decoder, dataset, figs_path
)
print("test results")

wrong_preds = evaluate(decoded_words_test, lang)

# predict for unseen
figs_path = "figs/" + lang + "/unseen"
if not os.path.exists(figs_path):
    os.makedirs(figs_path)

decoded_words_unseen = decode_dataset(
    "data/" + lang + "/" + lang + "_unseen.txt", encoder, decoder, dataset, figs_path
)
print("unseen results")

wrong_preds_unseen = evaluate(decoded_words_unseen, lang)

# VERY FANCY DECODE

# full_stats = {}
# for num in range(19,args.n_epochs):
#
#    enc_state = sorted([x for x in os.listdir('checkpoints/' + lang)
#                             if x.startswith('enc')])[num-1]
#    dec_state = sorted([x for x in os.listdir('checkpoints/' + lang)
#                             if x.startswith('dec')])[num-1]
#
#    encoder.load_state_dict(torch.load('checkpoints/' + lang + '/' + enc_state))
#    decoder.load_state_dict(torch.load('checkpoints/' + lang + '/' + dec_state))
#
#
#    #figs_path = 'figs/' + lang + '/test'
#    #if not os.path.exists(figs_path):
#    #    os.makedirs(figs_path)
#
#    decoded_words[lang] = decode_dataset("data/" + lang + '/' + lang + "_test.txt",
#                                     encoder, decoder, dataset, figs_path=None)
#    print('test results')
#    try:
#        full_stats[num] = evaluate(decoded_words[lang])
#    except Exception:
#        pass
#
# max_acc = 0
# max_lev = 0
# for epoch,stats in full_stats.items():
#    try:
#        if stats['acc'] > max_acc:
#            max_acc = stats['acc']
#            print('max_acc', epoch, stats)
#        if stats['lev'] > max_lev:
#            max_lev = stats['lev']
#            print('max_lev', epoch, stats)
#    except KeyError:
#        pass

# full_unseen = {}
# for num in range(20):
#    print()
#    print(num)
#    enc_state = sorted([x for x in os.listdir('checkpoints/' + lang)
#                             if x.startswith('enc')])[num-1]
#    dec_state = sorted([x for x in os.listdir('checkpoints/' + lang)
#                             if x.startswith('dec')])[num-1]
#    print(enc_state)
#    print(dec_state)
#    encoder.load_state_dict(torch.load('checkpoints/' + lang + '/' + enc_state))
#    decoder.load_state_dict(torch.load('checkpoints/' + lang + '/' + dec_state))
#
#
#    #figs_path = 'figs/' + lang + '/test'
#    #if not os.path.exists(figs_path):
#    #    os.makedirs(figs_path)
#
#    decoded_words[lang] = decode_dataset("data/" + lang + '/' + lang + "_unseen.txt",
#                                     encoder, decoder, dataset, figs_path=None)
#    print('unseen results')
#    full_unseen[num] = evaluate(decoded_words[lang])
#
# max_acc_u = 0
# max_lev_u = 0
# for epoch,stats in full_unseen.items():
#    try:
#        if stats['acc'] > max_acc:
#            max_acc = stats['acc']
#            print('max_acc', epoch, stats)
#        if stats['lev'] > max_lev:
#            max_lev = stats['lev']
#            print('max_lev', epoch, stats)
#    except KeyError:
#        pass
