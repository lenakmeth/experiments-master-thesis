a  #!/usr/bin/env python3
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
from dataset_Exp3 import Dataset
from models import EncoderRNN, Attn, LuongAttnDecoderRNN
from train import train
from decode import decode_dataset, evaluate

args = parse_args()
lang = "ang"
# lang  = args.lang

langs = [
    #        'ang',
    # 'ang_char',
    # 'dan',
    # 'dan_char',
    # 'deu',
    # 'deu_char',
    "est",
    "est_char",
    # 'fao',
    # 'fao_char',
    # 'fin',
    # 'fin_char',
    # 'gle',
    # 'gle_char',
    # 'hun',
    # 'hun_char',
    # 'kat',
    # 'kat_char',
    # 'lat',
    # 'lat_char',
    # 'lav',
    # 'lav_char',
    ## 'lit',
    ## 'lit_char',
    ## 'mkd',
    ## 'mkd_char',
    ## 'pol',
    ## 'pol_char',
    # 'sme',
    # 'sme_char',
    ## 'spa',
    ## 'spa_char',
    ## 'swe',
    ## 'swe_char',
    # 'syc',
    # 'syc_char'
]

# if not os.path.exists('checkpoints/pov/' + lang):
#    os.mkdir('checkpoints/pov/' + lang)

decoded_words_unseen = {}
decoded_words = {}
wrong_preds = {}
wrong_preds_unseen = {}


for lang in langs:
    ## Initialize dataset
    dataset = Dataset("data/" + lang + "/" + lang + "_train.txt", train=True)  # toy?

    # Initialize dataset
    dataset = Dataset("data/" + lang + "/" + lang + "_train.txt", train=True)  # toy?
    print(len(dataset.inputs))

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
    #      args.batch_size,
    #      args.n_epochs,
    #      encoder,
    #      decoder,
    #      encoder_optimizer,
    #      decoder_optimizer,
    #      criterion,
    #      'checkpoints/pov',
    #      lang)

    # evaluate

    # find the last encoder state
    encoder_last_state = sorted(
        [
            x
            for x in os.listdir(
                "/Users/lena/Desktop/thesis-more/checkpoints/pov/" + lang
            )
            if x.startswith("enc")
        ]
    )[-1]
    print(encoder_last_state)
    # find the last decoder state
    decoder_last_state = sorted(
        [
            x
            for x in os.listdir(
                "/Users/lena/Desktop/thesis-more/checkpoints/pov/" + lang
            )
            if x.startswith("dec")
        ]
    )[-1]
    print(decoder_last_state)

    encoder.load_state_dict(
        torch.load(
            "/Users/lena/Desktop/thesis-more/checkpoints/pov/"
            + lang
            + "/"
            + encoder_last_state
        )
    )
    decoder.load_state_dict(
        torch.load(
            "/Users/lena/Desktop/thesis-more/checkpoints/pov/"
            + lang
            + "/"
            + decoder_last_state
        )
    )

    # predict for test

    figs_path = "figs/" + lang + "/pov_test"
    # if not os.path.exists(figs_path):
    #    os.makedirs(figs_path)

    decoded_words[lang] = decode_dataset(
        "data/" + lang + "/" + lang + "_test.txt", encoder, decoder, dataset, figs_path
    )
    print("test results")
    wrong_preds[lang] = evaluate(decoded_words[lang])


###predict for unseen
##figs_path = 'figs/' + lang + '/pov_unseen'
###if not os.path.exists(figs_path):
###    os.makedirs(figs_path)
##
##decoded_words2 = decode_dataset("data/" + lang + '/' + lang + "_unseen.txt",
##                                 encoder, decoder, dataset, figs_path)
##print('unseen results')
##evaluate(decoded_words2)
#
#
##with open(lang + '-exp3_is_done.txt', 'w') as f:
#    f.write("i'm done bitch")
