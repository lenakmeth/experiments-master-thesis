#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 18:33:08 2019

@author: lena
"""

import os
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from masked_cross_entropy import *

from configure import parse_args
from dataset import Dataset
from utils import my_collate_fn

from Levenshtein import ratio

import matplotlib.pyplot as plt

plt.switch_backend("agg")

import matplotlib.ticker as ticker
import numpy as np

args = parse_args()


def decode(input_seq, input_len, encoder, decoder, in_vocab, out_vocab, max_length=40):
    with torch.no_grad():
        # input_lengths = [len(input_seq)]
        # input_seqs = [indexes_from_sentence(input_lang, input_seq)]
        # input_batches = Variable(torch.LongTensor(input_seqs)).transpose(0, 1)
        input_seq = Variable(input_seq)

        if args.USE_CUDA:
            input_seq = input_seq.cuda()

        # Set to not-training mode to disable dropout
        encoder.train(False)
        decoder.train(False)

        # Run through encoder
        encoder_outputs, encoder_hidden = encoder(input_seq, input_len, None)

        # Create starting vectors for decoder
        decoder_input = Variable(torch.LongTensor([args.SOS_TOKEN]))  # SOS
        decoder_hidden = encoder_hidden[
            : decoder.n_layers
        ]  # Use last (forward) hidden state from encoder

        if args.USE_CUDA:
            decoder_input = decoder_input.cuda()

        # Store output words and attention states
        decoded_chars = []
        decoder_attentions = torch.zeros(max_length + 1, max_length + 1)

        # Run through decoder
        for t in range(max_length):

            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            #            decoder_attentions[t,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

            # Choose top word from output
            prob, token_idx = decoder_output.data.topk(1)
            tok = token_idx[0][0].item()
            if tok == args.EOS_TOKEN:
                break
            else:
                try:
                    decoded_chars.append(out_vocab[tok])
                except KeyError:
                    pass

            # Next input is chosen word
            decoder_input = Variable(torch.LongTensor([tok]))
            if args.USE_CUDA:
                decoder_input = decoder_input.cuda()

        # Set back to training mode
        encoder.train(True)
        decoder.train(True)

        return (
            " ".join(decoded_chars),
            decoder_attentions[: t + 1, : len(encoder_outputs)],
        )


def decode_dataset(file_name, encoder, decoder, train_dataset, figs_path):

    test_dataset = Dataset(file_name, train=False)
    test_dataset.in_vocab = train_dataset.in_vocab
    test_dataset.out_vocab = train_dataset.out_vocab
    in_vocab = train_dataset.in_vocab[0]
    out_vocab = train_dataset.out_vocab[0]
    test_iter = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=my_collate_fn,
    )

    decoded_words = []
    attention_counter = 0
    for input_seq, input_len, target_seq, _ in test_iter:
        decoded_word, attentions = decode(
            input_seq, input_len, encoder, decoder, in_vocab, out_vocab
        )
        #        # plot attention
        #        attention_counter += 1
        #        if attention_counter < 1000:
        #            show_attention([in_vocab[int(i)] for i in input_seq],
        #                          decoded_word, attentions, figs_path)

        decoded_words.append(
            [
                " ".join([in_vocab[int(i)] for i in input_seq]),
                " ".join([out_vocab[int(i)] for i in target_seq]),
                decoded_word,
            ]
        )

    print()

    return decoded_words


def show_attention(inputs, prediction, attentions, figs_path):
    # Set up figure with colorbar

    fig = plt.figure()
    #    fig.subplots_adjust(bottom=2)
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap="gray")
    fontdict = {"fontsize": 14}
    plt.subplots_adjust(bottom=0.1)
    plt.subplots_adjust(top=1)
    #    fig.subplots_adjust(bottom=-1)
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([""] + inputs + ["</w>"], fontdict=fontdict, rotation=90)
    ax.set_yticklabels([""] + prediction.split(" "), fontdict=fontdict)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    #    plt.show()
    plt.savefig(
        figs_path + "/" + prediction + ".png", bbox_inches="tight", pad_inches=1
    )
    plt.close()


# def plot_attention(sentence, predicted_sentence, attention):
#    fig = plt.figure(figsize=(10,10))
#    ax = fig.add_subplot(1, 1, 1)
#    ax.matshow(attention, cmap='viridis') #bone
#
#    fontdict = {'fontsize': 14}
#
#    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
#    ax.set_yticklabels([''] + predicted_sentence.split(' '), fontdict=fontdict)
#    plt.show()
#    plt.savefig('figs/' + predicted_sentence +'.png')
#    plt.close()


def evaluate(decoded_words, lang):
    print("Length of test set:\t", str(len(decoded_words)))

    wrong_preds = []
    right = 0
    wrong = 0
    lev_percentage = 0
    for word_set in decoded_words:

        expected = word_set[1].replace(" ", "").replace("</w>", "")
        expected = expected.replace("s_", "").replace("p_", "")
        predicted = word_set[2].replace(" ", "").replace("</w>", "")
        predicted = predicted.replace("s_", "").replace("p_", "")
        #        print(expected, predicted)
        similarity = ratio(expected, predicted)

        if similarity == 1.0:
            right += 1

        else:
            wrong += 1
            #            print('word_set: ',expected, predicted)
            wrong_preds.append((expected, predicted))

        lev_percentage += similarity

    stats = {}
    stats["total num"] = len(decoded_words)
    stats["no or rights"] = right
    stats["no of wrongs"] = wrong
    stats["acc"] = right / len(decoded_words)
    stats["lev"] = lev_percentage / len(decoded_words)

    print("total num:\t", len(decoded_words))
    print("no or rights:\t", str(right))
    print("no of wrongs:\t", str(wrong))
    print("acc:\t", str(right / len(decoded_words)))
    print("lev:\t", str(lev_percentage / len(decoded_words)))

    filename = os.getcwd() + "/results/{}-{}.txt".format(
        lang, time.strftime("%d%m%y-%H%M%S")
    )
    with open(filename, "w", encoding="utf-8") as f:
        f.write("total num:\t" + str(len(decoded_words)) + '\n')
        f.write("no or rights:\t" + str(right) + '\n')
        f.write("no of wrongs:\t" + str(wrong) + '\n')
        f.write("acc:\t" + str(right / len(decoded_words)) + '\n')
        f.write("lev:\t" + str(lev_percentage / len(decoded_words)) + '\n')

    return wrong_preds


#    return stats
