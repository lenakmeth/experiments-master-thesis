#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 18:29:48 2019

@author: lena
"""

import numpy as np
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.data import DataLoader
from masked_cross_entropy import compute_loss

from configure import parse_args
from utils import my_collate_fn

args = parse_args()
USE_CUDA = False


def train_step(
    src_batch,
    src_lens,
    trg_batch,
    trg_lens,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    criterion,
):

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # loss = 0 # Added onto for each word

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(src_batch, src_lens, None)

    if USE_CUDA:
        encoder_outputs = encoder_outputs
        encoder_hidden = encoder_hidden

    # Prepare input and output variables
    if USE_CUDA:
        decoder_input = Variable(torch.LongTensor([args.SOS_TOKEN] * args.batch_size))
        decoder_hidden = encoder_hidden[
            : decoder.n_layers
        ]  # Use last (forward) hidden state from encoder
    else:
        decoder_input = Variable(torch.LongTensor([args.SOS_TOKEN] * args.batch_size))
        decoder_hidden = encoder_hidden[: decoder.n_layers]

    max_trg_len = max(trg_lens)
    all_decoder_outputs = Variable(
        torch.zeros(max_trg_len, args.batch_size, decoder.output_size)
    )

    # Move new Variables to CUDA
    if USE_CUDA:
        decoder_input = decoder_input
        all_decoder_outputs = all_decoder_outputs

    # Run through decoder one time step at a time
    for t in range(max_trg_len):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )

        all_decoder_outputs[t] = decoder_output
        decoder_input = trg_batch[t]  # Next input is current target

    # Loss calculation and backpropagation
    loss = compute_loss(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        trg_batch.transpose(0, 1).contiguous(),  # -> batch x seq
        trg_lens,
    )
    loss.backward()

    # Clip gradient norms
    enc_grads = torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.clip)
    dec_grads = torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()

    #    return loss.data[0]#, enc_grads, dec_grads
    return loss.item()


def save_checkpoint(encoder, decoder, n_epoch, checkpoint_dir, lang):
    enc_filename = "{}/{}/enc-{}-ep{}.pth".format(
        checkpoint_dir, lang, time.strftime("%d%m%y-%H%M%S"), n_epoch
    )
    dec_filename = "{}/{}/dec-{}-ep{}.pth".format(
        checkpoint_dir, lang, time.strftime("%d%m%y-%H%M%S"), n_epoch
    )
    torch.save(encoder.state_dict(), enc_filename)
    torch.save(decoder.state_dict(), dec_filename)
    print("Model saved.")


def train(
    dataset,
    batch_size,
    n_epochs,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    criterion,
    checkpoint_dir,
    lang,
    save_every=2000,
):
    train_iter = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=my_collate_fn,
    )
    for n_epoch in range(n_epochs):
        tick = time.clock()
        print("Epoch {}/{}".format(n_epoch + 1, n_epochs))
        losses = []
        for batch_idx, batch in enumerate(train_iter):
            input_batch, input_lengths, target_batch, target_lengths = batch

            if USE_CUDA:
                input_batch = input_batch.cuda()
                input_lengths = input_lengths.cuda()
                target_batch = target_batch.cuda()
                target_lengths = target_lengths.cuda()

            if input_batch.size()[1] == batch_size:
                loss = train_step(
                    input_batch,
                    input_lengths,
                    target_batch,
                    target_lengths,
                    encoder,
                    decoder,
                    encoder_optimizer,
                    decoder_optimizer,
                    criterion,
                )
                losses.append(loss)
                if batch_idx % 100 == 0:
                    print("batch: {}, loss: {}".format(batch_idx, loss))

        # save at the end of epoch
        if checkpoint_dir:
            save_checkpoint(encoder, decoder, n_epoch + 1, checkpoint_dir, lang)
        tock = time.clock()
        print("Time: {} Avg loss: {}".format(tock - tick, np.mean(losses)))

    if checkpoint_dir:
        save_checkpoint(encoder, decoder, n_epoch + 1, checkpoint_dir, lang)
