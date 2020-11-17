import numpy as np
import torch


def my_collate_fn(data):

    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).
    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).
    Args:
        data: list of tuple (src_seq, trg_seq).
            - src_seq: torch tensor of shape (?); variable length.
            - trg_seq: torch tensor of shape (?); variable length.
    Returns:
        src_seqs: torch tensor of shape (batch_size, padded_length).
        src_lengths: list of length (batch_size); valid length for each padded source sequence.
        trg_seqs: torch tensor of shape (batch_size, padded_length).
        trg_lengths: list of length (batch_size); valid length for each padded target sequence.

    Code adapted from https://github.com/yunjey/seq2seq-dataloader
    """

    def _pad_sequences(seqs):
        lens = [len(seq) for seq in seqs]
        padded_seqs = torch.zeros(len(seqs), max(lens)).long()
        for i, seq in enumerate(seqs):
            end = lens[i]
            padded_seqs[i, :end] = torch.LongTensor(seq[:end])
        return padded_seqs, lens

    data.sort(key=lambda x: len(x[0]), reverse=True)
    src_seqs, trg_seqs = zip(*data)
    src_seqs, src_lens = _pad_sequences(src_seqs)
    trg_seqs, trg_lens = _pad_sequences(trg_seqs)

    # (batch, seq_len) => (seq_len, batch)
    src_seqs = src_seqs.transpose(0, 1)
    trg_seqs = trg_seqs.transpose(0, 1)

    return src_seqs, src_lens, trg_seqs, trg_lens
