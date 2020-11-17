import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class Dataset(Dataset):
    def __init__(self, file_name, train=True):
        if train:
            train_threshold = 300000
            inputs, outputs = load_dataset(file_name, train_threshold)
            inputs, outputs, in_vocab, out_vocab = preprocess_data(
                inputs, outputs, train=True
            )
            self.inputs = inputs
            self.outputs = outputs
            self.in_vocab = in_vocab
            self.out_vocab = out_vocab
        else:
            inputs, outputs = load_dataset(file_name, 0)
            inputs, outputs = preprocess_data(inputs, outputs, train=False)
            self.inputs = inputs
            self.outputs = outputs

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        src = get_input_indices(self.inputs[index], self.in_vocab[1])
        trg = get_output_indices(self.outputs[index], self.out_vocab[1])
        return src, trg


PAD_TOKEN = "<pad>"
SOS_TOKEN = "<w>"
EOS_TOKEN = "</w>"
UNK_TOKEN = "U"


def load_dataset(file_name, train_threshold):
    inputs = []
    outputs = []
    counter = 0
    with open(file_name) as f:
        for line in f:
            counter += 1
            #            if counter < train_threshold:
            try:
                l = line.strip().split("\t")
                l[0] = l[0].lower().split(" ")  # source
                #                l[0] = ''.join(list(reversed(l[0].lower())))
                l[2] = l[2].lower()  # target
                #                l[2] = ''.join(list(reversed(l[2].lower())))
                l[1] = l[1].split(";")  # source features
                l[3] = ["OUT=" + i for i in l[3].split(";")]  # target features
                inputs.append([l[3], l[2]])
                outputs.append(l[2])

            except IndexError:
                # then the line is probably missing some features
                pass

    return np.array(list(inputs)), np.array(list(outputs))


def enhance_dataset(inputs, outputs):
    inputs_cpy = inputs.copy()
    outputs_cpy = outputs.copy()
    inputs = np.concatenate((inputs, inputs_cpy), axis=0)
    outputs = np.concatenate((outputs, outputs_cpy), axis=0)
    return inputs, outputs


def preprocess_data(inputs, outputs, train):

    if train:
        if len(inputs) < 20000:
            inputs, outputs = enhance_dataset(inputs, outputs)
        inputs = transform_to_sequences(inputs)
        input_vocab = make_input_vocab(inputs)
        output_vocab = make_output_vocab(outputs)
        return inputs, outputs, input_vocab, output_vocab
    else:
        inputs = transform_to_sequences(inputs)
        return inputs, outputs



def transform_to_sequences(inputs):
    input_seq = np.array(
        [
            np.concatenate((inputs[i, 0], list(inputs[i, 1])))
            for i in range(inputs.shape[0])
        ]
    )
    return input_seq


def make_input_vocab(data):
    idx_to_char = {0: PAD_TOKEN, 1: SOS_TOKEN, 2: EOS_TOKEN, 3: UNK_TOKEN}
    char_to_idx = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
    char_set = set([])
    for i in range(0, data.shape[0]):
        char_set.update(data[i])
    char_set = sorted(char_set)
    for i in range(0, len(char_set)):
        idx_to_char[i + 4] = char_set[i]
        char_to_idx[char_set[i]] = i + 4
    return idx_to_char, char_to_idx


def make_output_vocab(data):
    idx_to_char = {0: PAD_TOKEN, 1: SOS_TOKEN, 2: EOS_TOKEN, 3: UNK_TOKEN}
    char_to_idx = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
    char_set = set([])
    for i in range(0, data.shape[0]):
        char_set.update(data[i].split(" "))
    char_set = sorted(char_set)
    for i in range(0, len(char_set)):
        idx_to_char[i + 4] = char_set[i]
        char_to_idx[char_set[i]] = i + 4
    return idx_to_char, char_to_idx

def get_input_indices(input, vocab):
    v = [vocab[ch] for ch in input] + [vocab[EOS_TOKEN]]
    # v[3] = "ö"

    return v


def get_output_indices(output, vocab):
    v = [vocab[ch] for ch in output.split(" ")] + [vocab[EOS_TOKEN]]
    # v[3] = "ö"

    return v
