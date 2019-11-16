import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class Dataset(Dataset):
    def __init__(self, file_name, train=True):
        if train:
            train_threshold = 3000
            inputs, outputs = load_dataset(file_name, train_threshold)
            inputs, outputs, in_vocab, out_vocab = preprocess_data(
                inputs, outputs, train=True
            )
            self.inputs = inputs
            self.outputs = outputs
            self.in_vocab = in_vocab
            self.out_vocab = out_vocab
        else:
            inputs, outputs = load_dataset(file_name, 500)
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
UNK_TOKEN = "<unk>"


def load_dataset(file_name, train_threshold):
    inputs = []
    outputs = []
    counter = 0
    with open(file_name) as f:
        for line in f:
            counter += 1
            if counter < train_threshold + 1:
                try:
                    l = line.strip().split("\t")
                    l[0] = l[0].lower()  # source
                    l[2] = l[2].lower()  # target

                    #                    l[0] = ''.join(reversed(list(l[0].lower()))) # source
                    #                    l[2] = ''.join(reversed(list(l[2].lower()))) # target

                    l[1] = l[1].split(";")  # source features
                    l[3] = l[3].split(";")  # target features
                    inputs.append([l[1], l[0], l[3]])
                    outputs.append(l[2])
                except IndexError:
                    # then the line is probably missing some features
                    pass
    print("og len: " + str(len(inputs)))
    return np.array(inputs), np.array(list(outputs))


def enhance_dataset(inputs, outputs):
    inputs_cpy = inputs.copy()
    outputs_cpy = outputs.copy()
    inputs_cpy[:, [0, 2]] = inputs_cpy[:, [2, 0]]
    inputs_cpy[:, 1], outputs_cpy[:] = outputs_cpy[:], inputs_cpy[:, 1]
    inputs = np.concatenate((inputs, inputs_cpy), axis=0)
    outputs = np.concatenate((outputs, outputs_cpy), axis=0)
    return inputs, outputs


def preprocess_data(inputs, outputs, train):

    if train:
        if len(inputs) < 20000:
            inputs, outputs = enhance_dataset(inputs, outputs)
        inputs = edit_tags(inputs)
        inputs[:, [1, 2]] = inputs[:, [2, 1]]
        inputs = transform_to_sequences(inputs)
        input_vocab = make_input_vocab(inputs)
        output_vocab = make_output_vocab(outputs)
        return inputs, outputs, input_vocab, output_vocab
    else:
        inputs = edit_tags(inputs)
        inputs[:, [1, 2]] = inputs[:, [2, 1]]
        inputs = transform_to_sequences(inputs)
        return inputs, outputs


def edit_tags(inputs):
    for i in range(0, inputs.shape[0]):
        inputs[i, 0] = np.array(["IN=" + x for x in inputs[i, 0]])
        inputs[i, 2] = np.array(["OUT=" + x for x in inputs[i, 2]])
    return inputs


def transform_to_sequences(inputs):
    input_seq = np.array(
        [
            np.concatenate((inputs[i, 0], inputs[i, 1], list(inputs[i, 2].split(" "))))
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


def get_input_indices(inputs, vocab):
    v = []
    for ch in inputs:
        try:
            v.append(vocab[ch])
        except KeyError:
            v.append(vocab[UNK_TOKEN])
    return v + [vocab[EOS_TOKEN]]


def get_output_indices(output, vocab):
    v = []
    for ch in output.split(" "):
        try:
            v.append(vocab[ch])
        except KeyError:
            v.append(vocab[UNK_TOKEN])
    return v + [vocab[EOS_TOKEN]]
