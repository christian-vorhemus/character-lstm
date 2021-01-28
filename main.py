import glob
import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import torch.autograd as autograd
import os
import string
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import hashlib

# https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

all_letters = string.ascii_letters + "._1234567890"
n_letters = len(all_letters)
maximum_word_length = 15


def letter_index(letter):
    return all_letters.find(letter)


def line_tensor(line):
    l_list = [char for char in all_letters]
    tensor = torch.zeros(maximum_word_length, n_letters)
    for li, letter in enumerate(line):
        if letter not in l_list:
            continue
        if li > maximum_word_length-1:
            break
        tensor[li][letter_index(letter)] = 1
    return tensor


class TextDataset(Dataset):
    def __init__(self, path_to_csv, vocab_size, sequence_length=40, delimiter=","):
        self.__classes = []
        self.__texts = []
        with open(path_to_csv, encoding='utf-8') as csvfile:
            rows = csv.reader(csvfile, delimiter=delimiter)
            headers = next(rows, None)
            for row in rows:
                if(row[1] == '1'):
                    self.__classes.append(1)
                    X = line_tensor(row[0])
                    self.__texts.append(X)
                else:
                    self.__classes.append(0)
                    X = line_tensor(row[0])
                    self.__texts.append(X)

    def __len__(self):
        return len(self.__texts)

    def __getitem__(self, index):
        textvector = self.__texts[index]

        label = self.__classes[index]
        return textvector, torch.tensor(label, dtype=torch.int64)


class LSTMClassifier(nn.Module):
    def __init__(self, hidden_dim):
        super(LSTMClassifier, self).__init__()
        output_size = 2
        self.embedding_dim = n_letters
        self.hidden_dim = hidden_dim
        self.vocab_size = maximum_word_length

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim,
                            num_layers=1, batch_first=True)

        self.hidden2out = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.dropout_layer = nn.Dropout(p=0.2)

    def init_hidden(self, batch_size):
        return(autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
               autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)))

    def forward(self, batch):
        batch_size = batch.size(0)
        self.hidden = self.init_hidden(batch_size)

        outputs, (ht, ct) = self.lstm(batch, self.hidden)

        output = self.dropout_layer(ht[-1])
        output = self.hidden2out(output)
        output = self.softmax(output)

        return output


def predict(username):
    model = torch.load("./name_char_model.pt")
    model.eval()

    vec = line_tensor(username)

    # Add batch dimension
    model_input = torch.tensor(vec)
    model_input = model_input.unsqueeze(0)

    pred = model(model_input)
    pred_idx = torch.max(pred, 1)[1]
    prediction = pred_idx.data[0].item()

    return prediction


def train():
    minibatch_size = 20
    epochs = 50
    hidden_dim = 32
    char_dim = 26
    char_vocab_size = 1000

    # Input is one-hot vectors of char_length characters.

    train_dataset = TextDataset(
        path_to_csv="./data/namegender_training.csv", vocab_size=char_vocab_size)
    test_dataset = TextDataset(
        path_to_csv="./data/namegender_test.csv", vocab_size=char_vocab_size)

    # tensor is 15x52 (15 max length and 52 is allowed ASCII characters)
    a = train_dataset.__getitem__(0)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=minibatch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=minibatch_size, shuffle=True)

    model = LSTMClassifier(hidden_dim)

    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss(size_average=False)
    #criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print('Epoch:', epoch)
        y_true = list()
        y_pred = list()
        total_loss = 0
        for batch_index, (texts, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            pred = model(texts)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            pred_idx = torch.max(pred, 1)[1]
            y_true += list(labels.int())
            y_pred += list(pred_idx.data.int())
            total_loss += loss

        acc = accuracy_score(y_true, y_pred)

        y_true = list()
        y_pred = list()
        total_test_loss = 0
        for test_texts, test_labels in test_loader:
            pred = model(test_texts)
            loss = criterion(pred, test_labels)
            pred_idx = torch.max(pred, 1)[1]
            y_true += list(test_labels.int())
            y_pred += list(pred_idx.data.int())
            total_test_loss += loss

        val_acc = accuracy_score(y_true, y_pred)
        val_loss = total_test_loss.data.float()/len(test_loader)

        print("Train loss: {} - acc: {}, Validation loss: {} - acc: {}".format(
            total_loss.data.float()/len(train_loader), acc, val_loss, val_acc))

    torch.save(model, "./name_char_model.pt")
    print("Finished training")


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train()
    else:
        pred = predict(sys.argv[1])
        if pred == 0:
            gender = "male"
        else:
            gender = "female"
        print(
            f"{sys.argv[1]} is probably a {gender} name")
