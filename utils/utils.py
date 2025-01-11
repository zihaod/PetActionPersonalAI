# -*- coding: utf-8 -*-

import numpy as np
import math
import copy
import matplotlib.pyplot as plt
import sklearn
import os
import glob
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch
import json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# json reader
def read_json_data(fpath):
    with open(fpath, 'r') as f:
        data = json.load(f)
    return data


def get_json_files(data_dir):
    json_files = []

    # Walk through directory and all subdirectories
    for root, dirs, files in os.walk(data_dir):
        # Find all .json files in current directory
        for file in files:
            if file.lower().endswith('.json'):
                # Get full path and append to list
                full_path = os.path.join(root, file)
                json_files.append(full_path)

    return json_files




#Dataset
def list_minus(l1, l2):
    return [l2[i] - l1[i] for i in range(len(l1))]

def list_acc_minus(l1, l2):
    return [(l2[i] - l1[i])/100 if i < 3 else l2[i]/100 for i in range(len(l1))]

def list_cat(l1, l2):
    return [l1[i] + l2[i] for i in range(len(l1))]



class ActionDataset(Dataset):
    def __init__(self, pet_data, actions, seq_len=10, step_size=1, diff_mode='all', augment=None, aug_prob=0.5):
        self.augment = augment
        self.aug_prob = aug_prob
        self.pet_data = pet_data
        self.seq_len = seq_len
        self.step_size = step_size
        self.diff_mode = diff_mode
        self.actions = actions

        self.data = []
        self.labels = []

        for file_data in pet_data:
            curr_idx = seq_len
            while curr_idx <= len(file_data):
                x = [[d['acc_x'], d['acc_y'], d['acc_z'],
                      d['gry_x'], d['gry_y'], d['gry_z']]
                      for d in file_data[curr_idx-seq_len : curr_idx]]

                y = file_data[curr_idx-1]['label']
                self.data.append(x)
                self.labels.append(self.actions.index(y))
                curr_idx += step_size

        if self.diff_mode == "all":
            for i in range(len(self.data)):
                self.data[i] = [list_minus(self.data[i][j], self.data[i][j-1])
                                for j in range(1, len(self.data[i]))]

        elif self.diff_mode == "acc":
            for i in range(len(self.data)):
                self.data[i] = [list_acc_minus(self.data[i][j], self.data[i][j-1])
                                for j in range(1, len(self.data[i]))]



    def interpolate(self, X, missing_idxs):

        def prev_valid_idx(idx):
            for i in range(idx, -1, -1):
                if not X[i] in missing_idxs:
                    return i
        def next_valid_idx(idx):
            for i in range(idx, len(X), 1):
                if not X[i] in missing_idxs:
                    return i

        fill_values = {}
        for idx in missing_idxs:
            prev_idx = prev_valid_idx(idx)
            next_idx = next_valid_idx(idx)
            fill_values[idx] = (X[prev_idx] + (X[next_idx] - X[prev_idx]) / (next_idx - prev_idx))

        return fill_values



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = torch.tensor(self.data[idx])
        Y = self.labels[idx]

        if not self.augment:
            return X, Y

        else:
            p = torch.rand(1)[0]
            if p < self.aug_prob:
                aug_idxs = np.random.choice(list(range(1, len(X)-1)), 2, replace=False)

                if self.augment == 'avg':
                    aug_samples = self.interpolate(X, aug_idxs)
                    for aug_idx in aug_idxs:
                        X[aug_idx] = aug_samples[aug_idx]

            return X, Y


class ActionDatasetFromGroup(Dataset):
    def __init__(self, pet_data, actions):
        self.actions = actions

        self.data = pet_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Model

class AirModel(nn.Module):
    def __init__(self, num_class, input_size, hidden_size=30, num_layers=10):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:,-1]
        x = self.linear(x)
        return x

def export_lstm_to_bin(model_name, model, hidden_size):

    save_dir = f'./{model_name}/'
    os.makedirs(save_dir, exist_ok=True)

    for name, param in model.named_parameters():
        if param.requires_grad:
            tensor = param.data.unsqueeze(1) if param.data.dim() == 1 else param.data
            if 'lstm' in name:
                w_or_b = 'w' if 'weight' in name else 'b'
                i_or_h = 'i' if 'ih' in name else 'h'
                ifgo = {0: 'i', 1: 'f', 2: 'g', 3: 'o'}
                layer = name[-1]

                for i in range(4):
                    sub_tensor = tensor[hidden_size*i:hidden_size*(i+1)]
                    fname = w_or_b + i_or_h + ifgo[i] + layer + '.bin'
                    fname = save_dir + fname

                    tensor_bytes = sub_tensor.detach().cpu().numpy().tobytes()

                    with open(fname, 'wb') as f:
                        f.write(tensor_bytes)

            elif 'linear' in name:
                fname = 'w_proj.bin' if 'weight' in name else 'b_proj.bin'
                fname = save_dir + fname

                tensor_bytes = tensor.detach().cpu().numpy().tobytes()

                with open(fname, 'wb') as f:
                    f.write(tensor_bytes)



# Training and Evaluation
def evaluate(model, val_loader, device):
    criterion = torch.nn.CrossEntropyLoss()

    total = 0
    correct = 0
    losses = []
    predictions = []
    labels = []

    model.eval()

    print('-------Evaluation Start-------')
    for i_batch, data_batch in enumerate(val_loader):

            x_batch = data_batch[0].float().to(device)
            y_batch = data_batch[1].to(device)

            output = model(x_batch)
            loss = criterion(output, y_batch)

            preds = torch.argmax(output, dim=1)

            total += y_batch.shape[0]
            correct += torch.sum(preds==y_batch)

            losses.append(loss.item())
            predictions.append(preds.detach().cpu().numpy())
            labels.append(y_batch.detach().cpu().numpy())

    final_loss = sum(losses) / len(losses)
    acc = correct / total

    print('------------------------------')
    print(f'Val Loss: {final_loss}')
    print(f'Val Acc: {acc}')
    print('------------------------------')

    return np.concatenate(predictions), np.concatenate(labels)

def train(model, train_loader, val_loader, num_epoch, batch_size, lr, device='cuda'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.to(device)

    print('-------Training Start-------')
    for _epoch in range(num_epoch):
        print(f'Epoch: {_epoch}')
        losses = []
        model.train()

        for i_batch, data_batch in enumerate(train_loader):
            #print(f'  iter: {i_batch}/{len(train_loader)}')

            x_batch = data_batch[0].float().to(device)
            y_batch = data_batch[1].to(device)

            output = model(x_batch)
            loss = criterion(output, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
     

        train_loss = sum(losses) / len(losses)
        evaluate(model, val_loader, device)

        print('Epoch: ', _epoch, ' train_loss: ', train_loss)

