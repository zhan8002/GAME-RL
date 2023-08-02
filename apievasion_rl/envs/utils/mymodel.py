#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import re
import hashlib
import json
import time
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import StratifiedShuffleSplit



epoches = 50
batch_size = 32
learning_rate = 0.0001
max_length = 200
bichoice = True

is_support = torch.cuda.is_available()
if is_support:
    device = torch.device('cuda:1')

# In[ ]:

class APIDataset(Dataset):
    def __init__(self, datasets, labels, batch_size=1, dim=max_length):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.datasets = datasets


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.labels)))

    def __getitem__(self, index):

        X, y = self.__data_generation(index)

        return X, y

    def __data_generation(self, index):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.dim, 8), dtype=float)
        # y = np.zeros(self.batch_size, dtype=float)

        # Generate data

        base_path = "/home/ubuntu/zhan/code/zhan/API/detector/dataset/dataset/dataset/{0}.npy"
        item = self.datasets.iloc[index]
        self_name = item['file_name']
        tmp = np.load(base_path.format(self_name))
        tmp = np.clip(tmp, -100, 100)
        if tmp.shape[0] > self.dim:
            X = tmp[:self.dim, :8]
        else:
            X[:tmp.shape[0], :] = tmp[:, :8]
        y = self.labels[index]

        return X, y

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bichoice):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.bichoice = bichoice
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=layer_dim, nonlinearity='relu', batch_first=True, bidirectional=bichoice)
        # 全连接层
        if bichoice:
            self.fc = nn.Linear(hidden_dim*2, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.to(torch.float32)
        if self.bichoice:
            # layer_dim, batch_size, hidden_dim
            h0 = torch.zeros(self.layer_dim*2, x.size(0), self.hidden_dim).requires_grad_().to(device)
        else:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
            # 分离隐藏状态，避免梯度爆炸
        out, hn = self.rnn(x, h0.detach())
        out = self.fc(out[:, -1, ])
        return out

class LSTM(nn.Module):
    """搭建LSTM神经网络"""
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bichoice):

        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.bichoice = bichoice
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=bichoice)
        # 全连接层
        if bichoice:
            self.fc = nn.Linear(hidden_dim*2, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.to(torch.float32)
        if self.bichoice:
            # layer_dim, batch_size, hidden_dim
            h0 = torch.zeros(self.layer_dim*2, x.size(0), self.hidden_dim).requires_grad_().to(device)
            # 初始化cell, state
            c0 = torch.zeros(self.layer_dim*2, x.size(0), self.hidden_dim).requires_grad_().to(device)
            # 分离隐藏状态，避免梯度爆炸
            out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
            out = self.fc(out[:, -1, :])
        else:
            # layer_dim, batch_size, hidden_dim
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
            # 初始化cell, state
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
            # 分离隐藏状态，避免梯度爆炸
            out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
            out = self.fc(out[:, -1, :])
        return out

class GRU(nn.Module):
    """搭建GRU神经网络"""
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bichoice):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.bichoice = bichoice
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=bichoice)
        # 全连接层
        if bichoice:
            self.fc = nn.Linear(hidden_dim*2, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.to(torch.float32)
        if self.bichoice:
            # layer_dim, batch_size, hidden_dim
            h0 = torch.zeros(self.layer_dim*2, x.size(0), self.hidden_dim).requires_grad_().to(device)

            # 分离隐藏状态，避免梯度爆炸
            out, hn = self.gru(x, h0.detach())
            out = self.fc(out[:, -1, :])
        else:
            # layer_dim, batch_size, hidden_dim
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

            # 分离隐藏状态，避免梯度爆炸
            out, hn = self.gru(x, h0.detach())
            out = self.fc(out[:, -1, :])
        return out

class simpleNet(nn.Module):
    def __init__(self, input_dim, n_hidden_1, n_hidden_2, output_dim):
        super(simpleNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, n_hidden_1),nn.ReLU())

        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU())

        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, output_dim))

    def forward(self, x):
        x = x.to(torch.float32)
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=max_length, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1)

        self.Linear1 = nn.Linear(128*8,64)
        self.Linear2 = nn.Linear(64, 1)

    def forward(self,x):
        x = x.to(torch.float32)
        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))


        x = x.view(-1, 128*8)
        x = F.relu(self.Linear1(x))
        x = self.Linear2(x)

        return x




    # model = LSTM(8, 100, 1, 1, bichoice)
    # model = GRU(8, 100, 1, 1, bichoice)
    # model = RNN(8, 64, 1, 1, bichoice)
    # model = simpleNet(max_length*8, 100, 32, 1)
    # model = CNN()




