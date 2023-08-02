#!/usr/bin/python

import hashlib
import os
import re
import sys
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from apievasion_rl.envs.utils.mymodel import RNN, LSTM, GRU, CNN, simpleNet
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

module_path = os.path.split(os.path.abspath(sys.modules[__name__].__file__))[0]

is_support = torch.cuda.is_available()
if is_support:
    device = torch.device('cuda:1')

def LoadAPIDetector(model_type):

    # NN models
    # if model_type == 'rnn' or 'brnn' or 'lstm' or 'blstm' or 'gru' or 'bgru' or 'nn' or 'cnn':
    if model_type == 'rnn':
        model = RNN(8,100,2,1,False)
        model.load_state_dict(torch.load(os.path.join(module_path, 'models', model_type + '.pth')))
    elif model_type == 'brnn':
        model = RNN(8,64,1,1,True)
        model.load_state_dict(torch.load(os.path.join(module_path, 'models', model_type + '.pth')))
    elif model_type == 'lstm':
        model = LSTM(8,100,2,1,False)
        model.load_state_dict(torch.load(os.path.join(module_path, 'models', model_type + '.pth')))
    elif model_type == 'blstm':
        model = LSTM(8,100,1,1,True)
        model.load_state_dict(torch.load(os.path.join(module_path, 'models', model_type + '.pth')))
    elif model_type == 'gru':
        model = GRU(8,100,1,1,False)
        model.load_state_dict(torch.load(os.path.join(module_path, 'models', model_type + '.pth')))
    elif model_type == 'bgru':
        model = GRU(8,100,1,1,True)
        model.load_state_dict(torch.load(os.path.join(module_path, 'models', model_type + '.pth')))
    elif model_type == 'nn':
        model = simpleNet(8*200,100,32,1)
        model.load_state_dict(torch.load(os.path.join(module_path, 'models', model_type + '.pth')))
    elif model_type == 'cnn':
        model = CNN()
        model.load_state_dict(torch.load(os.path.join(module_path, 'models', model_type + '.pth')))
    # ML models
    elif model_type == 'svm' or model_type =='lr' or model_type =='rf':
        with open(os.path.join(module_path, 'models', model_type + '.pkl'), 'rb') as f:
            model = pickle.load(f)
    return model


class APIdetector:
    def __init__(self, model_type):
        self.model = LoadAPIDetector(model_type)
        self.threshold = 0.5
        self.sequence_length = 200
        self.model_type = model_type
        self.dim = 8


    def extract(self, sequence):
        squ = np.zeros((self.sequence_length, self.dim))

        if len(sequence) > self.sequence_length:
            squ = sequence[:self.sequence_length, :self.dim]
        else:
            squ[:sequence.shape[0], :self.dim] = sequence[:, :self.dim]

        return squ

    def predict_sample(self, sequence):
        # NN models
        if self.model_type in ['rnn' , 'brnn' , 'lstm' , 'blstm' , 'gru' , 'bgru' , 'nn' , 'cnn']:
            model = self.model.to(device)
            feature = torch.from_numpy(sequence)
            feature = feature.unsqueeze(0)
            feature = feature.to(device)
            output = torch.sigmoid(model(feature).squeeze(-1))
            prediction = 1 if output > 0.5 else 0

        # ML models
        elif self.model_type in ['svm' , 'lr' , 'rf']:
            sequence = sequence.flatten()
            feature = sequence.reshape(1, -1)
            prediction = self.model.predict(feature)[0]

        return prediction
