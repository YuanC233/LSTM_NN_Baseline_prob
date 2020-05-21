import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import pandas as pd

class DataHolder(Dataset):
    def __init__(self, feature, label, exog, dim, length, device):
        self.feature = feature
        self.label = label
        self.dim = dim
        self.exog = exog
        self.length = length
        self.device = device

    def __getitem__(self, item):
        feature_tensor = self.feature[item]
        label_tensor = self.label[item]
        len_tensor = self.length[item]
        exog_tensor = self.exog[item]
        return feature_tensor, label_tensor, exog_tensor, len_tensor

    def __len__(self):
        return len(self.feature)

def load_data(filepath):
    f = pd.read_csv(filepath)
    # weather [hour, dayofweek, airtemp, power]
    data = np.asarray(f.iloc[:, [2, 3, 4, 13]])
    hours_sin = np.sin(data[:, 0:1] * np.pi / 12)
    hours_cos = np.cos(data[:, 0:1] * np.pi / 12)
    days_sin = np.sin(data[:, 1:2] * np.pi / 3.5)
    days_cos = np.cos(data[:, 1:2] * np.pi / 3.5)

    # [power, airtemp_f, hours_sin, hours_cos, days_sin, days_cos ]
    data = np.hstack((data[:, 3:4] , data[:, 2:3], hours_sin, hours_cos, days_sin, days_cos))

    # all available features to train or to perform feature engineering
    label_loc = 0

    # train/val data = [power, feature]
    data_feature = data
    data_label = data[:, label_loc: label_loc + 1]

    return data_feature, data_label


def nn_create_inout_sequences(input_data, window, history, forward, feature, label, device):
    #[power, airtemp_f, cloudcover, pressure, hours_sin, hours_cos, days_sin, days_cos]
    all_feature = []
    all_label = []
    all_exog = []
    feature_len = []
    n = input_data.shape[0]
    d = len(feature) + 1
    max_len = window - 1

    for i in range(n - window):
        for j in range(0, forward):
            current_feature = input_data[i : i + history + j, [label]+feature]
            current_len = current_feature.shape[0]
            # print(current_feature.shape)
            feature_len.append(torch.tensor([current_len], dtype=torch.long, device=device))

            # padded_feature = pad_seq(current_feature, current_len, max_len, d)

            if current_len < max_len:
                pad = np.zeros((max_len - current_len, d))
                padded_feature = np.vstack((current_feature, pad))
            else:
                padded_feature = current_feature

            exog_feature = input_data[i + history + j: i + history + j + 1, feature]
            current_label = input_data[i + history + j, label].reshape((1, 1))
            all_feature.append(torch.tensor(padded_feature, dtype=torch.float, device=device))
            all_exog.append(torch.tensor(exog_feature, dtype=torch.float, device=device))
            all_label.append(torch.tensor(current_label.squeeze(), dtype=torch.float, device=device))
    return all_feature, all_label, all_exog, feature_len


def nn_create_val_inout_sequences(input_data, window, history, forward, feature, label, device):
    all_feature = []
    all_label = []
    all_exog = []
    feature_len = []
    n = input_data.shape[0]
    d = len(feature) + 1
    max_len = window - 1

    for i in range(n - window):
        for j in range(0, forward):
            current_feature = input_data[i : i + history + j, [label] + feature]
            current_len = current_feature.shape[0]
            # print(current_feature.shape)
            feature_len.append(torch.tensor([current_len], dtype=torch.long, device=device))

            # padded_feature = pad_seq(current_len, max_len, d)
            # padding in the validation, not here

            exog_feature = input_data[i + history + j: i + history + j + 1, feature]
            current_label = input_data[i + history + j, label].reshape((1, 1))
            all_feature.append(torch.tensor(current_feature, dtype=torch.float, device=device))
            all_exog.append(torch.tensor(exog_feature, dtype=torch.float, device=device))
            all_label.append(torch.tensor(current_label.squeeze(), dtype=torch.float, device=device))
    return all_feature, all_label, all_exog, feature_len


def pad_tensor(tensor, tensor_len, max_tensor_len, device):
    pad = nn.ConstantPad2d((0, 0, 0, max_tensor_len - tensor_len), 0).to(device)
    padded_tensor = pad(tensor)
    return padded_tensor