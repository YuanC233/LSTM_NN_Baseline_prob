import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTM_NN(nn.Module):
    def __init__(self, input_size=3, hidden_layer_size=20, hidden_nn_size=20, output_size=1, exog_size=5, drop_rate=0.5):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size
        self.exog_size = exog_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

        self.nn_hidden = nn.Linear(hidden_layer_size + exog_size, hidden_nn_size)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(drop_rate)
        # two output layers, one for mean prediction and the other for variance prediction
        self.output_mean = nn.Linear(hidden_nn_size, output_size)

        self.output_std = nn.Linear(hidden_nn_size, output_size)

    def init_hidden(self, batch_size, device):
        self.hidden_cell = (torch.zeros(1, batch_size, self.hidden_layer_size, device=device),
                            torch.zeros(1, batch_size, self.hidden_layer_size, device=device))

    def forward(self, input_seq, exog_seq, length, batch_size):
        input_seq = pack_padded_sequence(input_seq.view(batch_size, -1, self.input_size), length.view(batch_size),
                                         batch_first=True, enforce_sorted=False)
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        # self.hidden_cell[0] = pad_packed_sequence(lstm_out)[-1]
        feature_seq = torch.cat((self.dropout(self.hidden_cell[0].view(batch_size, -1)), exog_seq.view(batch_size, self.exog_size)),
                                dim=1)
        feature_seq = self.dropout(self.nn_hidden(feature_seq))

        feature_seq = self.relu(feature_seq)

        mean_pred = self.output_mean(feature_seq)
        std_pred = self.output_std(feature_seq)

        return mean_pred, F.softplus(std_pred)

    @staticmethod
    def loss_function(mean, std, label, device):
        dist = torch.distributions.normal.Normal(mean, std + torch.ones_like(std, device=device) * torch.tensor(1e-6, device=device))
        return - dist.log_prob(label).mean()
