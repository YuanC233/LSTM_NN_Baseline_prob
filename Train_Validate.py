from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import nn_create_inout_sequences
from utils import nn_create_val_inout_sequences
from utils import DataHolder
from utils import pad_tensor


def train_model(train_data_normalized, batch_size, train_window, train_history, train_forward, model, optimizer, epochs, cuda, feature, label):
    model.train()
    #print(model.nn_hidden.weight)
    train_feature, train_label, train_exog, train_len = nn_create_inout_sequences(train_data_normalized, train_window,
                                                                                  train_history, train_forward,
                                                                                  feature, label, cuda)
    batch_train_data = DataHolder(train_feature, train_label, train_exog, 1, train_len, cuda)
    train_batches = DataLoader(batch_train_data, batch_size=batch_size, shuffle=True, drop_last=True)

    loss_function = model.loss_function
    model_optimizer = optimizer

    for i in range(epochs):
        epoch_loss = 0
        current_batch = tqdm(train_batches)
        for idx, batch in enumerate(current_batch):
            feature_tensor, label_tensor, exog_tensor, len_tensor = batch
            model.init_hidden(batch_size, cuda)

            model_optimizer.zero_grad()

            mean_pred, std_pred = model(feature_tensor, exog_tensor, len_tensor, batch_size)
            single_loss = loss_function(mean_pred.squeeze(), std_pred.squeeze(), label_tensor, cuda)
            single_loss.backward()
            model_optimizer.step()
            epoch_loss += single_loss.item()
            current_batch.set_description(f'epoch: {i:3} loss: {epoch_loss / (idx + 1):10.8f} ')
    #print(model.nn_hidden.weight)

    return epoch_loss / (idx + 1)


def model_validate(val_data_normalized, train_window, train_history, train_forward, model, cuda, feature, label):
    # model Validation
    model.eval()
    val_feature, val_label, val_exog, val_len = nn_create_val_inout_sequences(val_data_normalized,
                                                                              train_window, train_history,
                                                                              train_forward, feature, label, cuda)
    val_set = DataHolder(val_feature, val_label, val_exog, 1, val_len, cuda)
    val_batch = DataLoader(val_set, batch_size=1, shuffle=False, drop_last=True)
    max_tensor_len = train_window - 1
    val_single_loss = 0

    val_pred_all = []
    current_batch = tqdm(val_batch)

    for idx, batch in enumerate(current_batch):
        feature_tensor, label_tensor, exog_tensor, len_tensor = batch
        model.init_hidden(1, cuda)

        if idx % 12 == 0:
            if idx != 0:
                val_pred_all.append(val_pred_batch)
            val_pred_batch = []
            temp_tensor = feature_tensor
            temp_padded_tensor = pad_tensor(temp_tensor, len_tensor.squeeze().item(), max_tensor_len, cuda)
        else:
            temp_padded_tensor = pad_tensor(temp_tensor, len_tensor.squeeze().item(), max_tensor_len, cuda)

        with torch.no_grad():

            val_mean_pred, val_std_pred = model(temp_padded_tensor, exog_tensor, len_tensor, 1)
            val_pred_batch.append((val_mean_pred.clone().detach().cpu().numpy(),
                                 val_std_pred.clone().detach().cpu().numpy(),
                                label_tensor.clone().detach().cpu().numpy()))

            loss_function = nn.MSELoss()
            single_loss = loss_function(val_mean_pred.squeeze(), label_tensor.squeeze())

            val_single_loss += single_loss.item()
            #val_pred_all.append((val_mean_pred.squeeze().item(), val_std_pred.squeeze().item()))

            #print(val_pred.unsqueeze(0).size(), exog_tensor.size())
            temp_feature = torch.cat((val_mean_pred.unsqueeze(0), exog_tensor), dim=2)

            #print(temp_feature.size())
            #print(temp_tensor.size())
            temp_tensor = torch.cat((temp_tensor, temp_feature), dim=1)

    return val_single_loss, val_pred_all
