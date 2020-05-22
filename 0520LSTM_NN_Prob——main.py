import torch
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import date
from numpy import genfromtxt
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam, SGD
from Train_Validate import train_model
from Train_Validate import model_validate
from LSTM_NN_prob_model import LSTM_NN
from utils import load_data


parser = argparse.ArgumentParser()
parser.add_argument('--optimizer', '-o', action='store', default='sgd', help='optimizer')
parser.add_argument('--device', '-d', action='store', default='cuda:0', help='CUDA device index')
parser.add_argument('--reg_opt', '-r', action='store', default='a', help='regularizer_opts')
parser.add_argument('--name', '-n', action='store', default='not_specified', help='logging filename')
parser.add_argument('--select_features', '-f', action='store', default='12345', help='feature engineering parameteres')
parser.add_argument('--debug', action='store_const', const=True, default=False, help='Enable Debug Mode')
args = parser.parse_args()

args.device = args.device if torch.cuda.is_available() and args.device else 'cpu'


today = date.today().strftime("%m%d%Y")
logging.basicConfig(filename=f'./log/lr_tuning_{today}_{args.name}.log', level=logging.INFO,
                             format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logging.info('START')
logging.info(f'Arguments: optimizer : {args.optimizer} '
            f'- cuda_device : {args.device}'
            f'- regularizer : Option_{args.reg_opt}'
            f'- selected_features : {args.select_features}'
             f'- debug_mode : {args.debug}')

# load data
train_data, train_label = load_data('train_data.csv')
val_data, validation_label = load_data('validation_data.csv')

# data scaling
scaler = StandardScaler()
scaler_label = StandardScaler()

scaler.fit(train_data)
scaler_label.fit(train_label)

train_data_normalized = scaler.transform(train_data)
val_data_normalized = scaler.transform(val_data)
train_label_mormalized = scaler_label.transform(train_label)
validation_label_mormalized = scaler_label.transform(validation_label)

########################################################################################################################
# main loop                                                                                                            #
########################################################################################################################

# define model paramters for training
TRAIN_WINDOW = 24
TRAIN_FORWARD = 12
TRAIN_HISTORY = TRAIN_WINDOW - TRAIN_FORWARD
BATCH_SIZE = 128
epoch = 5
epoch_iters = 10

if args.debug:
    epoch = epoch_iters = 1
    train_data_normalized = train_data_normalized[:256]
    val_data_normalized = val_data_normalized[:128]

logging.info(f'Max epoch is set to {epoch * epoch_iters}')

device = torch.device(args.device)

lrs = [0.01, 0.05, 0.1, 0.2]
optimizers = {'adam': Adam, 'sgd': SGD}
optimizer_type = optimizers[args.optimizer.lower()]

feature_engr = {'12345': [1,2,3,4,5], '167': [1,2,3], '145': [1,4,5]} #TODO remove feature engr
selected_feature = feature_engr[args.select_features]
num_feature_exog = len(selected_feature) + 1
exog_size = len(selected_feature)

regularizer_opts = {'a':[0.00, 0.001], 'b':[0.005, 0.01]}
regularizers = regularizer_opts[args.reg_opt.lower()]

hidden_layer_sizes = [48, 64, 80, 96]
hidden_nn_size = [16, 32, 48]


tuning_epoch_losses = []
val_losses = []

logging.info("START TRAINING")
logging.info("*******************************************")
logging.info("Parameter Settings: ")
logging.info(f"LRs = {lrs}")
#logging.info(f"Epochs = {epochs}")
logging.info(f"Optimizers = {optimizers}")
logging.info(f"Hidden_layer_sizes = {hidden_layer_sizes}")
logging.info(f"Hidden_nn_size = {hidden_nn_size}")
logging.info(f"Regularizers = {regularizers}")
logging.info("*******************************************")

# lr tuning
for lr in lrs:
    for regularizer in regularizers:
        #for i in range(3):
        # model training
        current_lr = lr

        model = LSTM_NN(input_size=num_feature_exog, hidden_layer_size=96, hidden_nn_size=48, output_size=1, exog_size=exog_size)
        model.to(device)
        for epoch_iter in range(epoch_iters):
            optimizer = optimizer_type(filter(lambda p: p.requires_grad, model.parameters()), current_lr, weight_decay=regularizer)
            final_epoch_loss = train_model(train_data_normalized, BATCH_SIZE, TRAIN_WINDOW, TRAIN_HISTORY, TRAIN_FORWARD, model, optimizer, epoch, device, selected_feature, 0)

            print(f'final_epoch_loss: {final_epoch_loss}, batch_size: {BATCH_SIZE}, train_window:{TRAIN_WINDOW}, '
                                        f'train_history{TRAIN_HISTORY}, train_forward: {TRAIN_FORWARD}, optimizer: {optimizer}, epoch: {epoch * (epoch_iter+1)}, lr: {current_lr}, regularizer: {regularizer}')
            logging.info(f'final_epoch_loss: {final_epoch_loss}, batch_size: {BATCH_SIZE}, train_window:{TRAIN_WINDOW}, '
                                        f'train_history{TRAIN_HISTORY}, train_forward: {TRAIN_FORWARD}, optimizer: {optimizer}, epoch: {epoch * (epoch_iter+1)}, lr: {current_lr}, regularizer: {regularizer}')
            # model validation
            val_loss, val_pred = model_validate(val_data_normalized, TRAIN_WINDOW, TRAIN_HISTORY, TRAIN_FORWARD, model, device, selected_feature, 0)

            logging.info(f'val_loss, {val_loss}')
            print(f'val_loss, {val_loss}')
            current_lr /= 2

            result_transformed = [[(scaler_label.inverse_transform(mean), scaler_label.inverse_transform(label))
                                   for mean, sd, label in batch] for batch in val_pred]
            mse_transformed = np.asarray([[np.square(mean - label).squeeze().tolist() for mean, label in batch]
                                          for batch in result_transformed])

            mse_transformed_hourly = [sum(mse_transformed[:, i]) / len(result_transformed) for i in range(TRAIN_FORWARD)]
            logging.info(f'mse_transformed_hourly, {mse_transformed_hourly}')

print('DONE')
logging.info('DONE')