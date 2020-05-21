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
from Tuned_Train_Validate import train_model
from Tuned_Train_Validate import model_validate
from LSTM_NN_prob_model import LSTM_NN
from utils import nn_create_val_inout_sequences


parser = argparse.ArgumentParser()
parser.add_argument('--optimizer', '-o', action='store', default='sgd', help='optimizer')
parser.add_argument('--device', '-d', action='store', default='cuda:0', help='CUDA device index')
parser.add_argument('--reg_opt', '-r', action='store', default='a', help='regularizer_opts')
parser.add_argument('--name', '-n', action='store', default='not_specified', help='logging filename')
parser.add_argument('--select_features', '-f', action='store', default='14567', help='feature engineering parameteres')
parser.add_argument('--debug', action='store_const', const=True, default=False, help='Enable Debug Mode')
args = parser.parse_args()

args.device = args.device if torch.cuda.is_available() and args.device else 'cpu'


today = date.today().strftime("%m%d%Y")
logging.basicConfig(filename=f'lr_tuning_{today}_{args.name}.log.txt', level=logging.INFO,
                             format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logging.info('START')
logging.info(f'Arguments: optimizer : {args.optimizer} '
            f'- cuda_device : {args.device}'
            f'- regularizer : Option_{args.reg_opt}'
            f'- selected_features : {args.select_features}'
             f'- debug_mode : {args.debug}')


# import weather data
file = pd.read_csv('./Austin_weather_11month_excludewintersaving.csv', delimiter=',')
# hour, airtemp_f, cloudcover, pressure, dayofweek
weather = np.asarray(file.iloc[:, 2:7])

# convert hours to sin and cos(pi*h/12)
# ref: https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/
hours_sin = np.sin(weather[:,0:1]*np.pi/12)
hours_cos = np.cos(weather[:,0:1]*np.pi/12)

# convert dayofweek to sin and cos(pi*h/3)
days_sin = np.sin(weather[:,4:5]*np.pi/3.5)
days_cos = np.cos(weather[:,4:5]*np.pi/3.5)

# [hour, airtemp_f, cloudcover, pressure, dayofweek, hours_sin, hours_cos, days_sin, days_cos]
weather = np.hstack((weather, hours_sin, hours_cos, days_sin, days_cos))

# import power data
data = genfromtxt('./ave_hourly_aggregatepower.csv', delimiter=',')
power = np.expand_dims(data, axis=1)

# stack poower and weather data
# [power, hour, airtemp_f, cloudcover, pressure, dayofweek, hours_sin, hours_cos, days_sin, days_cos]
data_all = np.hstack((power, weather))

# data scale
# all_data: # [power, hour, airtemp_f, cloudcover, pressure, dayofweek, hours_sin, hours_cos, days_sin, days_cos]
# features: per input argument
# label: power

# determin test/validation data size
test_data_size = 2400

# all available features to train or to perform feature engineering
feature_loc = [2, 3, 4, 6, 7, 8, 9]
label_loc = 0

# train/val data = [power, feature]
train_data = data_all[:-test_data_size, [label_loc, 2, 3, 4, 6, 7, 8, 9]]
val_data = data_all[-test_data_size:, [label_loc, 2, 3, 4, 6, 7, 8, 9]]
train_label = data_all[:-test_data_size, label_loc: label_loc+1]
validation_label = data_all[-test_data_size:, label_loc: label_loc+1]

# data scaling
scaler = StandardScaler()
scaler_label = StandardScaler()

scaler.fit(train_data)
scaler_label.fit(train_label)

train_data_normalized = scaler.transform(train_data)
val_data_normalized = scaler.transform(val_data)
train_label_mormalized = scaler_label.transform(train_label)
validation_label_mormalized = scaler_label.transform(validation_label)

# main loop
# define model paramters for training
TRAIN_WINDOW = 24
TRAIN_FORWARD = 12
TRAIN_HISTORY = TRAIN_WINDOW - TRAIN_FORWARD
BATCH_SIZE = 128
epoch = 5
epoch_iters = 3

if args.debug:
    epoch = epoch_iters = 1
    train_data_normalized = train_data[:256]
    val_data_normalized = val_data[:128]

logging.info(f'Max epoch is set to {epoch * epoch_iters}')

device = torch.device(args.device)

lrs = [0.01, 0.05, 0.1, 0.2]
optimizers = {'adam': Adam, 'sgd': SGD}
optimizer_type = optimizers[args.optimizer.lower()]

feature_engr = {'14567': [1,4,5,6,7], '167': [1,6,7], '145': [1,4,5], '1234567': [1,2,3,4,5,6,7]}
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


lr = 0.01
regularizer = 0

current_lr = lr

model = LSTM_NN(input_size=num_feature_exog, hidden_layer_size=64, hidden_nn_size=32, output_size=1, exog_size=exog_size)
model.to(device)
for epoch_iter in range(epoch_iters):
    optimizer = optimizer_type(filter(lambda p: p.requires_grad, model.parameters()), current_lr, weight_decay=regularizer)
    final_epoch_loss = train_model(train_data_normalized, BATCH_SIZE, TRAIN_WINDOW, TRAIN_HISTORY, TRAIN_FORWARD, model, optimizer, epoch, device, selected_feature, 0)

    print(f'final_epoch_loss: {final_epoch_loss}, batch_size: {BATCH_SIZE}, train_window:{TRAIN_WINDOW}, '
                                f'train_history{TRAIN_HISTORY}, train_forward: {TRAIN_FORWARD}, optimizer: {optimizer}, epoch: {epoch * (epoch_iter+1)}, lr: {current_lr}, regularizer: {regularizer}')
    logging.info(f'final_epoch_loss: {final_epoch_loss}, batch_size: {BATCH_SIZE}, train_window:{TRAIN_WINDOW}, '
                                f'train_history{TRAIN_HISTORY}, train_forward: {TRAIN_FORWARD}, optimizer: {optimizer}, epoch: {epoch * (epoch_iter+1)}, lr: {current_lr}, regularizer: {regularizer}')
    # model validation
    val_loss, pred_results = model_validate(val_data_normalized, TRAIN_WINDOW, TRAIN_HISTORY, TRAIN_FORWARD, model, device, selected_feature, 0)
    logging.info(f'val_loss, {val_loss}')
    print(f'val_loss, {val_loss}')
    current_lr /= 2

print('DONE')
logging.info('DONE')

val_pred_all = np.asarray(pred_results)
val_pred_mean = val_pred_all[:,0:1]
val_pred_std = val_pred_all[:,1:2]
val_pred_upper = val_pred_mean + val_pred_std
val_pred_lower = val_pred_mean - val_pred_std

actual_pre = scaler_label.inverse_transform(val_pred_mean)
pred_upper = scaler_label.inverse_transform(val_pred_upper)
pred_lower = scaler_label.inverse_transform(val_pred_lower)

val_feature, val_label, val_exog, val_len = nn_create_val_inout_sequences(val_data_normalized,
                                                                          TRAIN_WINDOW, TRAIN_HISTORY,
                                                                          TRAIN_FORWARD, selected_feature, 0, device)
actual_y = scaler_label.inverse_transform(val_label)


# plot
timesteps = range(0, actual_y.shape[0], 1000)
print(timesteps)
for step in timesteps:
    plt.figure()
    plt.plot(actual_pre[step:step+12])
    plt.plot(actual_y[step:step+12])
    plt.plot(pred_upper[step:step + 12])
    plt.plot(pred_lower[step:step + 12])

plt.show()