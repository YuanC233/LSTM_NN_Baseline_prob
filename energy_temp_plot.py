import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy import genfromtxt
import statsmodels.api as sm

def temp_split(dataset):
    temp76 = []
    power76 = []
    temp45 = []
    power45 = []
    temp4576 = []
    power4576 = []
    for i in range(dataset.shape[0]):
        if dataset[i, 0] >= 74:
            temp76.append(dataset[i, 0])
            power76.append(dataset[i, 1])
        elif 50 <= dataset[i, 0] < 74:
            temp4576.append(dataset[i, 0])
            power4576.append(dataset[i, 1])
        else:
            temp45.append(dataset[i, 0])
            power45.append(dataset[i, 1])
    return np.asarray(temp45), np.asarray(power45), np.asarray(temp4576), np.asarray(power4576), np.asarray(temp76), np.asarray(power76)


# import weather data
file = pd.read_csv('./Austin_weather_11month_excludewintersaving.csv', delimiter=',')
# hour, airtemp_f, cloudcover, pressure, dayofweek
weather = np.asarray(file.iloc[:, 2:7])
temp = weather[:, 1:2]

# import power data
data = genfromtxt('./ave_hourly_aggregatepower.csv', delimiter=',')
power = np.expand_dims(data, axis=1)

# combine
temp_power = np.hstack((temp, power))

# determine test/validation data size
test_data_size = 2400

# train/val data = [power, feature]
train = temp_power[:-test_data_size, :]
val = temp_power[-test_data_size:, :]

# split datasets according to outside temp
temp_45, power_45, temp_4576, power_4576, temp_76, power_76 = temp_split(train)

# curve fitting
curve45, [resid45, rank45, sv45, rcond45]  = np.polynomial.polynomial.Polynomial.fit(temp_45, power_45, deg=1, full=True)
curve4576, [resid4576, rank4576, sv4576, rcond4576]  = np.polynomial.polynomial.Polynomial.fit(temp_4576, power_4576, deg=1, full=True)
curve76, [resid76, rank76, sv76, rcond76]  = np.polynomial.polynomial.Polynomial.fit(temp_76, power_76, deg=1, full=True)

coef45 = curve45.convert().coef
coef4576 = curve4576.convert().coef
coef76 = curve76.convert().coef

# stats model

temp_45r = sm.add_constant(temp_45)

model45 = sm.OLS(power_45, temp_45r).fit()
predictions = model45.predict(temp_45r)
print_model = model45.summary()
print(print_model)


# validation
val_temp_45, val_power_45, val_temp_4576, val_power_4576, val_temp_76 , val_power_76 = temp_split(val)

pred45 = curve45(val_temp_45)
pred4576 = curve4576(val_temp_4576)
pred76 = curve76(val_temp_76)

loss = np.sum(np.square(pred45-val_power_45)) + np.sum(np.square(pred4576-val_power_4576)) + np.sum(np.square(pred76-val_power_76))
print(loss/val.shape[0])


plt.figure()
plt.scatter(temp_76, power_76, c='g', s=2)
plt.scatter(temp_45, power_45, c='b', s=2)
plt.scatter(temp_4576, power_4576, c='c', s=2)
plt.plot(temp_45, curve45(temp_45), 'k')
plt.plot(temp_4576, curve4576(temp_4576), 'k')
plt.plot(temp_76, curve76(temp_76), 'k')

plt.xlabel('Temperature (F)')
plt.ylabel('Power (kW)')
plt.title('Temperature vs Aggregate Power Consumption for 13 homes in Austin, TX')
plt.show()

