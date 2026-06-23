import numpy as np
import pandas as pd

df = pd.read_csv("framingham.csv")
np.random.seed(67)
shuffled_indices = np.random.permutation(df.index)
df_shuffled = df.iloc[shuffled_indices]

split_index = int(len(df_shuffled)*0.8)
df_train = df_shuffled.iloc[:split_index]
df_test = df_shuffled.iloc[split_index:]

x_train = df_train.drop("TenYearCHD", axis=1)
y_train = df_train["TenYearCHD"]
x_test = df_test.drop("TenYearCHD", axis=1)
y_test = df_test["TenYearCHD"]

train_medians = x_train.median()
x_train = x_train.fillna(train_medians)
x_test = x_test.fillna(train_medians)

scaling_columns = ['age', 'education', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']

x_train_scaled = x_train.copy()
x_test_scaled = x_test.copy()

train_stds = x_train[scaling_columns].std()
train_means = x_train[scaling_columns].mean()

x_train_scaled[scaling_columns] = ( x_train[scaling_columns] - train_means ) / train_stds

x_test_scaled[scaling_columns] = (x_test[scaling_columns]-train_means) / train_stds

# End of data pre-processing

x = x_train_scaled.values
y = y_train.values
iterations = 1000
m = len(x_train_scaled)
learning_rate = 0.01
w = np.zeros(15)
b = 0 

for i in range(iterations):
 g = np.dot(x_train_scaled, w) + b
 predictions = 1 / (1 + np.exp(-g))
 cost = -1/m * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))