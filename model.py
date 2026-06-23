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


