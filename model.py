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
iterations = 10000
m = len(x_train_scaled)
learning_rate = 1
w = np.zeros(15)
b = 0 

for i in range(iterations):
 g = np.dot(x, w) + b
 predictions = 1 / (1 + np.exp(-g))
 cost = -1/m * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))

 error = predictions - y
 w = w - (learning_rate/m) * np.dot(x.T, error)
 b = b - (learning_rate/m) * np.sum(error)

 if (i % 100 == 0 ):
  print(f'At iteration {i} cost is: {cost}')

print(f'Final Weights: {w}, final bias {b}')

# Seeing result on test data

y_2 = y_test.values
x_2 = x_test_scaled.values

predictions_test = 1 / (1 + np.exp(-(np.dot(x_2, w) + b)))
predictions_tf = predictions_test >= 0.152

correct_hits = np.array((predictions_tf == y_2)).astype(int)
num_acc = correct_hits.sum()
accuracy = (num_acc/len(y_2)) * 100
print(f"Accuracy of Model : {accuracy:.2f}% ")
print(num_acc)

# Seeing the recall
c = 0
for i in range(len(y_2)):
 if (y_2[i] == 1) & (np.array(predictions_tf[i]).astype(int) == 1):
  c=c+1

recall = c/np.sum(y_2)
print(f"Recall: {recall * 100:.2f}%")


"""

Accuracy of Model : 67.57% 
Recall: 67.67%

Final Weights: 

male: 0.4142
age: 0.5631
education: -0.0088
currentSmoker: -0.1454
cigsPerDay: 0.3011
BPMeds: 0.2740
prevalentStroke: 0.7070
prevalentHyp: 0.3653
diabetes: -0.0959
totChol: 0.0701
sysBP: 0.2041
diaBP: -0.0210
BMI: -0.0078
heartRate: 0.0033
glucose: 0.2011

Final Bias: -2.19957

"""