# import libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.autograd import Variable

"""
Section One: Preprocessing data
"""

# load data
data = pd.read_csv('Dataset/Absenteeism_at_work.csv', sep=';')

# drop id column (it's not useful)
data.drop(data.columns[0], axis=1, inplace=True)

# one hot encode categorical variable
one_hot = pd.get_dummies(data['Reason for absence'])
data = data.drop('Reason for absence', axis=1)
data = one_hot.join(data)

# normalise input data
for column in data:
    # the last column is target
    if column != 'Absenteeism time in hours':
        data[column] = data.loc[:, [column]].apply(lambda x: (x - x.mean()) / x.std())
    # or divide by x.max() - x.min()

# Divide the output data into three categories
data.at[data['Absenteeism time in hours'] <= 20, ['Absenteeism time in hours']] = 0
data.at[data['Absenteeism time in hours'] > 80, ['Absenteeism time in hours']] = 2
data.at[data['Absenteeism time in hours'] > 2, ['Absenteeism time in hours']] = 1


"""
for a in data.iloc[:, 46:47]:
    print(data[a])
print(data.iloc[0:10, 0:10])
"""

# shuffle data
data = data.sample(frac=1).reset_index(drop=True)

# randomly split data into training set (80%) and testing set (20%)
msk = np.random.rand(len(data)) < 0.8
train_data = data[msk]
test_data = data[~msk]


# split training data into input and target
n_features = train_data.shape[1] - 1
train_input = train_data.iloc[:, :n_features]
train_target = train_data.iloc[:, n_features]

# split test data into input and target
test_input = test_data.iloc[:, :n_features]
test_target = test_data.iloc[:, n_features]

# create Tensors and wrap in Variables
X = Variable(torch.Tensor(train_input.as_matrix()).float())
Y = Variable(torch.Tensor(train_target.as_matrix()).long())

"""
Section Two: Building the neural network
"""

