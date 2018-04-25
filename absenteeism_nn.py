# import libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.autograd import Variable

# Load data
data = pd.read_csv('Dataset/Absenteeism_at_work.csv', sep=';')
print(type(data))

# Drop id column (it's not useful)
data.drop(data.columns[0], axis=1, inplace=True)
print(data.iloc[:, 0:4])

# One hot encode categorical variable
one_hot = pd.get_dummies(data['Reason for absence'])
data = data.drop('Reason for absence', axis=1)
data = data.join(one_hot)

# Join the one-hot-encoded columns back in the location of the original column or else
# the next bit will normalise the target column

# normalise input data
for column in data:
    # the last column is target
    if column != data.shape[1] - 1:
        data[column] = data.loc[:, [column]].apply(lambda x: (x - x.mean()) / x.std())
    # or divide by x.max() - x.min()

print(data.iloc[:, 15:])
# Shuffle data


# Split into training set and testing set