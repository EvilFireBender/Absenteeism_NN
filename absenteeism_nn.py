# import libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.autograd import Variable

# Load data
data = pd.read_csv('Dataset/Absenteeism_at_work.csv', sep=';')

# Drop id column (it's not useful)
data.drop(data.columns[0], axis=1, inplace=True)


# One hot encode categorical variable
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
for a in data[:, -1:]:
    print(data[a])

#print(data.iloc[0:10, 0:10])

# Shuffle data
data = data.sample(frac=1).reset_index(drop=True)



# Split into training set and testing set