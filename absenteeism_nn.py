# import libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

"""
Section 1: Preprocessing data
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
data.at[data['Absenteeism time in hours'] <= 10, ['Absenteeism time in hours']] = 0
data.at[data['Absenteeism time in hours'] > 40, ['Absenteeism time in hours']] = 2
data.at[data['Absenteeism time in hours'] > 2, ['Absenteeism time in hours']] = 1

data.to_csv("Dataset/pre_processed_dataset.csv")

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
Section 2: Building the neural networks
"""

input_neurons = n_features  # n_features = 46
# the number of hidden neurons usually lies between the number of input and output units. So we start with their mean
hidden_neurons = 24
output_neurons = 3
learning_rate = 0.5
num_epoch = 5000


# define a simple neural network
class TwoLayerNet(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(TwoLayerNet, self).__init__()
        self.hidden = torch.nn.Linear(n_input, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        h_input = self.hidden(x)
        h_output = F.sigmoid(h_input)
        y_pred = self.out(h_output)

        return y_pred


# define a deep neural network
class DeepNet(torch.nn.Module):
    def __init__(self, n_input, n_hidden_1, n_hidden_2, n_output):
        super(DeepNet, self).__init__()
        self.layer_1 = torch.nn.Linear(n_input, n_hidden_1)
        self.layer_2 = torch.nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = torch.nn.Linear(n_hidden_2, n_output)

    def forward(self, x):
        h_input = self.layer_1(x)
        h_output_l1 = F.sigmoid(h_input)
        h_output_l2 = F.sigmoid(h_output_l1)
        y_pred = self.layer_3(h_output_l2)

        return y_pred


# define a neural network using the customised structure
net = TwoLayerNet(input_neurons, hidden_neurons, output_neurons)

# d_net = DeepNet(input_neurons, hidden_neurons, hidden_neurons, output_neurons)

# define loss function
loss_func = torch.nn.CrossEntropyLoss()

# define optimiser
optimiser = torch.optim.SGD(net.parameters(), lr=learning_rate)

# store all losses for visualisation
all_losses = []

"""
Section 3: Training the network
"""

for epoch in range(num_epoch):
    # Perform forward pass: compute predicted y by passing x to the model.
    Y_pred = net(X)

    # Compute loss
    loss = loss_func(Y_pred, Y)
    all_losses.append(loss.data[0])

    # print progress
    if epoch % 50 == 0:
        # convert three-column predicted Y values to one column for comparison
        _, predicted = torch.max(F.softmax(Y_pred, dim=1), 1)

        # calculate and print accuracy
        total = predicted.size(0)
        correct = predicted.data.numpy() == Y.data.numpy()

        print('Epoch [%d/%d] Loss: %.4f  Accuracy: %.2f %%'
              % (epoch + 1, num_epoch, loss.data[0], 100 * sum(correct) / total))

    net.zero_grad()

    loss.backward()

    optimiser.step()

"""
Plotting confusion matrix for training
"""

confusion = torch.zeros(output_neurons, output_neurons)

Y_pred = net(X)
_, predicted = torch.max(F.softmax(Y_pred, dim=1), 1)

for i in range(train_data.shape[0]):
    actual_class = Y.data[i]
    predicted_class = predicted.data[i]

    confusion[actual_class][predicted_class] += 1

print('')
print('Confusion matrix for training:')
print(confusion)

"""
Section 4: Testing the neural network
"""

X_test = Variable(torch.Tensor(test_input.as_matrix()).float())
Y_test = Variable(torch.Tensor(test_target.as_matrix()).long())

Y_pred_test = net(X_test)

_, predicted_test = torch.max(Y_pred_test, 1)

# calculate accuracy
total_test = predicted_test.size(0)
correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())

print('Testing Accuracy: %.2f %%' % (100 * correct_test / total_test))

"""
Plotting confusion matrix for testing

"""

confusion_test = torch.zeros(output_neurons, output_neurons)

for i in range(test_data.shape[0]):
    actual_class = Y_test.data[i]
    predicted_class = predicted_test.data[i]

    confusion_test[actual_class][predicted_class] += 1

print('')
print('Confusion matrix for testing:')
print(confusion_test)

"""
Section 5: Pruning by badness:

The badness factor for a hidden unit is the sum of back-propagated error
components over all patterns.
"""

i = 0  # loop variable
all_errors = torch.zeros(24)  # storage for cumulative errors of all hidden nodes

while i < train_input.shape[0]:
    pattern = Variable(torch.Tensor(train_input[i:i + 5].as_matrix()).float())  # Batching input into patterns
    pattern_target = Variable(torch.Tensor(train_target[i:i + 5].as_matrix()).long())
    pattern_pred = net(pattern)
    net_loss = loss_func(pattern_pred, pattern_target)  # feeding pattern forward through the network
    # print(net.out.weight)
    temp_tensor = torch.sum(net.out.weight.grad, 0)
    all_errors.add_(temp_tensor.data)  # adding errors at hidden nodes to the previously accumulated error components
    # temp_tensor.data is used because temp_tensor is a Variable, and all_errors is a floatTensor
    i += 5

all_errors = all_errors.abs()
# print(all_errors)
max_error = all_errors.max()
index = (all_errors == max_error).nonzero()
# print(max_error, index)
# Setting all connections out of the 'bad' node to 0, effectively pruning it
net.out.weight.data[:, index] = 0.0
net.out.weight.data[:, index].requires_grad = False  # Making sure that the values at that remain zero
# print(net.out.weight)

"""
Section 6: Testing the neural network after pruning
"""

X_test = Variable(torch.Tensor(test_input.as_matrix()).float())
Y_test = Variable(torch.Tensor(test_target.as_matrix()).long())

Y_pred_test = net(X_test)

_, predicted_test = torch.max(Y_pred_test, 1)

# calculate accuracy
total_test = predicted_test.size(0)
correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())

print('Testing Accuracy: %.2f %%' % (100 * correct_test / total_test))

"""
Plotting confusion matrix for testing

"""

confusion_test = torch.zeros(output_neurons, output_neurons)

for i in range(test_data.shape[0]):
    actual_class = Y_test.data[i]
    predicted_class = predicted_test.data[i]

    confusion_test[actual_class][predicted_class] += 1

print('')
print('Confusion matrix for testing:')
print(confusion_test)

"""
Plotting the error-epoch graph
"""
plt.figure()
plt.plot(all_losses)
plt.show()
