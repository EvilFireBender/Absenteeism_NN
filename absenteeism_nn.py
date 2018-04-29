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
data.at[data['Absenteeism time in hours'] <= 10, ['Absenteeism time in hours']] = 0
data.at[data['Absenteeism time in hours'] > 40, ['Absenteeism time in hours']] = 2
data.at[data['Absenteeism time in hours'] > 2, ['Absenteeism time in hours']] = 1


"""
for a in data.iloc[:, 46:47]:
    print(data[a])
print(data.iloc[0:10, 0:10])
"""

# shuffle data
data = data.sample(frac=1).reset_index(drop=True)
# pd.set_option('display.max_rows', None)
# randomly split data into training set (80%) and testing set (20%)
msk = np.random.rand(len(data)) < 0.8
train_data = data[msk]
test_data = data[~msk]
#print(train_data.iloc[ :, 46:47])

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
# define the number of neurons for input layer, hidden layer and output layer
# define learning rate and number of epochs on training
input_neurons = n_features # n_features = 46
# the number of hidden neurons usually lies between the number of input and output units. So we start with their mean
hidden_neurons = 20
output_neurons = 3
learning_rate = 0.5
num_epoch = 5000


# define a customised neural network structure
class TwoLayerNet(torch.nn.Module):

    def __init__(self, n_input, n_hidden, n_output):
        super(TwoLayerNet, self).__init__()
        # define linear hidden layer output
        self.hidden = torch.nn.Linear(n_input, n_hidden)
        # define linear output layer output
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        """
            In the forward function we define the process of performing
            forward pass, that is to accept a Variable of input
            data, x, and return a Variable of output data, y_pred.
        """
        # get hidden layer input
        h_input = self.hidden(x)
        # define activation function for hidden layer
        h_output = F.sigmoid(h_input)
        # get output layer output
        y_pred = self.out(h_output)

        return y_pred

# define a neural network using the customised structure
net = TwoLayerNet(input_neurons, hidden_neurons, output_neurons)

# define loss function
loss_func = torch.nn.CrossEntropyLoss()

# define optimiser
optimiser = torch.optim.SGD(net.parameters(), lr=learning_rate)


# store all losses for visualisation
all_losses = []

# train a neural network
for epoch in range(num_epoch):
    # Perform forward pass: compute predicted y by passing x to the model.
    Y_pred = net(X)

    # Compute loss
    loss = loss_func(Y_pred, Y)
    all_losses.append(loss.data[0])

    # print progress
    if epoch % 50 == 0:
        # convert three-column predicted Y values to one column for comparison
        _, predicted = torch.max(F.softmax(Y_pred), 1)

        # calculate and print accuracy
        total = predicted.size(0)
        correct = predicted.data.numpy() == Y.data.numpy()

        print('Epoch [%d/%d] Loss: %.4f  Accuracy: %.2f %%'
              % (epoch + 1, num_epoch, loss.data[0], 100 * sum(correct)/total))

    # Clear the gradients before running the backward pass.
    net.zero_grad()

    # Perform backward pass
    loss.backward()

    # Calling the step function on an Optimiser makes an update to its
    # parameters
    optimiser.step()


# Optional: plotting historical loss from ``all_losses`` during network learning
# Please comment from next line to ``plt.show()`` if you don't want to plot loss

import matplotlib.pyplot as plt

plt.figure()
plt.plot(all_losses)
plt.show()



confusion = torch.zeros(output_neurons, output_neurons)

Y_pred = net(X)
_, predicted = torch.max(F.softmax(Y_pred), 1)

for i in range(train_data.shape[0]):
    actual_class = Y.data[i]
    predicted_class = predicted.data[i]

    confusion[actual_class][predicted_class] += 1

print('')
print('Confusion matrix for training:')
print(confusion)

"""
Step 3: Test the neural network

Pass testing data to the built neural network and get its performance
"""

# create Tensors to hold inputs and outputs, and wrap them in Variables,
# as Torch only trains neural network on Variables
X_test = Variable(torch.Tensor(test_input.as_matrix()).float())
Y_test = Variable(torch.Tensor(test_target.as_matrix()).long())

# test the neural network using testing data
# It is actually performing a forward pass computation of predicted y
# by passing x to the model.
# Here, Y_pred_test contains three columns, where the index of the
# max column indicates the class of the instance
Y_pred_test = net(X_test)

# get prediction
# convert three-column predicted Y values to one column for comparison
_, predicted_test = torch.max(Y_pred_test, 1)

# calculate accuracy
total_test = predicted_test.size(0)
correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())

print('Testing Accuracy: %.2f %%' % (100 * correct_test / total_test))

"""
Evaluating the Results

To see how well the network performs on different categories, we will
create a confusion matrix, indicating for every iris flower (rows)
which class the network guesses (columns). 

"""

confusion_test = torch.zeros(output_neurons, output_neurons)

for i in range(test_data.shape[0]):
    actual_class = Y_test.data[i]
    predicted_class = predicted_test.data[i]

    confusion_test[actual_class][predicted_class] += 1

print('')
print('Confusion matrix for testing:')
print(confusion_test)