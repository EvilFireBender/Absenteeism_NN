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

# one_hot2 = pd.get_dummies(data['Month of absence'])
# data = data.drop('Month of absence', axis=1)
# data = one_hot2.join(data)

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
Section 2: Building the neural network
"""

input_neurons = n_features  # n_features = 46
# the number of hidden neurons usually lies between the number of input and output units. So we start with their mean
hidden_neurons = 24
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

hidden_layer_biases = torch.zeros([24, 1])

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
              % (epoch + 1, num_epoch, loss.data[0], 100 * sum(correct) / total))

    net.zero_grad()

    loss.backward()

    optimiser.step()

    # for child in net.children():
    #     print(child)
    #     for name, param in child.named_parameters():
    #         if name == 'bias':
    #             print(param)
    #     break

    for child in net.children():
        for name, param in child.named_parameters():
            if name == 'bias':
                if epoch == 0:
                    hidden_layer_biases = param
                else:
                    hidden_layer_biases = hidden_layer_biases + param
        break


print(hidden_layer_biases)
hidden_layer_biases = hidden_layer_biases.abs()


# Plotting the error-epoch graph
plt.figure()
plt.plot(all_losses)
plt.show()

# for child in net.children():
#     print(child)
#     for name, param in child.named_parameters():
#         print(name)
#         print(param)
#         if name == 'bias':
#             hidden_layer_biases = param
#     break
#
# print(hidden_layer_biases)
# hidden_layer_biases = hidden_layer_biases + param
# print(hidden_layer_biases)

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
Section 3: Testing the neural network
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
Section 4: Evaluating the Results

"""

confusion_test = torch.zeros(output_neurons, output_neurons)

for i in range(test_data.shape[0]):
    actual_class = Y_test.data[i]
    predicted_class = predicted_test.data[i]

    confusion_test[actual_class][predicted_class] += 1

print('')
print('Confusion matrix for testing:')
print(confusion_test)
