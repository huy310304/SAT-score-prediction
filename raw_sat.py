# This code is adapted from the Bike Sharing Project by Udacity.
# Source: https://github.com/udacity/deep-learning-v2-pytorch/tree/c9404fc86181fc3f0906b368697268257f348535/project-bikesharing

# SAT score prediction using neural network implemented based on numpy 
# Using GPA_small.csv file with original hyperparematers

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

data_location = 'GPA_Small.csv'

data = pd.read_csv(data_location)
data = data.sample(frac=1)
test_data = data[-200:]
data = data[:-200]

# Countable features
quant_features = ['colgpa', 'SAT', 'hsize', 'hsrank', 'hsperc']
scaled_features = dict()

# Convert data into mean of 0 and std 1
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    # store into a dict for later retrieving
    scaled_features[each] = [mean, std]
    # assign scaling values
    data.loc[:, each] = (data[each] - mean)/std
    test_data.loc[:, each] = (test_data[each] - mean)/std

target_feature = ['SAT']

features = data.drop(target_feature, axis=1) 
targets = data[target_feature]
test_features = test_data.drop(target_feature, axis=1)
test_targets = test_data[target_feature]

train_features, train_targets = features[:-350], targets[:-350]
val_features, val_targets = features[-350:], targets[-350:]

from helper import NeuralNetwork

def MSE(y, Y):
    return np.mean((y-Y)**2)

import sys

from helper import iterations, learning_rate, hidden_nodes, output_nodes


N_i = train_features.shape[1]
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

losses = {'train':[], 'validation':[]}
for ii in range(iterations):
    # Go through a random batch of 128 records from the training data set
    
    batch = np.random.choice(list(range(len(train_features - 1))), size=240)

    X, y = train_features.iloc[batch].values, train_targets.iloc[batch]['SAT']
                             
    network.train(X, y)
    
    # Printing out the training progress
    train_loss = MSE(np.array(network.run(train_features)).T, train_targets['SAT'].values)
    val_loss = MSE(np.array(network.run(val_features)).T, val_targets['SAT'].values)
    sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii/float(iterations)) \
                     + "% ... Training loss: " + str(train_loss)[:5] \
                     + " ... Validation loss: " + str(val_loss)[:5])
    sys.stdout.flush()
    
    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)
plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
_ = plt.ylim()
plt.show()

mean, std = scaled_features['SAT']
SAT_predictions = np.array(network.run(test_features)) * std + mean
SAT_targets = np.array(test_targets) * std + mean

print("_______________________________________________")

score = r2_score(SAT_targets, SAT_predictions)
print(f"The accuracy of our model on the test data (200 entries) is {round(score, 2) *100}%")
mae_score = mean_absolute_error(SAT_targets, SAT_predictions)
print("The Mean Absolute Error of our Model is {}".format(round(mae_score, 2)))