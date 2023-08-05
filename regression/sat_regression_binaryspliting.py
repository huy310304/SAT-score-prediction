# USING BIG SAT CSV FILE WITH MORE VARIATION TO PREVENT OVERFITTING
# USING BINARY SPLITING FOR REGRESSION

import time
import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

# data path
data_location = "GPA_Big.csv"

# read the data csv file
data = pd.read_csv(data_location)
for _ in range(3):
    data = data.sample(frac=1)

# convert hsize and hsrank into binary
def binary_convert(feature):

    hsize_binary_list = list()
    for each in data[feature]:
        b_list = []
        binary = bin(each)[2:]
        # convert into 10 digits binary by adding 0
        while len(binary) < 10:
            binary = "0" + binary 
        
        for b in binary:
            b_list.append(int(b))

        hsize_binary_list.append(b_list)

    binary_array = np.array(hsize_binary_list)
    # return a list a binary representation for each entry
    return binary_array

# add hsize binary into the data
hsize_binary_array = binary_convert("hsize")
hsize_loc = 4
for i in range(10):
    data.insert(hsize_loc+i, "hsize_binary", hsize_binary_array.T[i], allow_duplicates=True)

# add hsrank binary into the data
hsrank_binary_array = binary_convert("hsrank")
hsrank_loc = 15
for i in range(10):
    data.insert(hsrank_loc + i, "hsrank_binary", hsrank_binary_array.T[i], allow_duplicates=True)

# drop the original hsize and hsrank
data = data.drop(['hsize', 'hsrank'], axis=1)

# test_data is the last 800 entries (10%)
test_data = data[-800:]
data = data[:-800]

# Countable features
quant_features = ['colgpa', 'hsperc', 'SAT']
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

# drop the SAT feature, move to targets
features = data.drop(target_feature, axis=1) 
targets = data[target_feature]
test_features = test_data.drop(target_feature, axis=1)
test_targets = test_data[target_feature]

# validation data is the last 800 entries (10%)
train_features, train_targets = features[:-800], targets[:-800]
val_features, val_targets = features[-800:], targets[-800:]

valloader = pd.concat([val_features, val_targets], axis=1)
trainloader = pd.concat([train_features, train_targets], axis=1)
testloader = pd.concat([test_features, test_targets], axis=1)

# convert train data into tensor
trainloader = torch.tensor(trainloader.values, dtype=torch.float64)
trainloader = torch.utils.data.DataLoader(trainloader, batch_size = 128, shuffle = True)

# convert validation data into tensor
valloader = torch.tensor(valloader.values, dtype=torch.float64)
valloader = torch.utils.data.DataLoader(valloader, batch_size = 64, shuffle = True)

# convert test data into tensor
testloader = torch.tensor(testloader.values, dtype=torch.float64)

# define the model for regression problem
model = nn.Sequential(nn.Linear(24, 256),
                       nn.LeakyReLU(),
                       nn.Dropout(p = 0.3),
                       nn.Linear(256, 256),
                       nn.LeakyReLU(),
                       nn.Dropout(p = 0.3),
                       nn.Linear(256, 128),
                       nn.LeakyReLU(),
                       nn.Dropout(p = 0.3),
                       nn.Linear(128, 128),
                       nn.LeakyReLU(),
                       nn.Dropout(p = 0.2),
                       nn.Linear(128, 64),
                       nn.ReLU(),
                       nn.Dropout(p = 0.2),
                       nn.Linear(64, 16),
                       nn.ReLU(),
                       nn.Dropout(p = 0.2),
                       nn.Linear(16, 1))

# Change the dtype of weights and biases to torch.float64 to match the dtype of inputs
new_dtype = torch.float64  

for param in model.parameters():
    param.data = param.data.to(new_dtype)

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.002)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10)

epochs = 200
train_losses = []
val_losses = []
accuracies = []
times = []

for e in range(epochs):

    start_epoch = time.time()

    training_loss = 0
    for row in trainloader:
        features = row[:, :-1]
        targets = row[:, -1].unsqueeze(1)
        output = model(features)
        loss = loss_function(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss += loss.item()

    else: 
        scheduler.step(training_loss)

        accuracy = 0
        MAE = 0
        validation_loss = 0  

        with torch.no_grad():
            
            model.eval()

            for row in valloader:
                features = row[:, :-1]
                targets = row[:, -1].unsqueeze(1)
                output = model(features)
                validation_loss += loss_function(output, targets)
                
                # Calculate the accuracy of the validation set
                mean, std = scaled_features['SAT']
                predictions = output * std + mean
                targets = targets * std + mean
                score = r2_score(targets.detach().numpy(), predictions.detach().numpy())
                accuracy += score
                mae_score = mean_absolute_error(targets.detach().numpy(), predictions.detach().numpy())
                MAE += mae_score


            model.train()

        train_losses.append(training_loss/len(trainloader))
        val_losses.append(validation_loss/len(valloader))
        accuracies.append(accuracy/len(valloader))

    end_epoch = time.time()
    epoch_time = end_epoch - start_epoch
    times.append(epoch_time)

    print(f"Epoch: {e+1}/{epochs}")
    print(f"Training Loss: {training_loss/len(trainloader)}")
    print(f"Validation Loss: {validation_loss/len(valloader)}")
    print(f"Validation Accuracy: {round(accuracy/len(valloader), 2)*100}%")
    print(f"Validation MAE error: {round(MAE/len(valloader), 2)}")
    print("______________________________________________________")

print(f"Highest accuracy for the validation set is {round(max(accuracies), 2)*100}%")

# Plot losses for training and validation + accuracy
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.plot(accuracies, label="Accuracy")
plt.plot(times, label="Time")
plt.xlabel('Epoch')
plt.legend()
_ = plt.ylim()
plt.show()

# Pass in test data for the model to predict
mean, std = scaled_features['SAT']
SAT_predictions = torch.round(model(testloader[:, :-1]) * std + mean)
SAT_targets = testloader[:, -1] * std + mean

# Convert output into numpy array
SAT_targets = SAT_targets.detach().numpy()
SAT_predictions = SAT_predictions.detach().numpy()

# Modify the limit of SAT score
SAT_predictions[SAT_predictions > 1600] = 1599
SAT_predictions[SAT_predictions < 400] = 400

# Using r2 score and mae to calculate accuracy and error
score = r2_score(SAT_targets, SAT_predictions)
print(f"The accuracy of our model on the test data (800 entries) is {round(score, 2) *100}%")
mae_score = mean_absolute_error(SAT_targets, SAT_predictions)
print(f"The Mean Absolute Error of our Model is {round(mae_score, 2)}")


# Plot predicting results
plt.plot(SAT_predictions, label='SAT_prediction')
plt.plot(SAT_targets, label='SAT_targets')
plt.legend()
_ = plt.ylim()
plt.show()