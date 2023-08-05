# ORIGINAL SAT PROJECT WITH ORIGINAL CHOICE OF MODEL, EPOCH, BATCH_SIZE, ...
# USING GPA_Small.csv FILE

import time
import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

# data path
data_location = 'GPA_Small.csv'

# read the data csv file
data = pd.read_csv(data_location)
data = data.sample(frac=1)

# test_data is the last 200 entries
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

# drop the SAT feature, move to targets
features = data.drop(target_feature, axis=1) 
targets = data[target_feature]
test_features = test_data.drop(target_feature, axis=1)
test_targets = test_data[target_feature]

# validation data is the last 350 entries
train_features, train_targets = features[:-350], targets[:-350]
val_features, val_targets = features[-350:], targets[-350:]

valloader = pd.concat([val_features, val_targets], axis=1)
trainloader = pd.concat([train_features, train_targets], axis=1)
testloader = pd.concat([test_features, test_targets], axis=1)

# convert train data into tensor
trainloader = torch.tensor(trainloader.values, dtype=torch.float64)
trainloader = torch.utils.data.DataLoader(trainloader, batch_size = 64, shuffle = True)

# convert validation data into tensor
valloader = torch.tensor(valloader.values, dtype=torch.float64)
valloader = torch.utils.data.DataLoader(valloader, batch_size = 64, shuffle = True)

# convert test data into tensor
testloader = torch.tensor(testloader.values, dtype=torch.float64)

model = nn.Sequential(nn.Linear(6, 256),
                       nn.ReLU(),
                       nn.Dropout(p = 0.3),
                       nn.Linear(256, 256),
                       nn.ReLU(),
                       nn.Dropout(p = 0.3),
                       nn.Linear(256, 128),
                       nn.ReLU(),
                       nn.Dropout(p = 0.3),
                       nn.Linear(128, 128),
                       nn.ReLU(),
                       nn.Dropout(p = 0.3),
                       nn.Linear(128, 64),
                       nn.ReLU(),
                       nn.Dropout(p = 0.2),
                       nn.Linear(64, 64),
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
optimizer = optim.SGD(model.parameters(), lr = 0.02)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=10)

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
        accuracies.append(round(accuracy/len(valloader), 2))

    end_epoch = time.time()
    epoch_time = end_epoch - start_epoch 
    times.append(epoch_time)

    print(f"Epoch: {e+1}/{epochs}")
    print(f"Training Loss: {training_loss/len(trainloader)}")
    print(f"Validation Loss: {validation_loss/len(valloader)}")
    print(f"The accuracy of the model is {round(accuracy/len(valloader), 2)*100}%")
    print(f"The Mean Absolute Error of our Model is {round(MAE/len(valloader), 2)}")
    print("______________________________________________________")

# Plot losses for training and validation + accuracy
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.plot(accuracies, label="Accuracy")
plt.plot(times, label="Time")
plt.xlabel('Epoch')
plt.legend()
# _ = plt.ylim()
plt.show()

# Pass in test data for the model to predict
mean, std = scaled_features['SAT']
SAT_predictions = model(testloader[:, :-1]) * std + mean
SAT_targets = testloader[:, -1] * std + mean

score = r2_score(SAT_targets.detach().numpy(), SAT_predictions.detach().numpy())
print(f"The accuracy of our model on the test data (200 entries) is {round(score, 2) *100}%")
mae_score = mean_absolute_error(SAT_targets.detach().numpy(), SAT_predictions.detach().numpy())
print("The Mean Absolute Error of our Model is {}".format(round(mae_score, 2)))

# Plot predicting results
plt.plot(SAT_predictions.detach().numpy(), label='SAT_prediction')
plt.plot(SAT_targets.detach().numpy(), label='SAT_targets')
plt.legend()
_ = plt.ylim()
plt.show()

# Demo of predictions
inputs = input("colgpa, athlete, hsize, hsrank, hsperc, female: ").split(",")
inputs = list(float(i) for i in inputs)
d = dict()
for key, value in zip(["colgpa", "athlete", "hsize", "hsrank", "hsperc", "female"], inputs):
    d[key] = value
for key in ["colgpa", "hsize", "hsrank", "hsperc"]:
    mean, std = scaled_features[key]
    d[key] = (d[key]-mean) / std

inputs = torch.tensor(list(d.values()), dtype=torch.float64)
print(inputs)
SAT_predictions = 0
for _ in range(10):
    with torch.no_grad():
        output = model(inputs)
        mean, std = scaled_features["SAT"]
        score = torch.round(output * std + mean)
        print(score)
        SAT_predictions += score

print(SAT_predictions.detach().numpy() / 10)