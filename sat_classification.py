# SAT score prediction project using classification techniques that split SAT score into 12 brackets with range 100

import time
import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

# data path
data_location = 'GPA_Big.csv'

# read the data csv file
data = pd.read_csv(data_location)
data = data.sample(frac=1)

# Categorize into 12 brackets of SAT score with range 100 each
data.loc[:, "SAT"] = (data["SAT"] - 400) // 100


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
    return binary_array

hsize_binary_array = binary_convert("hsize")
hsize_loc = 4
for i in range(10):
    data.insert(hsize_loc+i, "hsize_binary", hsize_binary_array.T[i], allow_duplicates=True)

hsrank_binary_array = binary_convert("hsrank")
hsrank_loc = 15
for i in range(10):
    data.insert(hsrank_loc + i, "hsrank_binary", hsrank_binary_array.T[i], allow_duplicates=True)

data = data.drop(['hsize', 'hsrank'], axis=1)

# test_data is the last 400 entries
test_data = data[-800:]
data = data[:-800]

# Countable features
quant_features = ['colgpa', 'hsperc']
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

# validation data is the last 400 entries
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

model = nn.Sequential(nn.Linear(24, 256),
                       nn.ReLU(),
                       nn.Dropout(p = 0.2),
                       nn.Linear(256, 256),
                       nn.ReLU(),
                       nn.Dropout(p = 0.2),
                       nn.Linear(256, 128),
                       nn.ReLU(),
                       nn.Dropout(p = 0.2),
                       nn.Linear(128, 128),
                       nn.ReLU(),
                       nn.Dropout(p = 0.2),
                       nn.Linear(128, 64),
                       nn.ReLU(),
                       nn.Dropout(p = 0.2),
                       nn.Linear(64, 16),
                       nn.ReLU(),
                       nn.Dropout(p = 0.2),
                       nn.Linear(16, 12),
                       nn.LogSoftmax(dim=1))

# Change the dtype of weights and biases to torch.float64 to match the dtype of inputs
new_dtype = torch.float64  

for param in model.parameters():
    param.data = param.data.to(new_dtype)

loss_fn = nn.NLLLoss()
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
        targets = targets.view(targets.shape[0])
        targets = targets.type(torch.LongTensor)
        
        log_ps = model(features)
        loss = loss_fn(log_ps, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss += loss.item()
    else: 
        validation_loss = 0
        accuracy = 0

        scheduler.step(training_loss)
        
        with torch.no_grad():
            
            model.eval()

            for row in valloader:
                features = row[:, :-1]
                
                targets = row[:, -1].unsqueeze(1)
                targets = targets.view(targets.shape[0])
                targets = targets.type(torch.LongTensor)
                
                log_ps = model(features)
                validation_loss += loss_fn(log_ps, targets)

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim = 1)
                equals = top_class == targets.view(top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

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
    print(f"Validation Accuracy: {accuracy/len(valloader)}")

plt.plot(times, label = "Time")
plt.plot(train_losses, label = "Training loss")
plt.plot(val_losses, label = "Validation loss")
plt.plot(accuracies, label = "Accuracy")
plt.legend()
plt.show()

SAT_predictions = torch.exp(model(testloader[:, :-1]))
SAT_targets = testloader[:, -1]
SAT_targets = SAT_targets.type(torch.LongTensor)
top_p, top_class = SAT_predictions.topk(1, dim = 1)
equals = top_class == SAT_targets.view(top_class.shape)
test_accuracy = torch.mean(equals.type(torch.FloatTensor))
print(f"Test_accuracy for 800 test entries is: {test_accuracy}")

# Plot predicting results
plt.plot(top_class.detach().numpy(), label='SAT_prediction')
plt.plot(SAT_targets.detach().numpy(), label='SAT_targets')
plt.legend()
plt.show()