# script to test resnet on independent dataset

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import random
import pandas as pd 
import os  
import pickle

from resnet18 import ResNet, BasicBlock
# evaluate what is needed here
from utils_resnet_esc50 import Dataset
from test_utils_resnet_esc50 import predict

# read file to create list of IDs 
filepath = '/content/drive/MyDrive/soundclass_resnet18_v3/'
filename_csv = 'esc50_edited.csv'
filename_keypairs = 'labels_after_training.pkl'
filename_model = 'trained_model_threechannel_50epochs.pt'
filename_predictions = 'predictions_and_truelabels.csv'

# Create Datasets 'partition' and 'labels'
csv_data = pd.read_csv(os.path.join(filepath,filename_csv), sep = ';')
file_IDs = csv_data['filename'].to_list() # convert to list
file_IDs = [s.strip('.wav') for s in file_IDs] # remove extension from file_IDs
cat_IDs = csv_data['cat_high'].to_list()
labels   = csv_data['cat_high'].to_list() # create labels

# The data split is as follows: 1500 train sounds (300 per category), 400 validation sounds (80 per category), and 100 test sounds (20 per category)
indices_ss = [i for i, elem in enumerate(cat_IDs) if 'Soundscape' in elem] # get indices of each category
indices_do = [i for i, elem in enumerate(cat_IDs) if 'Domestic' in elem]
indices_an = [i for i, elem in enumerate(cat_IDs) if 'Animals' in elem]
indices_hu = [i for i, elem in enumerate(cat_IDs) if 'Human' in elem]
indices_ex = [i for i, elem in enumerate(cat_IDs) if 'Exterior' in elem]
# Create lists of train and test indices
indices_test = indices_ss[380:]+indices_do[380:]+indices_an[380:]+indices_hu[380:]+indices_ex[380:]
# Create data and label splits
labels_test = dict(zip(list(file_IDs[i] for i in indices_test),list(labels[i] for i in indices_test)))

# Read dictionary pkl file
with open(os.path.join(filepath,filename_keypairs), 'rb') as fp:
    keys = pickle.load(fp)
    print(keys)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
params_test = {'batch_size': 100,
              'shuffle': False}

# Define parameters.
nr_channels    = 1 # This specifies the number of channels that is used. Can be set either to 1 or 3.
nr_classes     = 5 # For the ESC50 dataset, the nr of classes is 5. 

# Generators
test_set = Dataset(list(file_IDs[i] for i in indices_test),labels_test)
test_generator = torch.utils.data.DataLoader(test_set,**params_test)

# Initialise model
model = ResNet(img_channels=nr_channels, num_layers=18, block=BasicBlock, num_classes=nr_classes).to(device)
# load state dict
model.load_state_dict(torch.load(os.path.join(filepath,filename_model)))

# main execution block
if __name__ == '__main__':
    # Perform prediction
    predictions, true_labels, keys = predict(
        model, 
        test_generator, 
        device,
        keys
        )

# Calculate accuracy
evaluation_correct = (predictions == true_labels).sum().item()
accuracy_evaluation = evaluation_correct

print('evaluation on independent dataset complete, accuracy = ' + str(accuracy_evaluation) + '%')
#print(predictions)
#print(true_labels)
# check whether key-value pairs match training key-value pairs
#print(keys)

# Save predictions and true labels for subsequent analysis
predictions = predictions.cpu() # bring back to cpu for numpy
true_labels = true_labels.cpu()
df = pd.DataFrame({'predictions':predictions.numpy(),'true_labels':true_labels.numpy()}) #convert to a dataframe
df.to_csv(os.path.join(filepath,filename_predictions), index={'Predictions','TrueLabels'}) #save to file
