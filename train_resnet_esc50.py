import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import random
import wandb
import pandas as pd 
import os  
import pickle

from resnet18 import ResNet, BasicBlock
# evaluate what is needed here
from utils_resnet_esc50 import Dataset
from training_utils_resnet_esc50_v2 import train, validate

# read file to create list of IDs 
filepath = '/content/drive/MyDrive/soundclass_resnet18_v3/'
filename = 'esc50_edited.csv'

# Create Datasets 'partition' and 'labels'
csv_data = pd.read_csv(os.path.join(filepath,filename), sep = ';')
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
indices_train = indices_ss[:300]+indices_do[:300]+indices_an[:300]+indices_hu[:300]+indices_ex[:300]
indices_val = indices_ss[300:380]+indices_do[300:380]+indices_an[300:380]+indices_hu[300:380]+indices_ex[300:380]
random.shuffle(indices_train) # shuffle order
random.shuffle(indices_val) # shuffle order
# Create data and label splits
labels_train = dict(zip(list(file_IDs[i] for i in indices_train),list(labels[i] for i in indices_train)))
labels_val = dict(zip(list(file_IDs[i] for i in indices_val),list(labels[i] for i in indices_val)))
partition = {'train': list(file_IDs[i] for i in indices_train), 'validation': list(file_IDs[i] for i in indices_val)} 

print(indices_train)
print(indices_val)
print(labels_train)
print(labels_val)

# Set parameters for WandB logging -- Not yet, try to make this work only later
wandb.init(project="PROJECTNAME") # set the wandb project where this run will be logged
wandb.run.name = 'RUN_NAME' # Include useful information, e.g. 'NAME_BatchSize_XX_LR_XX_Epochs_XX_Gamma_XX' 

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Define learning and training parameters.
batsize        = 32 # define batch size
learning_rate  = 0.0002 # define learning rate
nr_channels    = 1 # This specifies the number of channels that is used. Can be set either to 1 or 3.
nr_classes     = 5 # For the ESC50 dataset, the nr of classes is 5. 
max_epochs     = 50

# track hyperparameters and run metadata
wandb.config={
    "learning_rate": learning_rate,
    "batch_size": batsize,
    "classes": nr_classes,
    "architecture": "ResNet-18",
    "dataset": "ESC-50_MelSpect",
    "epochs": max_epochs,
    }

# Define parameters
params_train = {'batch_size': batsize,
          'shuffle': True}
params_test = {'batch_size': batsize,
          'shuffle': True}

# Data generators
training_set = Dataset(partition['train'],labels_train)
training_generator = torch.utils.data.DataLoader(training_set, **params_train)

validation_set = Dataset(partition['validation'],labels_val)
validation_generator = torch.utils.data.DataLoader(validation_set, **params_test)

# Model
model = ResNet(img_channels=nr_channels, num_layers=18, block=BasicBlock, num_classes=nr_classes).to(device)
# Optimizer.
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [40,60], gamma=.75) # 60 is added in case a higher number of epochs is used
# Loss function.
criterion = nn.CrossEntropyLoss()

# main execution block
if __name__ == '__main__':
    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    # Start the training.
    for epoch in range(max_epochs):
        print(f"[INFO]: Epoch {epoch+1} of {max_epochs}")
        train_epoch_loss, train_epoch_acc, keys_train = train(
            model, 
            training_generator, 
            optimizer, 
            scheduler,
            criterion,
            device
        )
        valid_epoch_loss, valid_epoch_acc, keys_valid = validate(
            model, 
            validation_generator, 
            criterion,
            device
        )
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-'*50)

        wandb.log({
          "Epoch": epoch,
          "Train Loss": train_epoch_loss,
          "Train Acc": train_epoch_acc,
          "Valid Loss": valid_epoch_loss,
          "Valid Acc": valid_epoch_acc})

    print('TRAINING COMPLETE')

    # save key-value dictionary for evaluation on independent dataset
    with open(os.path.join(filepath,'labels_after_training.pkl'), 'wb') as fp:
        pickle.dump(keys_train, fp)
        print('dictionary saved successfully to file')

# save model weights after training
torch.save(model.state_dict(), os.path.join(filepath,'trained_model_onechannel_50epochs.pt'))
