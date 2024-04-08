#--------------------------------------------------------------------------

#-------------------------- Library Imports -------------------------------

#--------------------------------------------------------------------------

 

import imageio

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from torchvision.io import read_image

from torchvision.transforms import ToPILImage, Compose, Resize, ToTensor, Normalize

import numpy as np

import time

import os

import glob

import shutil

import random

import matplotlib.pyplot as plt

from matplotlib import colors

from numpy.random import default_rng

from pathlib import Path

 

from google.colab import drive

drive.mount('/content/drive')

 

#--------------------------------------------------------------------------

#----------------------- Training Resnet Model ----------------------------

#--------------------------------------------------------------------------

 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Device: {0}'.format(device))

 

from torch.utils.data import Dataset

from torchvision.io import read_image

from torchvision.transforms import Compose, ToPILImage, Resize, ToTensor, Normalize

from pathlib import Path

 

class AntDataset(Dataset):

    def __init__(self, directory, transform=None):

        self.directory = Path(directory)

        self.transform = Compose([

            ToPILImage(),  # Add this line to convert tensors to PIL Images

            Resize((224, 224)), 

            ToTensor(),

            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ]) if transform is None else transform

        self.images = list(self.directory.glob('**/*.png'))

        self.labels = [0 if 'noLeaves' in str(image.parents[0]) else 1 for image in self.images]

 

    def __len__(self):

        return len(self.images)

 

    def __getitem__(self, idx):

        image_path = self.images[idx]

        image = read_image(str(image_path)).float()

        label = self.labels[idx]

        if self.transform:

            image = self.transform(image)

        return image, label

 

# Create dataset objects

train_dataset = AntDataset('/content/drive/My Drive/BinaryClass/BinaryClassifierNN/finalData/train')

val_dataset = AntDataset('/content/drive/My Drive/BinaryClass/BinaryClassifierNN/finalData/val')

test_dataset = AntDataset('/content/drive/My Drive/BinaryClass/BinaryClassifierNN/finalData/test')

 

# Create dataloaders

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

 

from torchvision.models import resnet18

 

# Initialize a ResNet model

model = resnet18(pretrained=False)  # Set pretrained=False to not use ImageNet weights

 

# Change the final fully connected layer for binary classification

num_ftrs = model.fc.in_features

model.fc = torch.nn.Linear(num_ftrs, 2)  # Outputs for two classes

model.to(device)

 

# Assuming you've set up the device, model, dataloaders, criterion, optimizer

# Optimizer

optimizer = optim.Adam(model.parameters(), lr=1e-3)

 

# Segmentation loss

criterion = nn.CrossEntropyLoss()

 

num_epochs = 1000

save_interval = 500

for epoch in range(num_epochs):

    model.train()  # Set model to training mode

    running_loss = 0.0  # Variable to store the total loss for the epoch

 

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

 

        running_loss += loss.item() * inputs.size(0)  # Multiply by batch size to get total loss

 

    epoch_loss = running_loss / len(train_loader.dataset)  # Average loss for the epoch

 

    print(f'Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}')

 

model_save_path = '/content/drive/My Drive/AntImages/Models/'

if (epoch + 1) % save_interval == 0:

        checkpoint_path = os.path.join(model_save_path, f'model_epoch_{epoch+1}.pth')

        torch.save({

            'epoch': epoch,

            'model_state_dict': model.state_dict(),

            'optimizer_state_dict': optimizer.state_dict(),

            'loss': epoch_loss,

        }, checkpoint_path)

        print(f'Model saved to {checkpoint_path}')

    # Validation loop can go here