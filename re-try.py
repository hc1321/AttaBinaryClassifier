import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms

# Define your transformations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.Resize((224, 224)),  # Resize the image to fit the model expected size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the dataset
ant_data = datasets.ImageFolder('finalData', transform=transform)

# Define sizes for split
test_size = 0.1
valid_size = 0.2

# Determine data sizes
num_data = len(ant_data)
indices_data = list(range(num_data))
np.random.shuffle(indices_data)
split_tt = int(np.floor(test_size * num_data))
train_idx, test_idx = indices_data[split_tt:], indices_data[:split_tt]

num_train = len(train_idx)
indices_train = list(range(num_train))
np.random.shuffle(indices_train)
split_tv = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices_train[split_tv:], indices_train[:split_tv]

# Define samplers
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
test_sampler = SubsetRandomSampler(test_idx)

# Define data loaders
batch_size = 32
num_workers = 0

train_loader = DataLoader(ant_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
valid_loader = DataLoader(ant_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
test_loader = DataLoader(ant_data, sampler=test_sampler, batch_size=batch_size, num_workers=num_workers)

# Define the classes, in this case, 0 for 'noLeaves' and 1 for 'withLeaves'
classes = [0, 1]


ant_data = datasets.ImageFolder('finalData', transform=transform)
image, label = ant_data[0]  # For example, get the first item
print(f"Label: {label}")  # This will print 0 if the image is from 'noLeaves', 1 if from 'withLeaves'
train_loader = DataLoader(ant_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
valid_loader = DataLoader(ant_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
test_loader = DataLoader(ant_data, sampler=test_sampler, batch_size=batch_size, num_workers=num_workers)

import matplotlib.pyplot as plt
import torchvision

# This function takes in a PyTorch Tensor and displays it
def imshow(img, label):
    img = img.numpy().transpose((1, 2, 0))  # Convert from Tensor image
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    img = std * img + mean  # Unnormalize the image
    img = np.clip(img, 0, 1)  # Clip the image pixel values
    plt.imshow(img)
    plt.title(f"Label: {label}")
    plt.show()

# Get the first batch of images
images, labels = next(iter(train_loader))

# Display the first few images in the batch
for i in range(4,10):  # Let's display 4 images from the batch
    label = labels[i].item()
    class_name = 'noLeaves' if label == 0 else 'withLeaves'
    imshow(images[i], class_name)
