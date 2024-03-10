import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import numpy as np

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32*8*8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 32*8*8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load CIFAR-10 dataset (as an example)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Split the labeled and unlabeled data
labeled_indices = list(range(5000))  # Assuming 5000 labeled samples
unlabeled_indices = list(range(5000, len(train_dataset)))

labeled_sampler = SubsetRandomSampler(labeled_indices)
unlabeled_sampler = SubsetRandomSampler(unlabeled_indices)

# DataLoaders for labeled and unlabeled data
labeled_loader = DataLoader(train_dataset, batch_size=64, sampler=labeled_sampler)
unlabeled_loader = DataLoader(train_dataset, batch_size=64, sampler=unlabeled_sampler)

# Initialize the model, optimizer, and criterion
model = SimpleCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop (using both labeled and unlabeled data)
for epoch in range(num_epochs):
    for labeled_data, unlabeled_data in zip(labeled_loader, unlabeled_loader):
        labeled_images, labels = labeled_data
        unlabeled_images, _ = unlabeled_data
        
        optimizer.zero_grad()
        
        # Forward pass for labeled data
        labeled_outputs = model(labeled_images)
        labeled_loss = criterion(labeled_outputs, labels)
        
        # Forward pass for unlabeled data (pseudo-labeling or other SSL techniques can be applied here)
        unlabeled_outputs = model(unlabeled_images)
        
        # Backpropagation and optimization
        total_loss = labeled_loss  # For simplicity, considering only labeled data loss
        total_loss.backward()
        optimizer.step()
