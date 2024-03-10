import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve


# Define transform to preprocess the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)

# Create a data loader for labeled and unlabeled data
labeled_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
unlabeled_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

# Define your neural network architecture
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # Define layers...

    def forward(self, x):
        # Define forward pass...

# Initialize the model
model = SimpleNet()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training loop
for epoch in range(num_epochs):
    for labeled_data, unlabeled_data in zip(labeled_loader, unlabeled_loader):
        labeled_images, labels = labeled_data
        unlabeled_images, _ = unlabeled_data
        
        # Forward pass for labeled data
        labeled_outputs = model(labeled_images)
        labeled_loss = criterion(labeled_outputs, labels)
        
        # Forward pass for unlabeled data (pseudo-labeling)
        unlabeled_outputs = model(unlabeled_images)
        pseudo_labels = torch.argmax(unlabeled_outputs, dim=1)  # Get pseudo-labels
        unlabeled_loss = criterion(unlabeled_outputs, pseudo_labels)
        
        # Calculate total loss
        total_loss = labeled_loss + unlabeled_loss
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

# Define transform to preprocess the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)

# Create a data loader for labeled and unlabeled data
labeled_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
unlabeled_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

# Define your neural network architecture
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # Define layers...

    def forward(self, x):
        # Define forward pass...

# Initialize the model
model = SimpleNet()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training loop
for epoch in range(num_epochs):
    for labeled_data, unlabeled_data in zip(labeled_loader, unlabeled_loader):
        labeled_images, labels = labeled_data
        unlabeled_images, _ = unlabeled_data
        
        # Forward pass for labeled data
        labeled_outputs = model(labeled_images)
        labeled_loss = criterion(labeled_outputs, labels)
        
        # Forward pass for unlabeled data (pseudo-labeling)
        unlabeled_outputs = model(unlabeled_images)
        pseudo_labels = torch.argmax(unlabeled_outputs, dim=1)  # Get pseudo-labels
        unlabeled_loss = criterion(unlabeled_outputs, pseudo_labels)
        
        # Calculate total loss
        total_loss = labeled_loss + unlabeled_loss
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

unique_labels = np.unique(y_true)
print("Unique Labels:", unique_labels)

# Metric Evaluation
# Evaluate performance metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Comparison with Baselines
# If applicable, compare with baseline models or traditional supervised learning approaches
# For simplicity, let's assume baseline_accuracy is the accuracy of a traditional supervised learning model
baseline_accuracy = 0.85

if accuracy > baseline_accuracy:
    print("SSL has provided improvements over the baseline.")
else:
    print("SSL did not outperform the baseline.")

# Day 100: Visualization

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks(unique_labels, unique_labels)
plt.yticks(unique_labels, unique_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# ROC Curve
# For binary classification tasks, adjust as needed for multi-class tasks
fpr, tpr, _ = roc_curve(y_true, y_pred)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true, y_pred)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='green', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()