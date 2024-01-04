import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import models

# Define a simple encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.base_encoder = models.resnet18(pretrained=False)
        self.base_encoder.fc = nn.Identity()  # Remove the last classification layer

    def forward(self, x):
        return self.base_encoder(x)

# SimCLR Loss
class SimCLRLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        z_i = nn.functional.normalize(z_i, dim=1)  # Normalize representations
        z_j = nn.functional.normalize(z_j, dim=1)
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature
        mask = torch.eye(len(representations), dtype=bool)  # Mask diagonal elements
        loss = nn.CrossEntropyLoss()(similarity_matrix[~mask], torch.arange(len(representations)))
        return loss

# Example usage
encoder = Encoder()
simclr_loss = SimCLRLoss()

# Assuming train_loader contains data in pairs for contrastive learning
for images, _ in train_loader:
    images = images.to(device)
    images_aug1, images_aug2 = augment(images)  # Augment images for positive pairs

    representations1 = encoder(images_aug1)
    representations2 = encoder(images_aug2)

    loss = simclr_loss(representations1, representations2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
