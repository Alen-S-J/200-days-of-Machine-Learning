import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Creating a simple Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 128),  # Input dimension of 10 (can be adjusted)
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )

    def forward(self, x1, x2):
        output1 = self.fc(x1)
        output2 = self.fc(x2)
        return output1, output2

# Generating synthetic dataset
def generate_data(size=1000, dim=10):
    data = np.random.rand(size, dim)
    labels = np.random.randint(0, 2, size=size)  # Binary labels
    return data, labels

# Contrastive Loss function
def contrastive_loss(output1, output2, labels, margin=1.0):
    euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
    loss_contrastive = torch.mean((1-labels) * torch.pow(euclidean_distance, 2) +
                                  (labels) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    return loss_contrastive

# Training the model
def train_siamese_network(model, data, labels, epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        input1 = torch.tensor(data[:, :5], dtype=torch.float32)  # Splitting data into two parts for pairs
        input2 = torch.tensor(data[:, 5:], dtype=torch.float32)
        output1, output2 = model(input1, input2)
        loss = contrastive_loss(output1, output2, torch.tensor(labels))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Generating synthetic data and training the Siamese Network
data, labels = generate_data()
model = SiameseNetwork()
train_siamese_network(model, data, labels)
