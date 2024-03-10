import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

class DomainAdversarialNN(nn.Module):
    def __init__(self, num_classes=2):
        super(DomainAdversarialNN, self).__init__()
        self.feature_extractor = models.resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        class_output = self.feature_extractor.fc(features)
        domain_output = self.domain_classifier(features)
        return class_output, domain_output

# Assuming you have source and target datasets prepared as PyTorch Datasets
source_dataset = YourSourceDataset(...)
target_dataset = YourTargetDataset(...)

source_dataloader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
target_dataloader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)

model = DomainAdversarialNN()
class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    for i, ((source_data, _), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):
        # Source data
        source_inputs, source_labels = source_data
        source_domain_labels = torch.zeros(source_inputs.size(0), 1)  # Label source domain as 0
        
        # Target data
        target_inputs, _ = target_data
        target_domain_labels = torch.ones(target_inputs.size(0), 1)  # Label target domain as 1
        
        # Concatenate source and target data
        inputs = torch.cat((source_inputs, target_inputs), dim=0)
        domain_labels = torch.cat((source_domain_labels, target_domain_labels), dim=0)
        class_labels = torch.cat((source_labels, torch.zeros_like(target_domain_labels)), dim=0)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        class_output, domain_output = model(inputs)
        
        # Calculate losses
        class_loss = class_criterion(class_output[:len(source_labels)], source_labels)
        domain_loss = domain_criterion(domain_output, domain_labels)
        total_loss = class_loss + lambda_param * domain_loss  # lambda_param is the trade-off parameter
        
        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()
