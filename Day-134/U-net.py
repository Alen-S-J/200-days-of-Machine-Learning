import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Define the encoder (contracting path)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Define the decoder (expansive path)
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2)
        )
        
    def forward(self, x):
        # Forward pass through encoder
        x1 = self.encoder(x)
        # Forward pass through decoder
        x2 = self.decoder(x1)
        return x2

# Instantiate the U-Net model
model = UNet()

# Define your loss function (e.g., CrossEntropyLoss for segmentation tasks)
criterion = nn.CrossEntropyLoss()

# Define your optimizer (e.g., Adam)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Load your dataset and dataloaders

# Training loop
for epoch in range(num_epochs):
    for images, masks in train_dataloader:
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Compute the loss
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        
        # Optimize
        optimizer.step()
        
    # Print epoch loss
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Evaluate your model
# Remember to use your test dataset and dataloader
