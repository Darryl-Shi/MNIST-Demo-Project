# train_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# Define the ConvNet model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)  # Output: (batch, 32, 24, 24)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)  # Output: (batch, 64, 20, 20)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 4, 128)  # Adjusted input size after pooling
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Convolutional layers with ReLU and MaxPool
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # Output: (batch, 32, 12, 12)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # Output: (batch, 64, 4, 4)
        # Flatten
        x = x.view(-1, 64 * 4 * 4)
        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training settings
batch_size = 64
epochs = 5  # You can increase this number for better accuracy

# MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root='./data', train=True, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = Net()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        # Zero gradients
        optimizer.zero_grad()
        # Forward pass
        output = model(data)
        # Calculate loss
        loss = criterion(output, target)
        # Backward pass
        loss.backward()
        # Update weights
        optimizer.step()
        
        running_loss += loss.item()
        
        if batch_idx % 100 == 99:  # Print every 100 batches
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
    print('Epoch: {} \tAverage Loss: {:.6f}'.format(
        epoch + 1, running_loss / len(train_loader)))

# Save the trained model
torch.save(model.state_dict(), 'mnist_cnn.pth')
print("Model saved to mnist_cnn.pth")
