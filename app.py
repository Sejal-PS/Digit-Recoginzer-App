import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np

print("PyTorch Version:", torch.__version__)

# --- 1. Load the MNIST Dataset ---
# The MNIST dataset is directly available in torchvision and will be downloaded automatically.
# This dataset contains 60,000 training images and 10,000 test images of handwritten digits.
# Each image is 28x28 grayscale.
print("\nLoading MNIST dataset...")

# Download and load training data
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transforms.ToTensor(), download=True
)

# Download and load test data
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transforms.ToTensor(), download=True
)

print(f"Training data shape: {train_dataset.data.shape}") # (60000, 28, 28)
print(f"Testing data shape: {test_dataset.data.shape}")   # (10000, 28, 28)
print(f"Example label: {train_dataset.targets[0]}")

# --- 2. Preprocess the Data ---
# Create data loaders for batch processing
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- 3. Build the Convolutional Neural Network (CNN) Model ---
print("\nBuilding CNN model...")

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 7x7 after two pooling operations
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNN()
print(model)

# --- 4. Compile the Model ---
print("\nCompiling model...")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# --- 5. Train the Model ---
print("\nTraining model... (This may take a few minutes)")
epochs = 15

for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(train_loader, 0):
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if i % 100 == 99:
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}, acc: {100 * correct / total:.2f}%')
            running_loss = 0.0
    
    scheduler.step()
    print(f'Epoch {epoch + 1} completed: Loss: {running_loss / len(train_loader):.3f}, Accuracy: {100 * correct / total:.2f}%, LR: {scheduler.get_last_lr()[0]:.6f}')

# --- 6. Evaluate the Model ---
print("\nEvaluating model...")
correct = 0
total = 0
test_loss = 0.0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test loss: {test_loss / len(test_loader):.4f}')
print(f'Test accuracy: {100 * correct / total:.2f}%')

# --- 7. Save the Trained Model ---
# This step is crucial for deploying the model with Streamlit or other applications.
model_save_path = "digit_recognition_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"\nModel saved to {model_save_path}")
print("You can now download this file and use it in your Streamlit application.")
