import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from sklearn.metrics import f1_score

# Import Intel Extension for PyTorch for hardware optimization
import intel_extension_for_pytorch as ipex

# Set hyperparameters and dataset path
LR = 0.001
DOWNLOAD = True
DATA = "datasets/cifar10/"

# Define image transformations: resize, normalize, and convert to tensor
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Load CIFAR-10 training dataset with transformations
train_dataset = torchvision.datasets.CIFAR10(
    root=DATA,
    train=True,
    transform=transform,
    download=DOWNLOAD,
)

# Create DataLoader for training data
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128)

# Define a custom model
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # Define layers for the custom CNN model
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)  # Assuming image size 224x224 after transformation
        self.fc2 = nn.Linear(512, 10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        # Define forward pass
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten the tensor before passing to fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the custom model
model = CustomCNN()

# Define loss function (CrossEntropyLoss for multi-class classification)
criterion = torch.nn.CrossEntropyLoss()

# Define optimizer (Stochastic Gradient Descent with momentum)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

# Set the model to training mode
model.train()

# Move model and criterion to XPU for hardware acceleration
model = model.to("xpu")
criterion = criterion.to("xpu")

# Optimize the model and optimizer for better performance with Intel hardware
model, optimizer = ipex.optimize(model, optimizer=optimizer)

# Initialize lists to store predictions and targets for final metrics calculation
all_preds = []
all_targets = []

# Train the model and print loss and accuracy during training
for batch_idx, (data, target) in enumerate(train_loader):
    # Move data and target to XPU
    data = data.to("xpu")
    target = target.to("xpu")

    # Zero gradients, perform forward pass, compute loss, and update weights
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    # Calculate accuracy
    _, predicted = torch.max(output, 1)
    correct = (predicted == target).sum().item()
    accuracy = 100 * correct / target.size(0)

    # Append predictions and targets for F1 score calculation
    all_preds.extend(predicted.cpu().numpy())
    all_targets.extend(target.cpu().numpy())

    # Print loss and accuracy every 100 batches
    if batch_idx % 100 == 0:
        print(f"Batch {batch_idx}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")

# Calculate final accuracy and F1 score on all batches
final_accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_targets)) / len(all_targets)
final_f1 = f1_score(all_targets, all_preds, average='weighted')  # Use weighted average for multi-class

# Save the model and optimizer state
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    "checkpoint.pth",
)

# Print final accuracy and F1 score
print(f"Final Accuracy: {final_accuracy:.2f}%")
print(f"Final F1 Score: {final_f1:.4f}")

