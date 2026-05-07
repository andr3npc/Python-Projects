import matplotlib.pyplot as plt
import numpy as np
from project_utils import ProjectDataset
import pickle 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Load the data
dataset = pickle.load(open('ocr_insurance_dataset.pkl', 'rb'))

# Create data loaders
batch_size = 32
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the OCR Model with correct specifications
class OCRModel(nn.Module):
    def __init__(self, num_types=5):  # 5 insurance types: home, life, auto, health, other
        super(OCRModel, self).__init__()
        
        # Image processing layers (Sequential module called image_layer)
        # Conv2d layer with 1 input channel, 16 output channels, kernel_size=3, padding=1
        self.image_layer = nn.Sequential(
            # First convolutional layer as specified
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64 -> 32x32
            
            # Additional convolutional layers for better feature extraction
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4
            
            # Flatten the output
            nn.Flatten()
        )
        
        # Calculate the size after convolutions and pooling
        # Input: 1x64x64
        # After 4 maxpool layers (each reduces size by half): 64 -> 32 -> 16 -> 8 -> 4
        # Final feature map size: 128 * 4 * 4 = 2048
        self.image_fc_size = 128 * 4 * 4  # = 2048
        
        # Type embedding layer
        self.type_embedding = nn.Embedding(num_types, 16)
        
        # Combined layers (image features + type embedding)
        combined_size = self.image_fc_size + 16
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(combined_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)  # 2 output classes: primary and secondary ID
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # x contains both image and type data
        # Split the input into image and type components
        if isinstance(x, (list, tuple)):
            x_image = x[0]
            x_type = x[1]
        else:
            # If x is a single tensor, we need to separate it
            # Assuming the first part is image and the rest is type
            x_image = x
            x_type = torch.zeros(x.size(0), dtype=torch.long)
        
        # Process image through the image_layer
        image_features = self.image_layer(x_image)
        
        # Process type through embedding
        if len(x_type.shape) > 1 and x_type.shape[1] > 1:
            # Convert one-hot to class indices
            x_type = torch.argmax(x_type, dim=1)
        type_features = self.type_embedding(x_type)
        
        # Concatenate image and type features
        combined = torch.cat([image_features, type_features], dim=1)
        
        # Pass through fully connected layers
        x_out = F.relu(self.fc1(combined))
        x_out = self.dropout(x_out)
        x_out = F.relu(self.fc2(x_out))
        x_out = self.dropout(x_out)
        x_out = F.relu(self.fc3(x_out))
        x_out = self.dropout(x_out)
        x_out = self.fc4(x_out)
        
        return x_out

# Initialize the model
model = OCRModel(num_types=5)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop for 10 epochs
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(f"Training on: {device}")
print(f"Model architecture:\n{model}\n")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

# Training
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Based on the dataset structure from show_dataset_images
        # batch[0] contains the image and type, batch[1] contains labels
        images = batch[0][0]  # Extract image
        types = batch[0][1]   # Extract type (one-hot encoded)
        labels = batch[1]     # Extract labels
        
        # Move data to device
        images = images.to(device)
        types = types.to(device)
        labels = labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass - pass both image and type as a tuple
        outputs = model((images, types))
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Calculate statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        
        # Print batch progress every 10 batches
        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    # Calculate epoch statistics
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct_predictions / total_predictions
    
    print(f'Epoch [{epoch+1}/{num_epochs}] Summary - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
    print('-' * 60)

print("Training completed!")

# Test the model on a few samples
model.eval()
with torch.no_grad():
    # Get a batch of test data
    test_batch = next(iter(train_loader))
    test_images = test_batch[0][0][:5].to(device)
    test_types = test_batch[0][1][:5].to(device)
    test_labels = test_batch[1][:5]
    
    outputs = model((test_images, test_types))
    _, predicted = torch.max(outputs, 1)
    
    print("\nSample Predictions:")
    print("Actual -> Predicted")
    for i in range(len(test_labels)):
        actual_label = list(dataset.label_mapping.keys())[list(dataset.label_mapping.values()).index(test_labels[i].item())]
        pred_label = list(dataset.label_mapping.keys())[predicted[i].item()]
        print(f"Sample {i+1}: {actual_label} -> {pred_label}")

# Save the model (optional)
torch.save(model.state_dict(), 'ocr_model.pth')
print("\nModel saved as 'ocr_model.pth'")
