import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import glob
from model import UNet

# 1. CUSTOM DATASET LOADER
class UHIPatchDataset(Dataset):
    def __init__(self, patch_dir):
        self.patch_files = glob.glob(os.path.join(patch_dir, "*.npy"))

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        # Load the 5-layer patch [Landsat, Sentinel, NDVI, DEM, Mask]
        patch = np.load(self.patch_files[idx])
        
        # Split into Input and Target
        # Input: Sentinel (1), NDVI (2), DEM (3), Mask (4)
        # Target: Landsat LST (0)
        x = patch[1:5, :, :].copy()
        y = patch[0:1, :, :].copy()
        
        # Replace NaNs with 0 to prevent training errors
        x = np.nan_to_num(x)
        y = np.nan_to_num(y)

        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

# 2. TRAINING SETUP
PATCH_DIR = r"D:\UHI_Project\data_processed\patches"
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001

def train_model():
    # Setup Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Load Data
    dataset = UHIPatchDataset(PATCH_DIR)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize Model, Loss, and Optimizer
    model = UNet(in_channels=4, out_channels=1).to(device) # 4 input layers, 1 output
    criterion = nn.MSELoss() # Measures how far off the AI's guess is
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. THE TRAINING LOOP
    print("Starting Training...")
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            
            # Use the Mask (layer 4) to only calculate loss on CLEAR pixels
            # This is the 'Partial Convolution' logic
            mask = inputs[:, 3:4, :, :]
            loss = criterion(outputs * mask, targets * mask)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    # 4. SAVE THE BRAIN
    torch.save(model.state_dict(), "uhi_unet_model.pth")
    print("\n Training Complete! Model saved as 'uhi_unet_model.pth'")

if __name__ == "__main__":
    train_model()
