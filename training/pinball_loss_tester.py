import torch
from losses.pinball_loss import PinballLoss  # Replace with the actual path to your PinballLoss implementation

# Step 1: Mock Data Preparation
batch_size = 5
height, width = 64, 64  # Example dimensions of an output map
taus = [0.1, 0.5, 0.9]  # Quantile levels

# Simulated model output: 3 quantiles (0.1, 0.5, 0.9) per pixel
"""
predicted = torch.tensor(
    [[[0.1, 0.5, 0.9], [0.2, 0.6, 1.0]],  # Quantiles for the first sample
     [[0.3, 0.7, 1.1], [0.4, 0.8, 1.2]]],  # Quantiles for the second sample
    dtype=torch.float32
).reshape(batch_size, 3, height, width)  # [batch_size, 3 (quantiles), height, width]
"""

predicted = torch.rand(batch_size, len(taus), height, width, dtype=torch.float32)


# Ground truth values for each pixel
"""
ground_truth = torch.tensor(
    [[[0.15, 0.55], [0.25, 0.65]],  # Ground truth for the first sample
     [[0.35, 0.75], [0.45, 0.85]]],  # Ground truth for the second sample
    dtype=torch.float32
).reshape(batch_size, 1, height, width)  # [batch_size, 1, height, width]
"""

ground_truth = torch.rand(batch_size, 1, height, width, dtype=torch.float32)

# Step 2: Initialize the Pinball Loss
# taus = [0.1, 0.5, 0.9]  # Quantile levels
pinball_loss_fn = PinballLoss(taus)


# Step 3: Compute the Loss
loss = pinball_loss_fn(predicted, ground_truth)
print(f"Pinball Loss: {loss.item()}")

# Step 4: Validate Behavior (Optional)
# Test how the loss changes when predictions deviate more from the ground truth
predicted_offset = predicted + 0.1
loss_offset = pinball_loss_fn(predicted_offset, ground_truth)
print(f"Pinball Loss (offset predictions): {loss_offset.item()}")
