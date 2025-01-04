import torch
from losses.l1_loss import L1Loss
from losses.pinball_loss import PinballLoss
from losses.l2_loss import L2Loss 


# Test für PinballLoss
pinball_loss = PinballLoss()
out = torch.rand(4, 3, 100)  # [batch_size=4, quantiles=3, spatial_dim=100]
target = torch.rand(4, 100)  # [batch_size=4, spatial_dim=100]
loss = pinball_loss(out, target)
print("Pinball Loss:", loss.item())

# Test für L1Loss
l1_loss = L1Loss()
out = torch.rand(4, 100)  # [batch_size=4, spatial_dim=100]
target = torch.rand(4, 100)  # [batch_size=4, spatial_dim=100]
loss = l1_loss(out, target)
print("L1 Loss:", loss.item())

# Test für L2Loss
l2_loss = L2Loss()
out_l2 = torch.rand(4, 100)  # [batch_size=4, spatial_dim=100]
target_l2 = torch.rand(4, 100)  # [batch_size=4, spatial_dim=100]
loss_l2 = l2_loss(out_l2, target_l2)
print("L2 Loss:", loss_l2.item())