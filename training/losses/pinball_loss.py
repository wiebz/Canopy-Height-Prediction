import torch
import torch.nn as nn

class PinballLoss(nn.Module):
    """Pinball loss for quantile regression.
    Allows the model to predict multiple quantiles, 
    penalizing underestimations and overestimations asymmetrically.
    """

    def __init__(self, quantiles=[0.1, 0.5, 0.9], ignore_value=None, pre_calculation_function=None):
        """
        Initializes the PinballLoss class.
        :param quantiles: List of quantiles to be estimated (e.g., [0.1, 0.5, 0.9]).
        :param ignore_value: If provided, ignores these values in the loss computation.
        :param pre_calculation_function: Optional function to preprocess inputs and targets.
        """
        super().__init__()
        #self.quantiles = torch.tensor(quantiles, dtype=torch.float32).view(1, -1, 1)  # Shape (1, Q, 1)
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32).view(1, -1, 1))  
        self.ignore_value = ignore_value
        self.pre_calculation_function = pre_calculation_function

    def forward(self, out, target):
        """
        Computes the pinball loss for quantile regression.
        :param model_outputs: Predicted quantiles from the model. Shape: (B, Q, H, W)
        :param ground_truth: Ground truth tensor. Shape: (B, H, W)
        :return: Pinball loss (scalar)
        """
        # Apply any pre-processing if provided
        if self.pre_calculation_function is not None:
            out, target = self.pre_calculation_function(out, target)

        
        # Extract tensor shapes
        batch_size, num_quantiles, height, width = out.shape  # (B, Q, H, W)
        
        # print(f"Original shapes: out shape: {out.shape}, target shape: {target.shape}")

        # Reshape tensors for loss computation
        out = out.view(batch_size, num_quantiles, -1)         # [B, Q, H*W]
        target = target.view(batch_size, -1)                  # [B, H*W]
        # print(f"Shapes after reshaping: out shape: {out.shape}, target shape: {target.shape}")

        # Step 2: Create index
        target_index = torch.where(target != self.ignore_value)  # Tuple: (batch_idx, pixel_idx)
        # print(f"Shape of batch indices: {target_index[0].shape}")
        # print(f"Shape of pixel indices: {target_index[1].shape}")

        # Step 3: Gather target values
        target_filtered = target[target_index]                  # shape: [N]
        # print(f"target_filtered shape: {target_filtered.shape}")

        # Step 4: Gather out values aligned with quantiles
        # out: [B, Q, H*W] → for each (b, p), get all Qs
        batch_idx, pixel_idx = target_index
        out_filtered = out[batch_idx, :, pixel_idx]             # shape: [N, Q]
        # print(f"out_filtered shape: {out_filtered.shape}")

        # Step 5: Align shapes for loss
        target_filtered = target_filtered.unsqueeze(1)          # [N, 1]
        errors = target_filtered - out_filtered                 # [N, Q]
        # print(f"target_filtered shape: {target_filtered.shape}")
        # print(f"errors shape: {errors.shape}")

        # Step 6: Prepare quantiles
        quantiles_tensor = self.quantiles.squeeze(0).squeeze(-1)[:num_quantiles].unsqueeze(0)  # [1, Q]
        quantiles_tensor = quantiles_tensor.expand(errors.shape[0], -1)            # [N, Q]
        # print(f"quantiles_tensor shape: {quantiles_tensor.shape}")

        # Apply pinball loss formula
        loss = torch.max(quantiles_tensor * errors, (quantiles_tensor - 1) * errors)
        loss_per_pixel = loss.sum(dim=1)  # sum over Q → shape [N]
        # print(f"Shape of pinball loss: {loss.shape}")
        # print(f"Mean pinball loss: {loss.mean().item()}")  # Safer than printing full tensor

        # 🔹 Step 7: Return mean loss
        return loss_per_pixel.mean()


        """ FUNKTIONIERT !!!
        # out: (B, Q, H, W)
        # target: (B, 1, H, W)
        B, Q, H, W = out.shape

        # Step 1: Flatten spatial dims
        out = out.view(B, Q, -1)              # (B, Q, H*W)
        target = target.view(B, -1)           # (B, H*W) — drop channel dim since it's 1
        print(f"Shapes after spatial flattening: out shape: {out.shape}, target shape: {target.shape}")

        # Step 2: Create mask per pixel (B, H*W)
        mask = target != self.ignore_value    # bool mask

        # Step 3: Collapse batch and spatial dims
        out = out.permute(0, 2, 1)            # (B, H*W, Q)
        out = out.reshape(-1, Q)              # (B * H*W, Q)

        target = target.reshape(-1)           # (B * H*W,)
        mask = mask.reshape(-1)               # (B * H*W,)

        print(f"Shapes after reshaping: out shape: {out.shape}, target shape: {target.shape}")
        print(f"Shape of mask: {mask.shape}")

        # Step 4: Apply mask — keeps all Qs for each valid position
        out = out[mask]                       # (N, Q)
        target = target[mask]                 # (N,)


        print(f"Shapes after mask: out shape: {out.shape}, target shape: {target.shape}")

        target = target.unsqueeze(1)  # [N, 1]
        errors = target - out         # [N, Q]

        print(f"Shapes before loss calculation: out shape: {out.shape}, target shape: {target.shape}, errors shape: {errors.shape}")

        quantiles_tensor = self.quantiles.squeeze(0).squeeze(-1)[:out.shape[1]]         # [Q]
        quantiles_tensor = quantiles_tensor.unsqueeze(0).expand(out.shape[0], -1)       # [N, Q]

        loss = torch.max(quantiles_tensor * errors, (quantiles_tensor - 1) * errors)
        print(f"quantiles_tensor shape: {quantiles_tensor.shape}, loss shape: {loss.shape}")
        return loss.mean()
        """