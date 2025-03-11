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
        # 🔹 Step 1: Apply any pre-processing if provided
        if self.pre_calculation_function is not None:
            out, target = self.pre_calculation_function(out, target)

        # 🔹 Step 2: Extract tensor shapes
        batch_size, num_quantiles, height, width = out.shape  # (B, Q, H, W)
        target = target.view(batch_size, 1, height, width)  # Expand to match shape (B, 1, H, W)
        #target = target.view(batch_size, 1, -1).expand(-1, num_quantiles, -1)  # (B, Q, H*W)

        # 🔹 Step 3: Reshape tensors for loss computation
        out = out.view(batch_size, num_quantiles, -1)  # (B, Q, H*W)
        target = target.view(batch_size, 1, -1).expand(-1, num_quantiles, -1)  # Expand target to (B, Q, H*W)
        # (B, 1, H*W)

        # 🔹 Step 4: Handle ignored values if needed
        if self.ignore_value is not None:
            mask = target != self.ignore_value  # Mask valid values (B, Q, H*W)
            out = out[mask]  # Filter model outputs
            target = target[mask]  # Filter ground truth accordingly

        # # 🔹 Step 4: Handle ignored values if specified
        # if self.ignore_value is not None:
        #     mask = target != self.ignore_value  # Mask for valid values (B, 1, H*W)
        #     # Expand the mask across quantiles (B, Q, H*W)
        #     mask_expanded = mask.expand(-1, num_quantiles, -1)
            
        #     # Apply the mask to both tensors
        #     out = out[mask_expanded].view(-1, num_quantiles)  # Reshape after masking
        #     target = target[mask].view(-1, 1)  # Reshape after masking

            #out = out[mask.expand(-1, num_quantiles, -1)]  # Filter model outputs
            #target = target[mask]  # Filter ground truth accordingly

        # 🔹 Step 5: Compute quantile regression loss
        errors = target - out  # Compute errors (B, Q, H*W)

        quantiles_tensor = self.quantiles.to(out.device)  # ✅ Ensure `quantiles_tensor` is on the same device

        # Expand self.quantiles dynamically based on actual `num_quantiles`
        last_dim = errors.shape[-1]
        quantiles_tensor = self.quantiles[:, :num_quantiles, :].expand(batch_size, -1, last_dim)

        #quantiles_tensor = self.quantiles[:, :num_quantiles, :].expand(batch_size, -1, errors.shape[2])  # Ensure correct shape (B, Q, H*W)
        
        # Apply pinball loss formula
        loss = torch.max(quantiles_tensor * errors, (quantiles_tensor - 1) * errors)

        # 🔹 Step 6: Compute mean loss across all valid pixels
        return loss.mean()
