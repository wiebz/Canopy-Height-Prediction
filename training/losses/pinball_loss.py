import torch.nn as nn
import torch


class PinballLoss(nn.Module):
    """Pinball loss for quantile regression."""

    def __init__(
        self,
        taus = [0.1, 0.5, 0.9],
        # quantiles=[0.1, 0.5, 0.9],  # Default quantiles
        ignore_value=None,
        pre_calculation_function=None,
    ):
        """
        Initializes the PinballLoss class.
        :param quantiles: List of quantiles to calculate loss for
        :param ignore_value: Value to ignore in the target
        :param pre_calculation_function: Optional function to preprocess inputs
        """
        super().__init__()
        self.taus = taus
        # self.quantiles = quantiles
        self.ignore_value = ignore_value
        self.pre_calculation_function = pre_calculation_function

    def forward(self, out, target):
        """
        Applies the Pinball loss.
        :param out: output of the network (expected shape: [batch_size, len(quantiles), ...])
        :param target: target values (expected shape: [batch_size, ...])
        :return: Pinball loss
        """
        if self.pre_calculation_function is not None:
            out, target = self.pre_calculation_function(out, target)

        """
        # Ensure shapes are consistent
        batch_size = target.shape[0]
        num_quantiles = len(self.quantiles)

        assert out.shape[0] == batch_size, "Output batch size must match target batch size"
        assert out.shape[1] == num_quantiles, "Output must have a channel for each quantile"
        """

        # Flatten the tensors
        """
        out = out.flatten()
        target = target.flatten()
        """
        """
        out = out.view(batch_size, num_quantiles, -1)  # [batch_size, quantiles, flattened]
        target = target.view(batch_size, -1)  # [batch_size, flattened]

        if self.ignore_value is not None:
            # Mask out ignored values
            mask = target != self.ignore_value
            print(f"mask shape: {mask.shape}")
            print(f"out shape: {out.shape}")
            print(f"target shape: {target.shape}")
            out = out[:, :, mask]
            target = target[mask]
        """
            
        # Reshape tensors to [batch_size, num_quantiles, height * width]
        batch_size, num_quantiles, height, width = out.shape
        out = out.view(batch_size, num_quantiles, -1) # [batch_size, num_quantiles, spatial_dim]
        target = target.view(batch_size, -1)  # [batch_size, spatial_dim]

        if self.ignore_value is not None:
            # Create mask for valid entries
            mask = target != self.ignore_value # [batch_size, spatial_dim]
            print(f"mask shape: {mask.shape}")
            print(f"out shape: {out.shape}")
            print(f"target shape: {target.shape}")

            # Expand the mask to match the dimensions of `out`
            mask = mask.unsqueeze(1).expand(-1, num_quantiles, -1)  # [batch_size, num_quantiles, spatial_dim]

            # Apply mask to `out` and `target`
            out = out[mask].view(-1, num_quantiles)  # Filter `out` and reshape
            target = target[mask[:, 0, :]]  # Filter `target` accordingly
            """
            # Apply mask to filter valid entries
            out = out[:, :, mask]  # Apply mask to quantiles
            target = target[:, mask]  # Apply mask to target
            """

        # Initialize loss for each quantile
        loss = torch.zeros(num_quantiles, device=out.device)

        for i, tau in enumerate(self.taus):
            diff = target - out[:, i]  # [filtered_elements]
            loss[i] = torch.mean(torch.max(tau * diff, (tau - 1) * diff))

        return loss.mean()

        """
        # Compute Pinball loss for each quantile
        #loss = 0.0
        for i, q in enumerate(self.quantiles):
            error = target - out[:, i, :]  # Difference for the i-th quantile
            quantile_loss = torch.maximum((q - 1) * error, q * error)
            loss += quantile_loss.mean()

        # Average the loss across all quantiles
        return loss / len(self.quantiles)
        """