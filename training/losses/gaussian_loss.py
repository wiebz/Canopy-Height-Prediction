import torch.nn as nn
import torch

class GaussianNLLLoss(nn.Module):
    """Gaussian Negative Log-Likelihood Loss"""

    def __init__(
        self,
        ignore_value=None,
        pre_calculation_function=None,
    ):
        super().__init__()
        self.ignore_value = ignore_value
        self.pre_calculation_function = pre_calculation_function

    def forward(self, out, target):
        """
        Applies the Gaussian NLL loss.
        :param out: Network output, expected to have two channels [mean, log_variance], shape [batch_size, 2, spatial_dim] or [batch_size, 2, height, width].
        :param target: Target values, shape [batch_size, spatial_dim] or [batch_size, height, width].
        :return: Gaussian NLL loss.
        """
        # Check if `out` has the correct number of channels
        if out.shape[1] != 2:
            raise ValueError(f"Expected `out` to have 2 channels (mean, log_variance), but got {out.shape[1]} channels.")

        # Flatten spatial dimensions if necessary
        if out.ndimension() == 4:
            # Reshape 4D tensor to [batch_size, 2, spatial_dim]
            batch_size, _, height, width = out.shape
            out = out.view(batch_size, 2, -1)  # Flatten height and width
            target = target.view(batch_size, -1)[:, :out.shape[2]]  # Match target's spatial_dim to out's
        elif out.ndimension() == 3:
            # If already [batch_size, 2, spatial_dim], use directly
            batch_size, _, spatial_dim = out.shape
            target = target.view(batch_size, -1)[:, :spatial_dim]
        else:
            raise ValueError(f"Unexpected tensor dimensions for `out`: {out.shape}")

        # Split the output into mean and log_variance
        mean = out[:, 0, :]  # Shape: [batch_size, spatial_dim]
        log_variance = out[:, 1, :]  # Shape: [batch_size, spatial_dim]

        # Apply mask if `ignore_value` is specified
        if self.ignore_value is not None:
            mask = target != self.ignore_value  # Shape: [batch_size, spatial_dim]

            # Check consistency of dimensions
            if mask.shape[-1] != mean.shape[-1]:
                raise ValueError(f"Mask shape {mask.shape} does not match `mean` shape {mean.shape}.")

            # Apply mask
            mean = mean[mask]
            log_variance = log_variance[mask]
            target = target[mask]

        # Calculate the Gaussian NLL loss
        variance = torch.exp(log_variance)
        nll = 0.5 * ((target - mean) ** 2 / variance + log_variance)  # NLL formula
        return nll.mean()
