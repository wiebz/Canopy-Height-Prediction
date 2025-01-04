import torch.nn as nn
import torch


class L2Loss(nn.Module):
    """Mean Squared error"""

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
        Applies the L2 loss with robust masking and dimensional checks.
        :param out: Network output, expected shape [batch_size, num_quantiles, height, width] or [batch_size, num_quantiles, spatial_dim].
        :param target: Target values, expected shape [batch_size, height, width] or [batch_size, spatial_dim].
        :return: L2 loss.
        """
        # Flatten spatial dimensions if necessary
        if out.ndimension() == 4:
            # Reshape 4D tensor to [batch_size, num_quantiles, spatial_dim]
            batch_size, num_quantiles, height, width = out.shape
            out = out.view(batch_size, num_quantiles, -1)  # Flatten height and width
            target = target.view(batch_size, -1)[:, :out.shape[2]]  # Match target's spatial_dim to out's
            # print(f"L2 4D Shape: out: {out.shape}, target: {target.shape}")
        elif out.ndimension() == 3:
            # If already [batch_size, num_quantiles, spatial_dim], use directly
            batch_size, num_quantiles, spatial_dim = out.shape
            target = target.view(batch_size, -1)[:, :spatial_dim]
            # print(f"L2 3D Shape: out: {out.shape}, target: {target.shape}")
        else:
            raise ValueError(f"L2 Unexpected tensor dimensions for `out`: {out.shape}")

        # Apply mask if `ignore_value` is specified
        if self.ignore_value is not None:
            # Create a mask for valid target entries
            mask = target != self.ignore_value  # Shape: [batch_size, spatial_dim]

            # Check consistency of dimensions
            if mask.shape[-1] != out.shape[-1]:
                raise ValueError(f"L2 Mask shape {mask.shape} does not match `out` shape {out.shape}.")

            # Expand the mask to match the dimensions of `out`
            mask = mask.unsqueeze(1).expand(-1, num_quantiles, -1)  # Shape: [batch_size, num_quantiles, spatial_dim]

            # Debugging output
            # print(f"L2 Mask shape: {mask.shape}, Out shape: {out.shape}, Target shape: {target.shape}")

            # Apply the mask to `out` and `target`
            out = out[mask].view(-1, num_quantiles)  # Filter `out` and reshape
            target = target[mask[:, 0, :]]  # Filter `target` accordingly

        # Compute the L2 loss
        loss = (target - out.mean(dim=1)) ** 2  # Mean squared difference
        return loss.mean()


    def forward_old(self, out, target):
        """
        Applies the L2 loss
        :param out: output of the network
        :param target: target
        :return L2 loss
        """
        if self.pre_calculation_function != None:
            out, target = self.pre_calculation_function(out, target)

        # out = out.flatten()
        # target = target.flatten()

        # Flatten spatial dimensions for both tensors
        out = out.view(out.size(0), -1)  # [batch_size, spatial_dim]
        target = target.view(target.size(0), -1)

        """
        if self.ignore_value is not None:
            out = out[target != self.ignore_value]
            target = target[target != self.ignore_value]
        """

        if self.ignore_value is not None:
            # Create a mask to exclude values matching `ignore_value`
            mask = target != self.ignore_value
            out = out[mask]
            target = target[mask] 

        loss = (target - out) ** 2

        return loss.mean()
