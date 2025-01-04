import torch.nn as nn
import torch


class L1Loss(nn.Module):
    """Mean Absolute error"""

    def __init__(
        self,
        ignore_value=None,
        pre_calculation_function=None,
        lower_threshold=None,
    ):
        super().__init__()
        self.ignore_value = ignore_value
        self.pre_calculation_function = pre_calculation_function
        self.lower_threshold = lower_threshold or 0.

    def forward(self, out, target):
        """
        Applies the L1 loss with robust masking and dimensional checks.
        :param out: Network output, expected shape [batch_size, num_quantiles, height, width] or [batch_size, num_quantiles, spatial_dim].
        :param target: Target values, expected shape [batch_size, height, width] or [batch_size, spatial_dim].
        :return: L1 loss.
        """
        # Flatten spatial dimensions if necessary
        # print(f"L1 Original out shape: {out.shape}")  # After initial creation
        # print(f"L1 Original target shape: {target.shape}")
        if out.ndimension() == 4:
            # Reshape 4D tensor to [batch_size, num_quantiles, spatial_dim]
            batch_size, num_quantiles, height, width = out.shape
            out = out.view(batch_size, num_quantiles, -1)  # Flatten height and width
            # target = target.view(batch_size, -1)  # Flatten target
            target = target.view(batch_size, -1)[:, :out.shape[2]]  # Match target's spatial_dim to out's
            # print(f"L1 4D Shape: out: {out.shape}, target: {target.shape}")
        elif out.ndimension() == 3:
            # If already [batch_size, num_quantiles, spatial_dim], use directly
            batch_size, num_quantiles, spatial_dim = out.shape
            #target = target.view(batch_size, -1)
            target = target.view(batch_size, -1)[:, :spatial_dim]
            # print(f"L1 3D Shape: out: {out.shape}, target: {target.shape}")
        else:
            raise ValueError(f"L1 Unexpected tensor dimensions for `out`: {out.shape}")

        # Apply mask if `ignore_value` is specified
        if self.ignore_value is not None:
            # Create a mask for valid target entries
            mask = target != self.ignore_value  # Shape: [batch_size, spatial_dim]
            
            # Check consistency of dimensions
            if mask.shape[-1] != out.shape[-1]:
                raise ValueError(f"L1 Mask shape {mask.shape} does not match `out` shape {out.shape}.")

            # Expand the mask to match the dimensions of `out`
            mask = mask.unsqueeze(1).expand(-1, num_quantiles, -1)  # Shape: [batch_size, num_quantiles, spatial_dim]

            # Debugging output
            # print(f"L1 Mask shape: {mask.shape}, Out shape: {out.shape}, Target shape: {target.shape}")

            # Apply the mask to `out` and `target`
            out = out[mask].view(-1, num_quantiles)  # Filter `out` and reshape
            target = target[mask[:, 0, :]]  # Filter `target` accordingly

        # Compute the L1 loss
        loss = torch.abs(target - out.mean(dim=1))  # Reduce across quantiles if necessary
        return loss.mean()

    """
    def forward(self, out, target):
        
        Applies the L1 loss
        :param out: output of the network
        :param target: target
        :return: l1 loss
        
        if self.pre_calculation_function != None:
            out, target = self.pre_calculation_function(out, target)

        # out = out.flatten()
        # target = target.flatten()

        out = out.view(out.size(0), -1)  # Flatten spatial dimensions
        target = target.view(target.size(0), -1)

        # variant if other loss than pinball is calculated
        if self.ignore_value is not None:
            out = out[target != self.ignore_value]
            target = target[target != self.ignore_value]
        

        # variant if pinball loss is calculated

        if self.ignore_value is not None:
            # Create the original mask for valid entries
            mask = target != self.ignore_value  # [batch_size, spatial_dim]
            print(f"Original mask shape: {mask.shape}")  # After initial creation

            # Ensure the mask matches `out` dimensions
            batch_size, num_quantiles, spatial_dim = out.shape
            mask = mask.view(batch_size, -1)  # Ensure mask is [batch_size, spatial_dim]
            mask = mask.unsqueeze(1).expand(batch_size, num_quantiles, spatial_dim)  # Expand mask to [batch_size, num_quantiles, spatial_dim]
            print(f"Expanded mask shape: {mask.shape}")  # After unsqueeze and expand

            # Apply the mask to filter `out` and reshape
            out = out[mask].view(-1, num_quantiles)  # Filter `out` and reshape
            target = target.view(batch_size, -1)[mask[:, 0, :]]  # Filter `target` accordingly
            print(f"out shape: {out.shape}")  # Shape of `out` tensor before and after masking
            print(f"target shape: {target.shape}")  # Shape of `target` tensor before and after masking


        if self.lower_threshold > 0:
            mask = target > self.lower_threshold  # Mask for threshold
            out = out[mask]
            target = target[mask]

        # variant if other loss than pinball is calculated
        if self.lower_threshold > 0:
            out = out[target > self.lower_threshold]
            target = target[target > self.lower_threshold]
        

        loss = torch.abs(target - out)

        return loss.mean()
"""