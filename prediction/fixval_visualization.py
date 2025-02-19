import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_input_output_visualization(rgb_channels=[6, 5, 4]):
    """
    Generates input-output visualizations for FixVal dataset.
    
    :param rgb_channels: The indices of the channels to use for RGB representation.
    :return: Function to visualize input-output pairs.
    """

    def input_output_visualization(inputs, outputs, std_dev=None, quantile_spread=None):
        """
        Creates input-output visualizations with optional uncertainty heatmap.

        :param inputs: Input image tensor.
        :param outputs: Model prediction tensor.
        :param std_dev: Standard deviation (uncertainty), if applicable.
        :param quantile_spread: Spread between quantiles (alternative uncertainty metric), if applicable.
        """
        # Determine how many subplots are needed
        num_subplots = 2  # Default: input + output
        uncertainty = None

        if std_dev is not None:
            uncertainty = std_dev
            num_subplots += 1
        elif quantile_spread is not None:
            uncertainty = quantile_spread
            num_subplots += 1

        fig, axs = plt.subplots(1, num_subplots, figsize=(5*num_subplots, 5))

        inputs = inputs.cpu().detach().numpy()
        outputs = outputs.cpu().detach().numpy()

        print(f"DEBUG: input shape: {inputs.shape}, output shape: {outputs.shape}")

        # Normalize inputs for visualization
        inputs_normalized = np.clip(inputs / 3000, 0, 1)


        # Plot the RGB input image
        #axs[0].imshow(inputs_normalized[0, rgb_channels, :, :].transpose(1, 2, 0))  # Convert CHW to HWC
        axs[0].imshow(inputs_normalized[rgb_channels, :, :].transpose(1, 2, 0))  # Convert CHW to HWC
        axs[0].set_title("RGB Input Image")
        axs[0].axis("off")

        # Plot the model's prediction
        #im = axs[1].imshow(outputs[0, 0, :, :], cmap="viridis", vmin=0, vmax=35)
        im = axs[1].imshow(outputs[0, :, :], cmap="viridis", vmin=0, vmax=35)
        axs[1].set_title("Model Prediction")
        axs[1].axis("off")

        # Create a colorbar for model prediction
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")

        # 🔥 Handle Uncertainty Visualization (std_dev or quantiles)
        if std_dev is not None:
            uncertainty = std_dev.cpu().detach().numpy()  # Convert to NumPy array
            title = "Uncertainty Heatmap (Std Dev)"
        elif quantile_spread is not None:
            uncertainty = quantile_spread.cpu().detach().numpy()  # Convert to NumPy array
            title = "Uncertainty Heatmap (Quantile Spread)"
        else:
            uncertainty = None

        if uncertainty is not None:
            heatmap = axs[2].imshow(uncertainty.squeeze(), cmap="coolwarm", vmin=0, vmax=uncertainty.max())
            axs[2].set_title(title)
            axs[2].axis("off")

            # Add colorbar for the uncertainty heatmap
            divider = make_axes_locatable(axs[2])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(heatmap, cax=cax, orientation="vertical")

        return fig

    return input_output_visualization