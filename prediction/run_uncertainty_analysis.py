import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import wandb
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.visualization import get_visualization_boxplots, get_density_scatter_plot_visualization

# Load saved predictions
predictions = torch.load("prediction_outputs/predictions.pt")

# Prepare to store uncertainty results
uncertainty_results = {}

for (ensemble_filenames, mean_pred, std_dev, var_coef) in predictions["predictions"]:
    for i, filename in enumerate(ensemble_filenames):
        file_uncertainty = {
            "std_dev": std_dev[i].tolist(),
            "var_coef": var_coef[i].tolist(),
            "mean_pred": mean_pred[i].tolist(),
        }
        uncertainty_results[filename] = file_uncertainty

        # Generate uncertainty heatmap
        plt.figure(figsize=(10, 5))
        plt.imshow(std_dev[i].squeeze(), cmap="coolwarm", vmin=0, vmax=std_dev.max())
        plt.colorbar(label="Uncertainty (std dev)")
        plt.title(f"Uncertainty Heatmap - {filename}")
        plt.savefig(f"prediction_outputs/{filename}_uncertainty.png")
        plt.close()

        # Generate scatter & boxplot visualizations
        scatter_plot = get_density_scatter_plot_visualization()(None, torch.tensor(mean_pred[i]), torch.tensor(mean_pred[i]))
        box_plot = get_visualization_boxplots()(None, torch.tensor(mean_pred[i]), torch.tensor(mean_pred[i]))

        scatter_plot.savefig(f"prediction_outputs/{filename}_scatter.png")
        box_plot.savefig(f"prediction_outputs/{filename}_box.png")
        plt.close("all")

        # Log to WandB if enabled
        wandb.log({
            f"{filename}/scatter": wandb.Image(scatter_plot),
            f"{filename}/box_plot": wandb.Image(box_plot),
            f"{filename}/uncertainty_heatmap": wandb.Image(f"prediction_outputs/{filename}_uncertainty.png"),
        })

# Save uncertainty analysis results
with open("prediction_outputs/uncertainty_analysis.json", "w") as f:
    json.dump(uncertainty_results, f, indent=4)

print("✅ Uncertainty analysis completed. Results saved to `prediction_outputs/uncertainty_analysis.json`")
