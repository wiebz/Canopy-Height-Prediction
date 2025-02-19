# prediction/logging.py
import wandb
import matplotlib.pyplot as plt

def log_uncertainty_heatmap(image_name, std_dev):
    """
    Logs an uncertainty heatmap (std deviation) to WandB.
    """
    plt.figure(figsize=(10, 5))
    plt.imshow(std_dev, cmap="coolwarm", vmin=0, vmax=std_dev.max())
    plt.colorbar(label="Uncertainty (std dev)")
    plt.title(f"Uncertainty Heatmap - {image_name}")
    file_path = f"prediction_outputs/{image_name}_uncertainty.png"
    plt.savefig(file_path)
    plt.close()

    wandb.log({f"{image_name}/uncertainty_heatmap": wandb.Image(file_path)})

def log_predictions_to_wandb(image_name, input_image, uncertainty_metrics):
    """
    Logs predictions and uncertainty metrics to WandB.
    """
    wandb.log({
        f"{image_name}/mean_prediction": wandb.Image(uncertainty_metrics["mean_pred"]),
        f"{image_name}/std_dev": wandb.Image(uncertainty_metrics["std_dev"]),
        f"{image_name}/ci_90_lower": wandb.Image(uncertainty_metrics["ci_lower"]),
        f"{image_name}/ci_90_upper": wandb.Image(uncertainty_metrics["ci_upper"]),
    })
