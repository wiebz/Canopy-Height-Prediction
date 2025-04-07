import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import torch
from torchvision.transforms import transforms
from matplotlib.colors import Normalize
from matplotlib import cm
from scipy.interpolate import interpn
from tqdm import tqdm
import time
from prediction_utils import FixValDataset  # Import FixValDataset from training config

from sklearn.metrics import mean_absolute_error, mean_squared_error

GENERATE_SCATTER_PLOT = True
SAVE_SCATTER_PLOT = True
GERNERATE_BOXPLOT = True
SAVE_BOX_PLOT = True

model_variant = "gaussian"

# Paths
data_path = "/home/ubuntu/work/saved_data/Global-Canopy-Height-Map/satellite_data"  # Change to correct dataset path
fixval_csv = "fix_val.csv"  # Path to FixVal dataset CSV file

# Define paths

output_dir = os.path.join("prediction_outputs", model_variant)

print(f"📌 Starting FixVal Visualization for model variant: {model_variant}")

# Start time of full process
total_start_time = time.time()

wandb_name=f"Fixval viz {model_variant}"

# ✅ Initialize WandB and log to an existing project
wandb.init(
    mode="offline",  # Add this to prevent network calls
    project="fixval-predictions",
    name=wandb_name,
    resume=True  # ✅ Resume logging in case the script was interrupted before
)

print("✅ WandB initialized successfully!")

# Load FixVal dataset
print(f"📂 Loading FixVal dataset from: {data_path}")
transform = transforms.Compose([transforms.ToTensor()])
dataset = FixValDataset(data_path=data_path, dataframe=fixval_csv, image_transforms=transform)
print(f"✅ Dataset loaded. Total samples: {len(dataset)}")

# Load Predictions
pred_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".npz")])
print(f"📂 Found {len(pred_files)} prediction files in {output_dir}")

labels = []
predictions = []
lowers = []
uppers = []

# Map predictions and ground truth
print("🔄 Matching dataset samples with predictions...")
mapping_start_time = time.time()

for i in tqdm(range(len(dataset)), desc="Processing dataset", unit="sample"):
    image, label, file_name = dataset[i]  # Load dataset sample
    file_name = os.path.basename(file_name)

    pred_file_path = os.path.join(output_dir, f"{file_name}.npz")
    if not os.path.exists(pred_file_path):
        print(f"⚠️ Missing prediction for {file_name}, skipping...")
        continue

    # Load predictions
    data = np.load(pred_file_path, allow_pickle=True)
    # prediction_dict = data["outputs"].item()  # Convert from object array to dict
    
    # if model_variant in ['quantiles', 'ensemble']:
    #     # ✅ Debugging prediction extraction
    #     if 'q50' not in prediction_dict:
    #         print(f"❌ ERROR: 'q50' key missing in {file_name}, skipping...")
    #         continue
    #     prediction = prediction_dict["q50"]  # Extract median prediction
    # else: 
    #     # ✅ Debugging prediction extraction
    #     if "mean" not in prediction_dict:
    #         print(f"❌ ERROR: 'mean' key missing in {file_name}, skipping...")
    #         continue
    #     prediction = prediction_dict["mean"]  # Extract mean prediction

    outputs = data["outputs"].item() # Convert from object array to dict

    if model_variant in ["quantiles", "ensemble"]:
        prediction = outputs.get("q50")
        lower = outputs.get("q10")
        upper = outputs.get("q90")
    
    elif model_variant == "gaussian":
        prediction = outputs.get("mean")
        lower = outputs.get("lower_80")
        upper = outputs.get("upper_80")
    
    if prediction is None or lower is None or upper is None: continue
    
    # Reshape from (H, W) to 1D
    labels.append(label.reshape(-1))
    predictions.append(np.array(prediction).reshape(-1))
    lowers.append(np.array(lower).reshape(-1))
    uppers.append(np.array(upper).reshape(-1))
  

mapping_end_time = time.time()
print(f"✅ Mapping completed in {mapping_end_time - mapping_start_time:.2f} seconds.")

# Convert to numpy arrays
labels = np.concatenate(labels)
predictions = np.concatenate(predictions)
lowers = np.concatenate(lowers)
uppers = np.concatenate(uppers)

print(f"✅ Loaded Labels Shape: {labels.shape}, Predictions Shape: {predictions.shape}, Lower: {lowers.shape}, Upper: {uppers.shape}")

if labels.size == 0 or predictions.size == 0:
    print(f"❌ No valid prediction data available for {model_variant}. Exiting.")
    exit()
if labels.shape != predictions.shape:
    print(f"Mismatch: Labels {labels.shape} vs Predictions {predictions.shape}")


def debug_values(labels, predictions):
    print(f"Labels Shape: {labels.shape}, Predictions Shape: {predictions.shape}")
    print(f"NaN in Labels: {np.isnan(labels).sum()}, NaN in Predictions: {np.isnan(predictions).sum()}")
    print(f"Infinite in Labels: {np.isinf(labels).sum()}, Infinite in Predictions: {np.isinf(predictions).sum()}")
    print(f"Min/Max Labels: {np.nanmin(labels)}, {np.nanmax(labels)}")
    print(f"Min/Max Predictions: {np.nanmin(predictions)}, {np.nanmax(predictions)}")

# Call this before np.histogram2d
debug_values(labels, predictions)

def remove_sub_track_vis(labels, predictions):
    print(f"Before processing - labels shape: {labels.shape}, predictions shape: {predictions.shape}")
    
    if labels.shape[-1] == 3:  # If (H, W, 3), sum over last dimension
        labels = labels.sum(axis=-1)

    print(f"After processing - labels shape: {labels.shape}, predictions shape: {predictions.shape}")

    return labels, predictions



# ✅ Generate Scatter Plot with Symmetric Axes and Density Coloring like in training pipeline
def plot_symmetric_scatter(labels, predictions, ignore_value=0, min_value=0, max_value=60, bins=30, 
                           height_range=range(1, 60), **kwargs):
    """
    Creates a symmetric scatter plot with density coloring to match the existing visualization style.

    :param labels: Ground truth heights
    :param predictions: Predicted heights
    :param ignore_value: Value to ignore in labels
    :param min_value: Minimum axis value
    :param max_value: Maximum axis value
    :param bins: Number of bins for density calculation
    """

    print("📊 Generating scatter plot...")
    start_time = time.time()

    labels = labels.squeeze()
    predictions = predictions.squeeze()

    print(f"Original dimensions of label: {labels.shape} & predictions: {predictions.shape}")

    # labels, predictions = remove_sub_track_vis(labels, predictions)

    print(f"Dimensions after summing labels: {labels.shape} & predictions: {predictions.shape}")

    predictions = predictions[labels != ignore_value]
    labels = labels[labels != ignore_value]

    ax = plt.gca()
    x = np.array(labels).flatten()
    y = np.array(predictions).flatten()

    fig, ax = plt.subplots()

    # Only needed for Density Scatter Plot
    # Compute density for better visualization
    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)

    z = interpn(
        (x_e[:-1], y_e[:-1]), 
        data, 
        np.vstack([x, y]).T, method="splinef2d", 
        bounds_error=False
    )

    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    if np.min(z) < 0:
        z = z + abs(np.min(z) + 0.1)
    norm = Normalize(vmin=np.min(z), vmax=np.max(z), clip=True) # Normalize colors

    # Scatter plot with density-based colors
    ax.scatter(x, y, c=z, s=0.1, cmap="viridis", norm=norm, **kwargs) # alpha=0.5
    
    # Create the scatter plot with a single color
    #ax.scatter(x, y, color="blue", s=5, alpha=1.0)  # No density, fixed blue color

    plt.xlim([min_value, max_value])
    plt.ylim([min_value, max_value])

    fig = plt.gcf()
    plt.grid(False)

    # Colorbar
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, **kwargs
    )
    cbar.ax.set_ylabel("Density")

    # x=y reference line
    ax.plot(height_range, height_range, c="r", linewidth=3, linestyle="--", label="x=y")

    # Labels and title
    ax.set_xlabel("Ground Truth Height (m)")
    ax.set_ylabel("Predicted Height (m)")
    ax.set_title(f"FixVal: Scatter Plot ({model_variant})")

    # ax.legend()
    ax.legend(loc="upper left") # "upper left", "upper right", "lower left", "lower right", "center left"
    
    end_time = time.time()
    scatter_plot_time = end_time - start_time
    print(f"✅ Scatter plot generated in {end_time - start_time:.2f} seconds.")

    return fig



# ✅ Generate Boxplot like in training pipeline
def plot_boxplot(labels, predictions, ignore_value=0, min_value=0, max_value=60, step_size=5):
    """
    Creates an improved boxplot visualization for regression errors.

    :param labels: Ground truth heights
    :param predictions: Predicted heights
    :param ignore_value: Value to ignore in labels
    :param min_value: Minimum height
    :param max_value: Maximum height
    :param step_size: Bin width for the boxplot
    """

    print("📊 Generating boxplot...")
    start_time = time.time()

    # Reshape labels and outputs into 1D arrays
    labels_1d = labels.flatten()
    predictions_1d = predictions.flatten()

    # Calculate errors
    errors = predictions_1d - labels_1d

    # Only keep non-zero label locations
    mask = labels_1d != ignore_value
    labels_1d = labels_1d[mask]
    errors = errors[mask]

    # Define bins
    bins = np.arange(min_value, max_value, step_size)
    bin_indices = np.digitize(labels_1d, bins)

    # Organize errors into bins
    bin_errors = [[] for _ in range(len(bins) + 1)]
    bin_counts = [0] * (len(bins) + 1)
    for bin_idx, error in zip(bin_indices, errors):
        bin_errors[bin_idx].append(error)
        bin_counts[bin_idx] += 1

    # Modify x-axis labels with counts
    x_labels = []
    for i in range(len(bin_counts) - 1):
        x_labels.append(f"{bins[i]}-{bins[i] + step_size} (n={bin_counts[i]:,})")
    x_labels.append(f">{bins[-1]} (n={bin_counts[-1]:,})")

    # Create the boxplot
    plt.figure(figsize=(10, 6))
    box_plot = sns.boxplot(data=bin_errors, fliersize=2)

    box_plot.set_xticklabels(x_labels)
    
    plt.title(f"FixVal: Boxplot ({model_variant})")
    plt.xlabel("Label Bins")
    plt.ylabel("Prediction Error")
    plt.xticks(rotation=90)  # Make sure that the labels on x-axis don't overlap

    end_time = time.time()
    box_plot_time = end_time - start_time
    print(f"✅ Boxplot generated in {end_time - start_time:.2f} seconds.")

    return plt.gcf()  # return the current figure


def calculate_regression_metrics(labels, predictions, ignore_value=0):

    # Create mask of valid pixels
    mask = (labels != ignore_value) & ~np.isnan(labels) & ~np.isnan(predictions) & ~np.isinf(labels) & ~np.isinf(predictions)
    labels = labels[mask]
    predictions = predictions[mask]

    # Calculate prediction metrics
    mae = mean_absolute_error(labels, predictions)
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((labels - predictions) / (labels + 1e-8))) * 100  # avoid division by zero

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE (%)": mape
    }

def evaluate_uncertainty_coverage(labels, predictions, lowers, uppers, ignore_value=0):
    
    # Create mask of valid pixels
    mask = (labels != ignore_value) & ~np.isnan(labels) & ~np.isnan(lowers) & ~np.isnan(uppers)
    labels = labels[mask]
    predictions = predictions[mask]
    lowers = lowers[mask]
    uppers = uppers[mask]

    total = len(labels)
    if total == 0:
        return {}
    
    # Calculate prediction interval width and check if labels are inside
    interval_width = uppers - lowers
    rel_width = interval_width / (predictions + 1e-8)
    coverage = ((labels >= lowers) & (labels <= uppers)).sum()
    print(coverage)

    return {
        "PICP (80%)": (coverage / total) * 100,
        "Mean Lower Bound": np.mean(lowers),
        "Mean Prediction": np.mean(predictions),
        "Mean Upper Bound": np.mean(uppers),
        "MPIW": np.mean(interval_width),
        "Avg Relative Width (%)": np.mean(rel_width) * 100,
        "Count": total
    }


def regression_metrics_per_bin(labels, predictions, ignore_value=0, min_value=0, max_value=60, step_size=5):
    
    # Create valid mask
    mask = (labels != ignore_value) & ~np.isnan(labels) & ~np.isnan(predictions)
    labels = labels[mask]
    predictions = predictions[mask]

    # Create empty bins
    bins = np.arange(min_value, max_value, step_size)
    bin_indices = np.digitize(labels, bins)

    results = []

    # Fill canopy height bins with data
    for i in range(len(bins) + 1):
        bin_labels = labels[bin_indices == i]
        bin_preds = predictions[bin_indices == i]
        if len(bin_labels) == 0: continue

        # Calculate prediction metrics for each bin
        mae = mean_absolute_error(bin_labels, bin_preds)
        mse = mean_squared_error(bin_labels, bin_preds)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((bin_labels - bin_preds) / (bin_labels + 1e-8))) * 100
        
        label_str = f"{bins[i-1]}–{bins[i]}" if 0 < i < len(bins) else f">{bins[-1]}" if i == len(bins) else f"<{bins[0]}"

        results.append({
            "Bin": label_str,
            "Count": len(bin_labels),
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE (%)": mape,
        })
    return pd.DataFrame(results)


def uncertainty_metrics_per_bin(labels, predictions, lowers, uppers, ignore_value=0, min_value=0, max_value=60, step_size=5):

    # Create mask for valid values
    mask = (labels != ignore_value) & ~np.isnan(labels) & ~np.isnan(lowers) & ~np.isnan(uppers)
    labels = labels[mask]
    predictions = predictions[mask]
    lowers = lowers[mask]
    uppers = uppers[mask]

    # Create empty bins
    bins = np.arange(min_value, max_value, step_size)
    bin_indices = np.digitize(labels, bins)

    results = []

    # Fill canopy height bins with data
    for i in range(len(bins) + 1):
        bin_labels = labels[bin_indices == i]
        bin_preds = predictions[bin_indices == i]
        bin_lower = lowers[bin_indices == i]
        bin_upper = uppers[bin_indices == i]
        if len(bin_labels) == 0: continue

        label_str = f"{bins[i-1]}–{bins[i]}" if 0 < i < len(bins) else f">{bins[-1]}" if i == len(bins) else f"<{bins[0]}"

        # Calculate bin interval width and check if ground truth is in prediction interval
        interval_width = bin_upper - bin_lower
        rel_width = interval_width / (bin_preds + 1e-8)
        coverage = ((bin_labels >= bin_lower) & (bin_labels <= bin_upper)).sum()

        results.append({
            "Bin": label_str,
            "Count": len(bin_labels),
            "PICP (80%)": (coverage / len(bin_labels)) * 100,
            "Mean Lower Bound": np.mean(bin_lower),
            "Mean Prediction": np.mean(bin_preds),
            "Mean Upper Bound": np.mean(bin_upper),
            "MPIW": np.mean(interval_width),
            "Avg Relative Width (%)": np.mean(rel_width) * 100,
        })
    return pd.DataFrame(results)



# Save the scatter plot
if GENERATE_SCATTER_PLOT:
    scatter_fig = plot_symmetric_scatter(labels, predictions)

    if wandb.run:
        wandb.log({f"{model_variant}/scatter_plot": wandb.Image(scatter_fig)})
        print(f"✅ Scatter plot logged to WandB.")

    if SAVE_SCATTER_PLOT:
        # Saving the scatter plot to disk takes very long - saving with low resolution 
        scatter_path = os.path.join(output_dir, f"fixval_scatter_{model_variant}.png")
        save_start = time.time()
        # scatter_fig.savefig(scatter_path)
        scatter_fig.savefig(scatter_path, bbox_inches='tight', dpi=100) # remove whitespace for faster saving
        save_end = time.time()
        print(f"✅ Scatter plot saved to {scatter_path} in {save_end - save_start:.2f} seconds.")

    plt.close(scatter_fig)


# Save the box plot
if GERNERATE_BOXPLOT:
    # Save the boxplot
    boxplot_fig = plot_boxplot(labels, predictions)
    if wandb.run:
        wandb.log({f"{model_variant}/boxplot": wandb.Image(boxplot_fig)})
        print(f"✅ Boxplot logged to WandB.")

    if SAVE_BOX_PLOT:
        # Saving to disk takes quite long -> low resolution
        boxplot_path = os.path.join(output_dir, f"fixval_boxplot_{model_variant}.png")
        boxplot_fig.savefig(boxplot_path, bbox_inches='tight', dpi=100) # remove whitespace for faster saving
        print(f"✅ Boxplot saved: {boxplot_path}")

    plt.close(boxplot_fig)


# Calculate metrics
regression_summary = calculate_regression_metrics(labels, predictions)
uncertainty_summary = evaluate_uncertainty_coverage(labels, predictions, lowers, uppers)


# Convert overall metrics to a single-row DataFrame
summary_row = {
    "Bin": "Overall",
    **regression_summary,
    **uncertainty_summary
}
summary_df = pd.DataFrame([summary_row])

# Compute per-bin metrics
df_regression = regression_metrics_per_bin(labels, predictions)
df_uncertainty = uncertainty_metrics_per_bin(labels, predictions, lowers, uppers)


# Merge per-bin tables and combine with overall metrics
df_per_bin = pd.merge(df_regression, df_uncertainty, on="Bin", how="outer")
df_all_metrics = pd.concat([summary_df, df_per_bin], ignore_index=True)

# Save to CSV
csv_path = os.path.join(output_dir, f"fixval_metrics_{model_variant}.csv")
df_all_metrics.to_csv(csv_path, index=False)
print(f"✅ All metrics saved to CSV: {csv_path}")

# Log to WandB
wandb.log({f"{model_variant}/metrics_table": wandb.Table(dataframe=df_all_metrics)})
print("📤 Metrics table logged to WandB")




wandb.finish()
total_end_time = time.time()
print(f"🎉 Visualizations successfully generated in {total_end_time - total_start_time:.2f} seconds!")
