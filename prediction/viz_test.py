import os
import json
import torch
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import ipywidgets as widgets
import wandb

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

os.chdir('/home/ubuntu/work/saved_data/Global-Canopy-Height-Map')

# Adjust model variant
MODEL_VARIANT = 'gaussian' #'baseline', 'ensemble', 'gaussian', 'quantiles'

PREDICTION_FOLDER = f"prediction_outputs/{MODEL_VARIANT}"
print(PREDICTION_FOLDER)

# PREDICTIONS_FILE = f"prediction_outputs/predictions_{MODEL_VARIANT}.npz"
# print(PREDICTIONS_FILE)

def inspect_npz_structure(predictions_file, num_entries=3):
    """
    Loads and prints an overview of the JSON file structure.
    
    :param predictions_file: Path to the JSON file containing predictions.
    :param num_entries: Number of entries to print for inspection.
    """
    npz_data = np.load(predictions_file, allow_pickle=True)
    predictions = npz_data["predictions"].tolist()  # Convert numpy array back to list of dicts
    print(f"🔍 npz contains {len(predictions)} prediction entries.\n")
    
    # Inspect general structure
    print("🗂 Available keys in first entry:")
    for key in npz_data[0].keys():
        print(f"  - {key}")

    # Inspect details inside an entry
    print("\n🔹 Example structure of first entry:")
    example_entry = npz_data[0]
    for key, value in example_entry.items():
        if isinstance(value, dict):
            print(f"  {key}: (Dictionary with keys: {list(value.keys())})")
        elif isinstance(value, list):
            print(f"  {key}: (List of length {len(value)})")
        else:
            print(f"  {key}: {type(value)}")

    # Inspect available output keys
    print("\n🛠 Available output keys in 'outputs':")
    if "outputs" in example_entry:
        for output_key, output_value in example_entry["outputs"].items():
            print(f"  - {output_key}: {type(output_value)} | Shape: {np.shape(output_value)}")

    # Inspect individual ensemble members
    print("\n👥 Checking for ensemble member predictions...")
    if "ensemble_members" in example_entry["outputs"]:
        num_members = len(example_entry["outputs"]["ensemble_members"])
        print(f"  ✅ Found {num_members} ensemble member predictions.")
    else:
        print("  ❌ No ensemble member predictions found. Ensure they are saved correctly.")

    # Print a sample entry
    print("\n📌 Example entry preview:")
    print(example_entry)

# Run the inspection
#inspect_json_structure(PREDICTIONS_FILE)

"""old"""
def stream_predictions(predictions_file):
    """
    Stream predictions one-by-one from the JSON file to avoid high memory usage.
    """
    # Load npz file
    npz_data = np.load(predictions_file, allow_pickle=True)
    for entry in npz_data["predictions"]: # Process one entry at a time
        yield entry.item()  # Return one image entry at a time

def stream_predictions_from_folder(folder_path):
    """
    Generator that yields prediction entries from individual .npz files in a folder.
    """
    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith(".npz"):
            npz_path = os.path.join(folder_path, fname)
            try:
                data = np.load(npz_path, allow_pickle=True)
                # Reconstruct entry as dictionary (each file contains one prediction)
                entry = {key: data[key].item() if isinstance(data[key], np.ndarray) and data[key].shape == () else data[key] for key in data.files}
                yield entry
            except Exception as e:
                print(f"❌ Failed to load {fname}: {e}")



def visualize_prediction_interactive(entry, image_index=0, save_path="saved_visualizations", rgb_channels=[6, 5, 4]):
    """
    Interactive visualization for different model variants, now comparing Q0.1-Q0.9 (Quantiles) 
    with the 80% confidence interval (Gaussian).

    :param entry: Prediction entry from the JSON file.
    :param image_index: The index of the image within a batch.
    :param save_path: Directory where visualizations will be saved.
    :param rgb_channels: Indices for RGB visualization.
    """
    file_name = entry["file_name"]
    model_variant = entry["variant"]
    print(f"Creating visualizations for variant: {model_variant}, Image {image_index + 1}")
    
    # ✅ Extract prediction outputs
    outputs = entry["outputs"]

    # ✅ Extract correct input image from batch
    # x_input = np.array(entry["input_image"])  # Extract the correct image from batch
    x_input = entry["input_image"]
    print(f"Shape of entry['input_image'] saved into x_input: {x_input.shape}")
    
    # if x_input.ndim == 2:  # Missing channel dim (H, W)
    #     print("⚠️ WARNING: Detected missing channel dimension! Expanding to 1 channel.")
    #     x_input = np.expand_dims(x_input, axis=0)  # Convert (H, W) → (1, H, W)
    
    #x_input = x_input[image_index]  # Extract single image
    # print(f"📏 Input image shape after extraction: {x_input.shape}")

    # Normalize input for visualization
    x_input_normalized = np.clip(x_input / 3000, 0, 1)

    clear_output(wait=True)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    
    # 🔹 **First Column: Input Image**
    axs[0].imshow(x_input_normalized[rgb_channels, :, :].transpose(1, 2, 0))
    #axs[0].set_title(f"Input Image: {file_name}")
    axs[0].set_title(f"Input Image: {file_name}")
    axs[0].axis("off")


    # Determine visualization based on model variant
    if model_variant == "baseline":
        mean_prediction = outputs["mean"]
        
        im = axs[1].imshow(mean_prediction, cmap="viridis", vmin=0, vmax=35)
        axs[1].set_title("Baseline Prediction")

        # No uncertainty visualization for baseline → Leave axs[2] empty
        axs[2].axis("off")


    elif model_variant == "ensemble":
        mean_prediction = outputs["mean"]
        variance =  outputs["variance"]
        
        # Plot mean prediction
        im = axs[1].imshow(mean_prediction.numpy(), cmap="viridis", vmin=0, vmax=35)
        axs[1].set_title("Ensemble Mean Prediction")
        
        # Plot variance
        im_var = axs[2].imshow(variance.numpy(), cmap="magma", vmin=0, vmax=variance.max())
        axs[2].set_title("Ensemble Variance")

        divider2 = make_axes_locatable(axs[2])
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im_var, cax=cax2, orientation="vertical")


    
    elif model_variant == "quantiles":
        # print(f"🔍 Available keys in outputs: {outputs.keys()}")  # Debugging

        median_prediction = outputs["q50"]
        q10 = outputs["q10"]
        q90 = outputs["q90"]

        # ✅ Fix the shape issue before plotting
        print(f"Shapes - Median: {median_prediction.shape}, Q0.1: {q10.shape}, Q0.9: {q90.shape}")

        # Plot the median prediction
        im = axs[1].imshow(median_prediction, cmap="viridis", vmin=0, vmax=35)
        axs[1].set_title("Median Prediction (Quantiles)")

        # Plot quantile interval heatmap
        interval_width = np.array(q90) - np.array(q10)
        im_ci = axs[2].imshow(interval_width, cmap="magma", vmin=0, vmax=interval_width.max())
        axs[2].set_title("80% Interval Width (Q10 - Q90)")

        divider2 = make_axes_locatable(axs[2])
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im_ci, cax=cax2, orientation="vertical")


    elif model_variant == "gaussian":
        mean_prediction = outputs["mean"]
        lower_bound = outputs["lower_80"]
        upper_bound = outputs["upper_80"]

        im = axs[1].imshow(mean_prediction, cmap="viridis", vmin=0, vmax=35)
        axs[1].set_title("Gaussian Mean Prediction")
        
        # Plot confidence interval heatmap
        interval_width = np.array(upper_bound) - np.array(lower_bound)
        im_ci = axs[2].imshow(interval_width, cmap="magma", vmin=0, vmax=interval_width.max())
        axs[2].set_title("80% Confidence Interval Width")

        divider2 = make_axes_locatable(axs[2])
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im_ci, cax=cax2, orientation="vertical")


    # Add colorbar for predictions
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")

    # Show visualization
    plt.show()

    # 🔹 Show figure and wait for input
    display(fig)  # Works in Jupyter
    plt.show(block=True)  # For script environments
    
    # Prompt user for saving
    user_input = input("Save this visualization? (yes/no/exit): ").strip().lower()
    if user_input == "yes":
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, f"{file_name}_visualization.png")
        fig.savefig(save_file, bbox_inches="tight", dpi=300)
        print(f"✅ Saved visualization: {save_file}")
    elif user_input == "exit":
        return "exit"
    

def run_visualization_interactive_npz_folder(folder_path):
    """
    Iterates through individual .npz predictions in a folder and visualizes each one interactively.
    """
    prediction_stream = stream_predictions_from_folder(folder_path)
    print("Press ENTER to show next image, type 'yes' to save, or 'exit' to stop.")

    for entry in prediction_stream:
        result = visualize_prediction_interactive(entry)
        if result == "exit":
            break


"""old"""
def run_visualization_interactive(predictions_file):
    """
    Iterates through predictions and allows saving or skipping each one.
    """
    prediction_stream = stream_predictions(predictions_file)
    print("Press ENTER to show next image, type 'yes' to save, or 'exit' to stop.")

    # for entry in prediction_stream:
    #     result = visualize_prediction_interactive(entry)
    #     if result == "exit":
    #         break

    print(f"📡 Type of prediction_stream: {type(prediction_stream)} (Generator)")
    # Get first entry
    entry = next(prediction_stream)  # Take one item from the generator
    print(f"📂 Type of entry: {type(entry)} (Dictionary)")

    # Check keys inside entry
    print(f"🔑 Available keys in entry: {entry.keys()}")

    # Check structure of `input_image`
    if "input_image" in entry:
        input_image = np.array(entry["input_image"])  # Convert to NumPy if needed
        print(f"🖼 Shape of entry['input_image']: {input_image.shape}")
    else:
        print("⚠️ 'input_image' key not found in entry!")

    # Check structure of `outputs`
    if "outputs" in entry:
        for key, value in entry["outputs"].items():
            print(f"📊 Shape of outputs['{key}']: {np.shape(value)}")
    else:
        print("⚠️ 'outputs' key not found in entry!")


    for entry in prediction_stream:
        # result = visualize_prediction_interactive(entry)  # when predictions are stored in json file
        result = visualize_prediction_interactive(entry.item())  # when predictions are stored in npz file
        if result == "exit":
            return


    """
    for entry in prediction_stream:
        batch_size = np.shape(entry["input_image"]) 
        print(f"📦 Batch Size: {batch_size}")  # Should be 2

        print(f"🔍 Inspecting entry['input_image']: {type(entry['input_image'])}")
        print(f"📏 Raw Shape: {np.shape(entry['input_image'])}")  # Ensure it's (B, C, H, W)

        for i in range(batch_size):  # Iterate over each image separately
            print(f"\n🖼 Processing Image {i + 1}/{batch_size} in batch")

            single_entry = {
                "file_name": entry["file_name"],  
                "input_image": entry["input_image"][i],  # Extract only one image
                "outputs": {key: np.array(val)[i] for key, val in entry["outputs"].items()},  # Extract corresponding outputs
                "variant": entry["variant"],  # Model variant
            }

            # 🛠 Print Debugging Information
            print(f"📂 Image {i+1} Shape: {single_entry['input_image'].shape}")
            result = visualize_prediction_interactive(single_entry)  # Pass only one image
            if result == "exit":
                return  # Stop execution if user chooses to exit
    """
            

# Run the interactive visualization
# run_visualization_interactive(PREDICTIONS_FILE)

run_visualization_interactive_npz_folder(PREDICTION_FOLDER)
