import os
import json
import csv
import torch
import numpy as np
import gc
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from torchvision.transforms import transforms
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#import training.visualization as viz
from prediction.config import MODEL_PATHS, DEVICE, OUTPUT_DIR, BATCH_SIZE, NUM_WORKERS, MODEL_VARIANT, DEBUG_MODE
from prediction.prediction_utils import get_dataloader
from prediction.fixval_visualization import get_input_output_visualization
import wandb

BATCH_SIZE = 1
CHUNK_SIZE = 20  # Save after every 100 samples


class Predictor:
    def __init__(self, model_paths, device=DEVICE, log_to_wandb=True):
        """Initializes the Predictor class."""
        self.device = torch.device(device)
        self.models = []
        self.configs = []
        self.log_to_wandb = log_to_wandb
        self.ensemble_size = 0  # Only relevant for ensemble models
  
        for model_path in model_paths:
            if MODEL_VARIANT in model_path and model_path.endswith(".pt"):
                model_data = torch.load(model_path, map_location=self.device, weights_only=False)
                model_state_dict = model_data["state_dict"]
            
                # ✅ Ensure 'config' exists and extract 'model_variant'
                if isinstance(model_data, dict):
                    config = model_data.get("config", {})
                    
                    # Some models might store 'model_variant' separately—try both ways
                    model_variant = config.get("model_variant") or model_data.get("model_variant", "MISSING")
                    
                    if model_variant == "MISSING":
                        print(f"⚠️ WARNING: Model variant not found in {model_path}, defaulting to MISSING")
                else:
                    print(f"⚠️ WARNING: Model file {model_path} has unexpected structure. Defaulting variant to MISSING.")
                    config = {}
                    model_variant = "MISSING"
                

                model = self._initialize_model(config)
                model.load_state_dict(model_state_dict)
                model.to(self.device)
                model.eval()

                self.models.append((model, model_variant))
                self.configs.append(config)
                if model_variant == "ensemble":
                    self.ensemble_size += 1

                print(f"🔍 Loaded model from {model_path} with variant: {model_variant}")


        if self.ensemble_size > 0:
            print(f"✅ Loaded {self.ensemble_size} models for ensemble prediction.")


    def _initialize_model(self, config):
        """Initializes a model based on the saved config."""
        model_arch = config.get("arch", "unet")
        backbone = config.get("backbone", "resnet50")

        model_mapping = {
            "unet": smp.Unet,
            "unetpp": smp.UnetPlusPlus,
            "manet": smp.MAnet,
            "linknet": smp.Linknet,
            "fpn": smp.FPN,
            "pspnet": smp.PSPNet,
            "pan": smp.PAN,
            "deeplabv3": smp.DeepLabV3,
            "deeplabv3p": smp.DeepLabV3Plus,
        }

        if model_arch not in model_mapping:
            raise ValueError(f"Unknown model architecture: {model_arch}")

        return model_mapping[model_arch](
            encoder_name=backbone,
            encoder_weights=None,
            in_channels=14,
            classes=1,
        )
    

    def predict(self, dataloader, model_variant):
        """Handles inference and processes model outputs efficiently."""
        print(f"\n🚀 Starting prediction for {model_variant}...")
        predictions_list = []

        with torch.no_grad():
            for step, (x_input, file_names) in enumerate(tqdm(dataloader, desc=f"Processing Data ({model_variant})")):
                x_input = x_input.to(self.device, non_blocking=True)
                batch_predictions = []

                if model_variant == "ensemble":
                    for model, _ in self.models:
                        pred = model(x_input).cpu().numpy()
                        batch_predictions.append(pred)
                    batch_predictions = np.array(batch_predictions)

                    # Compute ensemble statistics
                    mean_prediction = np.mean(batch_predictions, axis=0)
                    std_dev = np.std(batch_predictions, axis=0)
                    safe_mean = np.where(mean_prediction == 0, np.nan, mean_prediction)
                    var_coef = np.where(np.isnan(safe_mean), 0, std_dev / safe_mean)
                    lower_bound = np.percentile(batch_predictions, 5, axis=0)
                    upper_bound = np.percentile(batch_predictions, 95, axis=0)
                
                elif model_variant == "quantile_regression":
                    model = self.models[0][0]
                    predictions = model(x_input).cpu().numpy()
                    quantiles = self.configs[0].get("quantiles", [0.05, 0.5, 0.95])
                    prediction_outputs = {f"q{int(q*100)}": predictions[i, q_idx].tolist() for q_idx, q in enumerate(quantiles)}

                elif model_variant == "gaussian_loss":
                    model = self.models[0][0]
                    predictions = model(x_input).cpu().numpy()
                    mean_prediction = predictions[:, 0].tolist()
                    variance_prediction = predictions[:, 1].tolist()
                    std_dev = np.sqrt(predictions[:, 1]).tolist()
                    prediction_outputs = {
                        "mean": mean_prediction,
                        "variance": variance_prediction,
                        "std_dev": std_dev
                    }

                else:  # Baseline Model
                    model = self.models[0][0]
                    mean_prediction = model(x_input).cpu().numpy()
                    std_dev = var_coef = lower_bound = upper_bound = None
                    prediction_outputs = {"mean": mean_prediction.tolist()}

                # Store predictions per sample
                for i, file_name in enumerate(file_names):
                    prediction_entry = {
                        "file_name": os.path.basename(file_name),
                        "input_image": x_input[i].cpu().numpy().tolist(),  # Store input image
                        "variant": model_variant,
                        "outputs": prediction_outputs if model_variant != "ensemble" else {
                            "mean": mean_prediction[i].tolist(),
                            "std_dev": std_dev[i].tolist() if std_dev is not None else None,
                            "var_coef": var_coef[i].tolist() if var_coef is not None else None,
                            "lower_bound": lower_bound[i].tolist() if lower_bound is not None else None,
                            "upper_bound": upper_bound[i].tolist() if upper_bound is not None else None,
                            "individual_predictions": batch_predictions[:, i].tolist()
                        },
                    }
                    predictions_list.append(prediction_entry)

                # Save every CHUNK_SIZE samples and reset the list
                if len(predictions_list) >= CHUNK_SIZE:
                    self.save_predictions(predictions_list, model_variant, intermediate=True)
                    predictions_list = []  # Reset list to free memory

                del x_input, batch_predictions, mean_prediction, std_dev, var_coef, lower_bound, upper_bound
                gc.collect()
                torch.cuda.empty_cache()

        # Save final results
        self.save_predictions(predictions_list, model_variant)

        return predictions_list
    

    def save_predictions(self, predictions, model_variant, intermediate=False):
        """Saves predictions in JSON and CSV formats, supports intermediate saving."""
        output_dir = "prediction_outputs/"
        os.makedirs(output_dir, exist_ok=True)
        suffix = "_intermediate" if intermediate else ""
        json_file = os.path.join(output_dir, f"predictions_{model_variant}{suffix}.json")
        with open(json_file, "w") as f:
            json.dump(predictions, f, indent=4)
        print(f"✅ JSON saved: {json_file}")

        # 🔹 Save NPZ
        """
        print(f"🛠 Start saving to NPZ...")
        npz_file = os.path.join(output_dir, f"predictions_{model_variant}.npz")
        np.savez_compressed(npz_file, predictions=predictions)
        print(f"✅ NPZ saved: {npz_file}")
        """


    def log_to_wandb(self, step, predictions):
        """Logs intermediate predictions to Weights & Biases in a memory-efficient manner."""
        if self.log_to_wandb and wandb.run:
            log_data = {f"step_{step}_sample_{i}": wandb.Image(pred["outputs"]["mean"]) for i, pred in enumerate(predictions[:3])}
            wandb.log(log_data)
            print(f"📡 Logged step {step} to WandB efficiently")


        
    def visualize_results(self, predictions, dataloader, model_variant, log_to_wandb=True):
        """
        Generates visualizations for predictions and logs them to WandB.
        
        :param predictions: Output of predict_*() functions.
        :param dataloader: The DataLoader used for inference.
        :param model_variant: Model variant ('baseline', 'ensemble', 'quantiles', 'gaussian').
        :param log_to_wandb: Whether to log images to WandB.
        """
        print(f"Start generating visualizations for {model_variant}...")
        if DEBUG_MODE:
            print(f"DEBUG: Entering visualize_results() with predictions type = {type(predictions)}")
            print(f"DEBUG: Available keys in predictions dict = {predictions.keys()}")

        if not isinstance(predictions, dict):
            raise TypeError(f"❌ Expected 'predictions' to be a dict, but got {type(predictions)} instead.")

        print(f"Start generating input-output visualizations...")

        # Load the visualization function
        viz_fn = get_input_output_visualization(rgb_channels=[6, 5, 4])

        #for entry in predictions["predictions"]:
        for i, entry in enumerate(predictions["predictions"]):  # Ensure iteration
            file_name = entry["file_name"]
            x_input = torch.tensor(entry["input_image"])  # Convert stored input back to tensor

            # 🔹 Extract outputs based on model type
            if model_variant == "quantiles":
                mean_prediction = torch.tensor(entry["outputs"].get("q50", entry["outputs"]["q50"]))  # Median Quantile
                # Dynamic Quantile Spread Calculation
                quantile_keys = [key for key in entry["outputs"].keys() if key.startswith("q")]
                if len(quantile_keys) >= 2:
                    quantile_keys.sort()  # Ensure ordering (e.g., q5, q50, q95)
                    lowest_quantile = quantile_keys[0]   # Smallest quantile (e.g., q5)
                    highest_quantile = quantile_keys[-1] # Largest quantile (e.g., q95)

                    quantile_spread = (torch.tensor(entry["outputs"][highest_quantile]) - torch.tensor(entry["outputs"][lowest_quantile]))
                    if DEBUG_MODE: print(f"DEBUG: Using {lowest_quantile} - {highest_quantile} for quantile spread visualization")
                else:
                    quantile_spread = None
                    print(f"⚠️ WARNING: Not enough quantiles available for spread calculation!")

                std_dev = None  # Not applicable for quantiles

            else:
                mean_prediction = torch.tensor(entry["outputs"]["mean"])  # Default to mean
                std_dev = torch.tensor(entry["outputs"].get("std_dev", None)) if "std_dev" in entry["outputs"] else None
                quantile_spread = None  # Not applicable for non-quantiles


            #print(f"DEBUG: File {file_name} | Mean Shape: {mean_prediction.shape}")

            # Ensure dimensions are correct
            if mean_prediction.dim() == 2:
                mean_prediction = mean_prediction.unsqueeze(0)  # Add missing channel dim


            # Create visualization (including heatmap if std_dev or quantile_spread exists)
            input_output_plot = viz_fn(inputs=x_input, outputs=mean_prediction, std_dev=std_dev, quantile_spread=quantile_spread)

            
            # Save the visualization
            output_dir = "./prediction_outputs"
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.basename(entry["file_name"])

            output_path = f"{output_dir}/{file_name}_{model_variant}_input_output.png"
            if DEBUG_MODE: print(f"DEBUG: Saving image to {output_path}")
            input_output_plot.savefig(output_path)


            # Log to WandB
            if log_to_wandb and wandb.run:
                wandb_log_data = {f"{file_name}/{model_variant}/input_output": wandb.Image(input_output_plot)}

                if quantile_spread is not None:
                    wandb_log_data[f"{file_name}/{model_variant}/quantile_spread"] = wandb.Image(input_output_plot)

                if std_dev is not None:
                    wandb_log_data[f"{file_name}/{model_variant}/std_dev_heatmap"] = wandb.Image(input_output_plot)

                wandb.log(wandb_log_data)


            print(f"✅ Visualization complete! Images saved to {output_dir}")

            plt.close("all")
