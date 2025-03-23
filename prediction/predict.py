import os
import json
import csv
import torch
import seaborn as sns
import numpy as np
import time
import gc
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import segmentation_models_pytorch as smp
from torchvision.transforms import transforms
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#import training.visualization as viz
from prediction.config import MODEL_PATHS, DEVICE, OUTPUT_DIR, BATCH_SIZE, NUM_WORKERS, MODEL_VARIANT, DEBUG_MODE
from prediction.prediction_utils import get_dataloader
from prediction.fixval_visualization import get_input_output_visualization
import wandb


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
                print(f"🔍 Looking for models matching variant: {MODEL_VARIANT}")
                print(f"🔍 Available models: {model_paths}")

                # Allow 'getattr' to avoid UnpicklingError
                torch.serialization.add_safe_globals([getattr])

                # Now load the model safely
                model_data = torch.load(model_path, map_location=self.device, weights_only=False)

                # DEBUG
                # print("🔍 Loaded checkpoint keys:", model_data.keys())  # Should contain 'state_dict'
                # print("🔍 Loaded state_dict keys:", model_data["state_dict"].keys())  # Should include decoder & segmentation head

                # DEBUG
                if "state_dict" not in model_data or "config" not in model_data:
                    raise ValueError(f"❌ ERROR: Model checkpoint missing critical components!")

                model_state_dict = model_data["state_dict"]
                config = model_data.get("config", {})

                print(f"🔍 Debug: Config content: {config}")


                model_variant = config.get("_items", {}).get("model_variant", "MISSING")

                if model_variant == "MISSING":
                    print(f"⚠️ WARNING: Model variant not found in {model_path}, defaulting to MISSING.")

                # # ✅ Ensure 'config' exists and extract 'model_variant'
                # if isinstance(model_data, dict):
                #     config = model_data.get("config", {})
                    
                #     # Some models might store 'model_variant' separately—try both ways
                #     model_variant = config.get("model_variant") or model_data.get("model_variant", "MISSING")
                    
                #     if model_variant == "MISSING":
                #         print(f"⚠️ WARNING: Model variant not found in {model_path}, defaulting to MISSING")
                # else:
                #     print(f"⚠️ WARNING: Model file {model_path} has unexpected structure. Defaulting variant to MISSING.")
                #     config = {}
                #     model_variant = "MISSING"


                model = self._initialize_model(config)

                self.models.append((model, model_variant))  # Store in list


                # Debugging print
                # print("🔍 Model structure at prediction time:")
                # print(model)  # Ensure decoder & segmentation head exist before loading state_dict


                # DEBUG
                #model_data = torch.load(model_path, map_location=self.device, weights_only=True)
                # print(f"🔍 Loaded model keys: {list(model_data['state_dict'].keys())[:10]}")  # Show first 10 keys

                # for name, param in model.named_parameters():
                #     print(f"🛠 Model Parameter: {name}, Shape: {param.shape}")

                # DEBUG
                missing_keys, unexpected_keys = model.load_state_dict(model_data["state_dict"], strict=False)

                if missing_keys:
                    print(f"⚠️ WARNING: Missing keys when loading state dict: {missing_keys}")
                if unexpected_keys:
                    print(f"⚠️ WARNING: Unexpected keys in state dict: {unexpected_keys}")

                # print("🔍 Model architecture at prediction time:", model)
                # print("🔍 Model expected state_dict keys:", model.state_dict().keys())  # Expected layers



                model.load_state_dict(model_state_dict, strict=False)
                model.to(self.device)
                model.eval()

                # self.models.append((model, model_variant)) # weiter oben
                self.configs.append(config)
                if model_variant == "ensemble":
                    self.ensemble_size += 1

                print(f"\n🔍 Loaded model from {model_path} with variant: {model_variant}")

                for model, variant in self.models:
                    sample_params = next(model.parameters()).detach().cpu().numpy().flatten()[:5]
                    print(f"🔍 Model: {variant}, First 5 Parameters: {sample_params}")

                for name, param in model.named_parameters():
                    if torch.isnan(param).any():
                        print(f"🚨 NaN detected in parameter: {name}")




        if self.ensemble_size > 0:
            print(f"✅ Loaded {self.ensemble_size} models for ensemble prediction.")


    def _initialize_model(self, config):
        """Initializes a model based on the saved config."""
        model_arch = config.get("arch", "unet")
        backbone = config.get("backbone", "resnet50")
        num_classes = config.get("num_classes", 1)
        
        print(f"\nInitilaizing model with arch: {model_arch}, backbone: {backbone}, and output classes: {num_classes}\n")

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
            classes=num_classes,
        )
    

    def predict(self, dataloader, model_variant):
        """Handles inference and processes model outputs efficiently."""
        print(f"\n🚀 Starting prediction for {model_variant}...")
        print(f"📌 Total samples in dataset: {len(dataloader.dataset)}")
        
        predictions_list = []
        all_predictions = [] # collect all predictions for final saving
        all_labels = []  # collect all ground truth labels

        start_time = time.time()

        with torch.no_grad():
            for step, (x_input, y_target, file_names) in enumerate(tqdm(dataloader, desc=f"Processing Data ({model_variant})")):
                
                x_input = x_input.to(self.device, non_blocking=True)
                # DEBUG
                if torch.isnan(x_input).sum() > 0:
                    print(f"🔍 x_input Min: {x_input.min()}, Max: {x_input.max()}, NaN Count: {torch.isnan(x_input).sum()}")

                y_target = y_target.to(self.device, non_blocking=True)  # Ground truth labels

                # with torch.amp.autocast('cuda'): # use mixed precision inference
                with torch.amp.autocast('cuda', enabled=False): # Disable autocast to test NaN behavior

                    if model_variant == "ensemble":

                        batch_predictions = []

                        for model, _ in self.models:
                            pred = model(x_input).cpu().numpy()
                            batch_predictions.append(pred)
                        predictions = np.array(batch_predictions)
                        #predictions = np.array(batch_predictions)

                        # Compute ensemble statistics
                        mean_prediction = np.mean(predictions, axis=0)
                        std_dev = np.std(predictions, axis=0)
                        safe_mean = np.where(mean_prediction == 0, np.nan, mean_prediction)
                        var_coef = np.where(np.isnan(safe_mean), 0, std_dev / safe_mean)
                        q10 = np.percentile(predictions, 10, axis=0)
                        q25 = np.percentile(predictions, 25, axis=0)
                        q50 = np.percentile(predictions, 50, axis=0)
                        q75 = np.percentile(predictions, 75, axis=0)
                        q90 = np.percentile(predictions, 90, axis=0)

                        prediction_outputs = {
                            "mean": mean_prediction.tolist(),
                            "std_dev": std_dev.tolist(),
                            "var_coef": var_coef.tolist(),
                            "q10": q10.tolist(),
                            "q25": q25.tolist(),
                            "q50": q50.tolist(),
                            "q75": q75.tolist(),
                            "q90": q90.tolist(),
                            "individual_predictions": predictions[:].tolist()
                        }
                        batch_predictions = []
                        
                    
                    elif model_variant == "quantiles":
                        model = self.models[0][0]
                        predictions = model(x_input).cpu().numpy()

                        if step == 0:
                            # ✅ Debug Step: Print NaN counts and check shapes
                            print(f"🔍 Step {step}: x_input shape: {x_input.shape}, predictions shape: {predictions.shape}")
                            print(f"   ❌ NaN count in predictions: {np.isnan(predictions).sum()}")
                            print(f"   🔹 Min/Max Predictions: {np.nanmin(predictions)}, {np.nanmax(predictions)}")

                        if np.isnan(predictions).sum() > 0:
                            print(f"🚨 WARNING: NaN values detected at Step {step}!")
                            print(f"   File names: {file_names}")
                            print(f"   Model Inputs: {x_input}")
                            print(f"   Model Predictions: {predictions}")
                            #break  # Stop at first NaN detection for debugging
                        
                        # Ensure we correctly extract quantiles from the model config
                        config_dict = self.configs[0]
                        assert config_dict in self.configs, f"config_dict not in self.configs" 
        

                        # # Handle cases where config is wrapped inside '_items'
                        # if isinstance(config_dict, dict):
                        #     if "quantiles" in config_dict:
                        #         quantiles = config_dict["quantiles"]
                        #     elif "_items" in config_dict and "quantiles" in config_dict["_items"]:
                        #         quantiles = config_dict["_items"]["quantiles"]
                        #     else:
                        #         print("❌ ERROR: Could not find quantiles in model config!")
                        #         quantiles = [0.05, 0.5, 0.95]  # Default values to avoid crashes
                        # else:
                        #     print("❌ ERROR: Unexpected config format in predict.py!")
                        #     quantiles = [0.05, 0.5, 0.95]  # Default values

                        if "_items" in config_dict and "quantiles" in config_dict["_items"]:
                                quantiles = config_dict["_items"]["quantiles"]
                        else:
                            print("❌ ERROR: Could not find quantiles in model config!")
                            quantiles = [0.05, 0.5, 0.95]  # Default values to avoid crashes


                        # ✅ Print only for the first step
                        if step == 0:
                            print(f"🔍 Quantile levels used: {quantiles}")

                        if not isinstance(quantiles, list) or len(quantiles) == 0:
                            print("❌ ERROR: No valid quantile levels found! Check model config.")
                    
                        prediction_outputs = {f"q{int(q*100)}": predictions[:, q_idx].tolist() for q_idx, q in enumerate(quantiles)}

                    elif model_variant == "gaussian":
                        model = self.models[0][0]
                        predictions = model(x_input).cpu().numpy()
                        
                        if step == 0:
                            print(f"🔍 Gaussian predictions shape: {predictions.shape}")
                        if predictions.shape[1] != 2:
                            print("❌ ERROR: Gaussian model must output 2 channels (mean, variance)!")

                        # Compute statistics
                        mean = predictions[:, 0]
                        variance = predictions[:, 1]
                        std_dev = np.sqrt(predictions[:, 1])

                        from scipy.stats import norm

                        z_50 = norm.ppf(0.5 + 0.50 / 2)  # 0.674
                        z_80 = norm.ppf(0.5 + 0.80 / 2)  # 1.282
                        n = len(predictions[:,1])

                        # z = norm.ppf(0.5 + confidence / 2)
                        lower_80 = mean - z_80 * std_dev / np.sqrt(n)
                        lower_50 = mean - z_50 * std_dev / np.sqrt(n)
                        upper_50 = mean + z_50 * std_dev / np.sqrt(n)
                        upper_80 = mean + z_80 * std_dev / np.sqrt(n)


                        prediction_outputs = {
                            "mean": mean.tolist(),
                            "variance": variance.tolist(),
                            "std_dev": std_dev.tolist(),
                            "lower_80": lower_80.tolist(),
                            "lower_50": lower_50.tolist(),
                            "upper_50": upper_50.tolist(),
                            "upper_80": upper_80.tolist()
                        }

                    else:  # Baseline Model
                        model = self.models[0][0]
                        # predictions = model(x_input).cpu().numpy()

                        # DEBUG
                        raw_predictions = model(x_input)  # Get raw model output (before activation)
                        # ✅ Debugging: Print Min/Max values and check for NaNs/Infs
                        if torch.isnan(raw_predictions).sum().item() > 0:
                            print(f"🔍 Raw Model Output - Min: {raw_predictions.min().item()}, Max: {raw_predictions.max().item()}")
                            print(f"❌ NaN count: {torch.isnan(raw_predictions).sum().item()}, Inf count: {torch.isinf(raw_predictions).sum().item()}")
                        predictions = raw_predictions.cpu().numpy()  # Move to CPU for further processing


                        if step == 0:
                            # ✅ Debug Step: Print NaN counts and check shapes
                            print(f"🔍 Step {step}: x_input shape: {x_input.shape}, predictions shape: {predictions.shape}")
                            print(f"   ❌ NaN count in predictions: {np.isnan(predictions).sum()}")
                            print(f"   🔹 Min/Max Predictions: {np.nanmin(predictions)}, {np.nanmax(predictions)}")

                        if np.isnan(predictions).sum() > 0:
                            print(f"🚨 WARNING: NaN values detected at Step {step}!")
                            print(f"   File names: {file_names}")
                            print(f"   Model Inputs: {x_input}")
                            print(f"   Model Predictions: {predictions}")
                            break  # Stop at first NaN detection for debugging


                        prediction_outputs = {"mean": predictions.tolist()}

                if np.isnan(predictions).sum() > 0:
                    print(f"🚨 First Batch NaN Count: {np.isnan(predictions).sum()}")
                    print(f"🔹 Model Output Sample: {predictions[:5]}")


                # Store predictions per sample
                for i, file_name in enumerate(file_names):
                    prediction_entry = {
                        "file_name": os.path.basename(file_name),
                        "input_image": x_input[i].cpu().numpy().tolist(),  # Store input image
                        "variant": model_variant,
                        "outputs": {key: value[i] if isinstance(value, list) else np.array(value)[i].tolist()
                                    for key, value in prediction_outputs.items()},
                        "ground_truth": y_target[i].cpu().numpy().tolist(),  # Save ground truth labels
                    }
                    predictions_list.append(prediction_entry)

                    # ✅ Store minimal data for visualization (no full image data)
                    all_labels.extend(y_target.cpu().numpy().tolist())  # Append full batch
                    if model_variant == 'quantiles':
                        all_predictions.extend(prediction_outputs["q50"])
                    else:
                        all_predictions.extend(prediction_outputs["mean"])  # Append full batch predictions

                #self.save_predictions(predictions_list, model_variant)  # Save immediately

                # ✅ Save every `SAVE_INTERVAL` batches instead of every batch
                # if step % CHUNK_SIZE == 0 or step == len(dataloader) - 1:
                if step > 0 and (step % CHUNK_SIZE == 0 or step == len(dataloader) - 1):
                    self.save_predictions(predictions_list, model_variant)
                    predictions_list.clear()  # Correct way to clear without breaking the reference


                # ✅ Print only for the first step
                if step == 0:
                    print(f"🔍 Final prediction outputs for {model_variant}: {prediction_outputs.keys()}")
                if not prediction_outputs:
                    print("❌ ERROR: No predictions found! Model might not be running correctly.")
                    

                # ✅ Improved Memory Management
                del x_input, y_target, prediction_outputs, predictions
                # if model_variant == "ensemble":
                #     del batch_predictions
                # else:
                #     del predictions

                # if model_variant == "ensemble":
                #     del batch_predictions, mean_prediction, std_dev, var_coef, lower_bound, upper_bound
                # elif model_variant == "quantiles":
                #     del predictions, prediction_outputs
                # elif model_variant == "gaussian":
                #     del predictions, mean_prediction, variance_prediction, std_dev, prediction_outputs
                # else:  # Baseline
                #     del mean_prediction, prediction_outputs
                if step > 0 and (step % CHUNK_SIZE == 0 or step == len(dataloader) - 1):
                    gc.collect()
                    torch.cuda.empty_cache()

                # if step % 10 == 0:
                #     print(f"🔄 Processed {step} samples so far...")


            total_time = time.time() - start_time
            total_minutes = total_time // 60
            total_seconds = total_time % 60
            print(f"⏳ Total Prediction Time: {total_time:.2f}s -> {int(total_minutes)}:{total_seconds:.2f} | Speed: {len(dataloader.dataset) / total_time:.2f} samples/sec")


        # self.save_predictions(predictions_list, model_variant, intermediate=False)
        # print(f"✅ Final prediction count: {len(predictions_list)}")
        # if len(predictions_list) != len(dataloader.dataset):
        #     print(f"❌ WARNING: Mismatch in expected predictions! {len(predictions_list)} saved vs. {len(dataloader.dataset)} expected.")
        # else:
        #     print(f"🎉 Successfully processed all {len(predictions_list)} samples!")


        # ✅ Convert lists to NumPy arrays for visualization
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)

        # Create visualizations
        # viz_start = time.time()
        # self.create_fixval_visualizations(all_labels, all_predictions, model_variant)
        # print(f"⏳ Visualization time: {time.time() - viz_start:.3f} seconds")

        # 🕒 Log total execution time
        print(f"⏳ Total prediction time: {time.time() - start_time:.3f} seconds")

        return all_labels, all_predictions


    def save_predictions(self, predictions, model_variant):
        """Saves each prediction as an individual file inside a model-variant folder."""
        output_dir = os.path.join("prediction_outputs", model_variant)
        os.makedirs(output_dir, exist_ok=True)

        for pred in predictions:
            file_name = pred["file_name"].split('.')[0]  # Remove extension
            npz_path = os.path.join(output_dir, f"{file_name}.npz")

            try:
                np.savez_compressed(npz_path, **pred)
                # print(f"✅ Saved prediction: {npz_path}")

            except Exception as e:
                print(f"❌ NPZ save failed for {file_name}: {e}")
        
        print(f"✅ {len(predictions)} images saved to {output_dir}.")

        # Free memory
        del predictions
        gc.collect()
        torch.cuda.empty_cache()

    

    def old_save_predictions(self, predictions, model_variant, intermediate=False):
        """Saves predictions in JSON and CSV formats, supports intermediate saving."""
        output_dir = "prediction_outputs/"
        os.makedirs(output_dir, exist_ok=True)
        suffix = "_intermediate" if intermediate else ""

        """
        json_file = os.path.join(output_dir, f"predictions_{model_variant}{suffix}.json")
        with open(json_file, "w") as f:
            json.dump(predictions, f, indent=4)
        print(f"✅ JSON saved: {json_file}")

        # 🔹 Save NPZ
        """
        print(f"🛠 Start saving to NPZ...")
        npz_file = os.path.join(output_dir, f"predictions_{model_variant}{suffix}.npz")

        try:
            np.savez_compressed(npz_file, predictions=predictions)  # Save everything at once
            print(f"✅ NPZ saved: {npz_file}")
        except Exception as e:
            print(f"❌ NPZ save failed: {e}")

        # try:
        #     with open(npz_file, "ab") as f:
        #         np.save(f, predictions)  # Append each prediction directly
        #     print(f"✅ NPZ updated: {npz_file} (Batch saved)")
        # except Exception as e:
        #     print(f"❌ NPZ save failed: {e}")

        # Free memory
        del predictions
        gc.collect()
        torch.cuda.empty_cache()

        # try:
        #     npz_file = os.path.join(output_dir, f"predictions_{model_variant}_intermediate.npz")

        #     if os.path.exists(npz_file):
        #         existing_data = np.load(npz_file, allow_pickle=True)
        #         existing_predictions = existing_data["predictions"].tolist()  # Convert to list
        #         combined_predictions = np.array(existing_predictions + predictions, dtype=object)
        #     else:
        #         combined_predictions = np.array(predictions, dtype=object)

        #     np.savez_compressed(npz_file, predictions=combined_predictions)
        #     print(f"✅ NPZ updated: {npz_file}")

        # except Exception as e:
        #     print(f"❌ NPZ save failed: {e}")


    def log_to_wandb(self, step, predictions):
        """Logs intermediate predictions to Weights & Biases in a memory-efficient manner."""
        if self.log_to_wandb and wandb.run:
            log_data = {f"step_{step}_sample_{i}": wandb.Image(pred["outputs"]["mean"]) for i, pred in enumerate(predictions[:3])}
            wandb.log(log_data)
            print(f"📡 Logged step {step} to WandB efficiently")


    def create_fixval_visualizations(self, labels, predictions, model_variant, ignore_value=0, min_value=0, max_value=60, bins=30):
        """Creates scatter plot and boxplot for FixVal dataset."""

        print(f"Start creating visualizations...")

        if labels.size == 0 or predictions.size == 0:
            print(f"❌ No prediction data available for {model_variant}, skipping visualizations.")
            return

        output_dir = os.path.join("prediction_outputs", model_variant)
        os.makedirs(output_dir, exist_ok=True)

        labels = np.array(labels)
        predictions = np.array(predictions)

        print(f"Label shape: {labels.shape}, Prediction shape: {predictions.shape}")

        # Remove NaNs from labels and predictions
        mask = (labels != ignore_value) & ~np.isnan(labels) & ~np.isnan(predictions)
        labels, predictions = labels[mask], predictions[mask]

        # Ensure both x and y values are within the range
        labels = np.clip(labels, min_value, max_value)
        predictions = np.clip(predictions, min_value, max_value)
        
        # ✅ Scatter Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(labels, predictions, alpha=0.5, s=5, color="blue")
        # Symmetric axis scaling
        ax.set_xlim([min_value, max_value])
        ax.set_ylim([min_value, max_value])
        ax.set_aspect('equal', adjustable='box')  # Force square aspect ratio
        ax.plot([min_value, max_value], [min_value, max_value], c="r", linestyle="--", linewidth=2, label="x=y") # x=y reference line red
        ax.set_xlabel("Ground Truth Height (m)")
        ax.set_ylabel("Predicted Height (m)")
        ax.set_title(f"FixVal: Scatter Plot ({model_variant})")
        #ax.legend()
        ax.legend(loc="upper left")

        # Save plot to wandb
        if wandb.run:
            wandb.log({f"{model_variant}/scatter_plot": wandb.Image(fig)})
            print(f"✅ Scatter plot logged to WandB.")

        scatter_path = os.path.join(output_dir, f"fixval_scatter_{model_variant}.png")
        #fig.savefig(scatter_path)
        fig.savefig(scatter_path, bbox_inches='tight', dpi=100) # remove whitespace and reduce resolution for faster saving
        plt.close(fig)
        print(f"✅ Scatter plot saved: {scatter_path}")

        # ✅ Boxplot
        
        # Remove ignored values
        mask = labels != ignore_value
        labels = labels[mask]
        predictions = predictions[mask]
        
        bin_size = 5
        bins = np.arange(min_value, max_value, bin_size)

        bin_indices = np.digitize(labels, bins)
        bin_errors = [[] for _ in range(len(bins) + 1)]
        bin_counts = [0] * (len(bins) + 1)

        for bin_idx, true_value, predicted_value in zip(bin_indices, labels, predictions):
            error = predicted_value - true_value
            bin_errors[bin_idx].append(error)
            bin_counts[bin_idx] += 1

        x_labels = [f"{bins[i]}-{bins[i] + bin_size} (n={bin_counts[i]:,})" for i in range(len(bin_counts) - 1)]
        x_labels.append(f">{bins[-1]} (n={bin_counts[-1]:,})")

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=bin_errors, fliersize=2)
        plt.xticks(rotation=90)
        plt.xlabel("Label Bins")
        plt.ylabel("Prediction Error")
        plt.title(f"FixVal: Boxplot ({model_variant})")

        # Save plot to wandb
        if wandb.run:
            wandb.log({f"{model_variant}/boxplot": wandb.Image(plt)})
            print(f"✅ Boxplot logged to WandB.")
        boxplot_path = os.path.join(output_dir, f"fixval_boxplot_{model_variant}.png")
        #plt.savefig(boxplot_path)
        fig.savefig(boxplot_path, bbox_inches='tight', dpi=100) # remove whitespace for faster saving
        plt.close()
        print(f"✅ Boxplot saved: {boxplot_path}")

        # ✅ Log to WandB
        # if wandb.run:
        #     wandb.log({f"{model_variant}/scatter_plot": wandb.Image(scatter_path)})
        #     wandb.log({f"{model_variant}/boxplot": wandb.Image(boxplot_path)})




    def old_create_fixval_visualizations(self, predictions, model_variant):
        """Creates scatter plot and boxplot for FixVal dataset."""
        output_dir = "prediction_outputs/"
        os.makedirs(output_dir, exist_ok=True)
        labels = np.array([p["ground_truth"] for p in predictions])
        if model_variant == "quantiles":
            preds = np.array([p["outputs"]["q50"] for p in predictions])
        else:
            preds = np.array([p["outputs"]["mean"] for p in predictions])

        # Scatter Plot
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(labels, preds, alpha=0.5, s=5, color="blue")
        ax.plot([0, 60], [0, 60], c="r", linestyle="--", linewidth=2, label="x=y")
        ax.set_xlabel("Ground Truth Height (m)")
        ax.set_ylabel("Predicted Height (m)")
        ax.set_title("FixVal: Scatter Plot of Predictions vs. Ground Truth")
        ax.legend()
        scatter_path = os.path.join(output_dir, f"fixval_scatter_{model_variant}.png")
        fig.savefig(scatter_path)
        plt.close(fig)
        print(f"✅ Scatter plot saved: {scatter_path}")

        # Boxplot
        bin_size = 5
        bins = np.arange(0, 60, bin_size)
        bin_indices = np.digitize(labels, bins)
        bin_errors = [[] for _ in range(len(bins) + 1)]
        bin_counts = [0] * (len(bins) + 1)

        for bin_idx, error in zip(bin_indices, preds - labels):
            bin_errors[bin_idx].append(error)
            bin_counts[bin_idx] += 1

        x_labels = [f"{bins[i]}-{bins[i] + bin_size} (n={bin_counts[i]:,})" for i in range(len(bin_counts) - 1)]
        x_labels.append(f">{bins[-1]} (n={bin_counts[-1]:,})")

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=bin_errors, fliersize=2)
        plt.xticks(rotation=90)
        plt.xlabel("Label Bins")
        plt.ylabel("Prediction Error")
        plt.title("FixVal: Boxplot of Errors for Tree Height Bins")
        boxplot_path = os.path.join(output_dir, f"fixval_boxplot_{model_variant}.png")
        plt.savefig(boxplot_path)
        plt.close()
        print(f"✅ Boxplot saved: {boxplot_path}")

        # Optionally log to WandB
        if wandb.run:
            wandb.log({f"fixval/scatter_plot": wandb.Image(scatter_path)})
            wandb.log({f"fixval/boxplot": wandb.Image(boxplot_path)})



        
    def old_visualize_results(self, predictions, dataloader, model_variant, log_to_wandb=True):
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
