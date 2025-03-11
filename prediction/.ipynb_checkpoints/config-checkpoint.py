import os
import sys
import torch

DEBUG_MODE = "--debug" in sys.argv

# Model Specifications
MODEL_VARIANT = "quantiles"  # Options: "ensemble", "baseline", "quantiles", "gaussian"

# 🔹 Model Paths
MODEL_LOAD_DIR = "models/prediction/"
# Get all models in the directory that match "ensemble_model_*.pt"
MODEL_PATHS = sorted([os.path.join(MODEL_LOAD_DIR, f) for f in os.listdir(MODEL_LOAD_DIR) if f.startswith(f"{MODEL_VARIANT}_model") and f.endswith(".pt")])
if not MODEL_PATHS:
    print(f"⚠️ Warning: No models found in {MODEL_LOAD_DIR}! Ensure training has been run.")
print(f"✅ Found {len(MODEL_PATHS)} {MODEL_VARIANT} model(s) for prediction.")

# 🔹 Data Paths
DATA_PATH = "satellite_data/"
CSV_FILE = "fix_val.csv"

# 🔹 Output Paths
OUTPUT_DIR = "prediction_outputs/"

# 🔹 Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 🔹 Logging
LOG_TO_WANDB = True

# 🔹 Dataloader
BATCH_SIZE = 2
NUM_WORKERS = 4
