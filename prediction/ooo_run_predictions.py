import os
import sys
import torch
import wandb
from predict import Predictor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.main import defaults as CONFIG
from training.config import PreprocessedSatelliteDataset
from torch.utils.data import DataLoader

# Initialize WandB (Only run this once!)
wandb.init(project="canopy-predictions", entity="base-001")

# Define model paths
model_paths = [
    "./models/test_ensemble_model_0.pt",
    "./models/test_ensemble_model_1.pt",
    "./models/test_ensemble_model_2.pt",
]

# Define test dataset path
test_data_path = "./"  # Update with correct test dataset location
fix_val_csv = "./fix_val.csv"

# Load test dataset
test_dataset = PreprocessedSatelliteDataset(data_path=test_data_path, dataframe=fix_val_csv)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4)

# Run predictions
predictor = Predictor(model_paths=model_paths)
predictions = predictor.predict(test_dataloader)

# Save predictions & generate visualizations
predictor.visualize_results(predictions, test_dataloader)
predictor.save_uncertainty(predictions)

print("✅ Prediction pipeline completed. Predictions saved.")

wandb.finish()  # End WandB logging
