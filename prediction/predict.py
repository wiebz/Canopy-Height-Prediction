import torch
import os
import numpy as np
import wandb
import sys

# Add the project's root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from training.config import FixValDataset, means, stds
from training.runner import Runner

class ConfigObject:
    """ Convert a dictionary to an object with attribute-style access. """
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

def load_training_config(run_id):
    api = wandb.Api()
    try:
        run = api.run(f"test-000/{run_id}")
        config = dict(run.config)  # Convert to standard dictionary
        print(f"Loaded training config from WandB run {run_id}: {config}")
        config["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
        return ConfigObject(config)  # Convert dictionary to object
    except wandb.errors.CommError:
        print(f"Run {run_id} not found. Listing available runs...")
        runs = api.runs("test-000")
        available_runs = [r.id for r in runs]
        print("Available runs:", available_runs)
        run_id = input("Enter a valid WandB run ID from the list above: ").strip()
        return load_training_config(run_id)

def load_ensemble_models(model_paths, config, device):
    models = []
    for model_path in model_paths:
        print(f"Loading model from {model_path}")
        runner = Runner(config=config, tmp_dir=None, debug=False)
        model = runner.get_model(reinit=True, model_path=model_path)
        model.to(device)
        model.eval()
        models.append(model)
    return models

def predict_with_ensemble(models, dataset, batch_size, device):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_predictions = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            model_preds = torch.stack([model(inputs) for model in models], dim=0)
            all_predictions.append(model_preds.cpu())

    all_predictions = torch.cat(all_predictions, dim=1)
    mean_predictions = all_predictions.mean(dim=0)
    variance_predictions = all_predictions.var(dim=0)

    return mean_predictions, variance_predictions, all_predictions

def inverse_transform(predictions, mean, std):
    return predictions * std + mean

def main():
    # Load config from trained model
    run_id = "latest"
    training_config = load_training_config(run_id)
    
    device = training_config.device
    model_save_dir = getattr(training_config, "model_save_dir", "./models")
    output_dir = getattr(training_config, "output_dir", "./predictions")
    data_path = getattr(training_config, "data_path", "/home/ubuntu/work/saved_data/Global-Canopy-Height-Map")
    fixval_csv = getattr(training_config, "fixval_csv", "/home/ubuntu/work/saved_data/Global-Canopy-Height-Map/fix_val.csv")
    batch_size = getattr(training_config, "batch_size", 8)
    dataset_name = getattr(training_config, "dataset", "dataset")
    mean = torch.tensor(means.get(dataset_name, [0] * 14)).view(1, -1, 1, 1)
    std = torch.tensor(stds.get(dataset_name, [1] * 14)).view(1, -1, 1, 1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    model_paths = [os.path.join(model_save_dir, f) for f in os.listdir(model_save_dir) if f.endswith(".pt")]
    assert model_paths, "No model files found."
    
    dataset = FixValDataset(data_path=data_path, dataframe=fixval_csv)
    
    models = load_ensemble_models(model_paths, training_config, device=device)
    mean_preds, var_preds, all_preds = predict_with_ensemble(models, dataset, batch_size, device)
    
    mean_preds = inverse_transform(mean_preds, mean, std)
    var_preds = inverse_transform(var_preds, mean, std)
    
    torch.save(mean_preds, os.path.join(output_dir, "mean_predictions.pt"))
    torch.save(var_preds, os.path.join(output_dir, "variance_predictions.pt"))
    torch.save(all_preds, os.path.join(output_dir, "all_predictions.pt"))
    print("Predictions saved.")
    
    wandb.join()

if __name__ == "__main__":
    main()
