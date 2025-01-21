import argparse
import torch
import os
import numpy as np
from argparse import Namespace
from training.config import FixValDataset
from training.runner import Runner


def load_model(model_path: str, config: dict, device: str):
    """
    Load a trained model from the given path.
    :param model_path: Path to the saved model file.
    :param config: Configuration dictionary for model initialization.
    :param device: Device to load the model on (e.g., 'cuda:0' or 'cpu').
    :return: The loaded model.
    """
    print(f"Loading model from {model_path}")
    # Initialize the Runner to access its model architecture setup
    runner = Runner(config=config, tmp_dir=None, debug=False)
    model = runner.get_model(reinit=True)  # Initialize model architecture

    # Load the saved weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # Set model to evaluation mode
    return model


def load_ensemble_models(model_paths, config, device):
    """
    Load all models in the ensemble.
    :param model_paths: List of paths to model files.
    :param config: Configuration dictionary for model initialization.
    :param device: Device to load the models onto.
    :return: List of loaded models.
    
    # Update device in the config Namespace directly
    config.device = device

    models = []
    for model_path in model_paths:
        print(f"Loading model from {model_path}")
        # Use Runner with a config dictionary or wandb-like object
        runner = Runner(config=config, tmp_dir=None, debug=False)
        model = runner.get_model(reinit=False, model_path=model_path).to(device)
        model.eval()
        models.append(model)
    return models
    """

def load_ensemble_models(model_paths, config, device):
    runner = Runner(config=config, tmp_dir=None, debug=False)
    models = []
    for model_path in model_paths:
        if runner.model is None:
            model = runner.get_model(reinit=True, model_path=model_path).to(device)
        else:
            model = runner.get_model(reinit=False, model_path=model_path).to(device)
        models.append(model)
    return models



def predict(model, dataset, batch_size: int, device: str):
    """
    Perform predictions on the dataset using the loaded model.
    :param model: The trained model.
    :param dataset: Dataset to run predictions on.
    :param batch_size: Batch size for prediction.
    :param device: Device to run the predictions on.
    :return: List of predictions.
    """
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu())  # Move predictions to CPU for further processing
    return torch.cat(predictions, dim=0)


def predict_with_ensemble(models, dataset, batch_size, device):
    """
    Make predictions using the ensemble.
    :param models: List of trained models in the ensemble.
    :param dataset: Dataset for prediction.
    :param batch_size: Batch size for prediction.
    :param device: Device to run predictions on.
    :return: Tuple of (mean_predictions, variance_predictions, all_predictions).
    """
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_predictions = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            # Collect predictions from each model
            model_preds = torch.stack([model(inputs) for model in models], dim=0)  # [n_models, batch_size, ...]
            all_predictions.append(model_preds.cpu())  # Move predictions to CPU for storage

    all_predictions = torch.cat(all_predictions, dim=1)  # Concatenate batches
    mean_predictions = all_predictions.mean(dim=0)  # Mean along the model axis
    variance_predictions = all_predictions.var(dim=0)  # Variance along the model axis

    return mean_predictions, variance_predictions, all_predictions


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Predict with trained model ensemble.")
    parser.add_argument("--model_dir", required=True, help="Path to the directory containing ensemble model files.")
    parser.add_argument("--data_path", 
        default="/Users/wiebkezink/Documents/Uni Münster/MA/dataset", 
        help="Path to the dataset directory (default: /Users/wiebkezink/Documents/Uni Münster/MA/dataset).")
    parser.add_argument("--fixval_csv", 
        default="/Users/wiebkezink/Documents/Uni Münster/MA/dataset/fix_val.csv", 
        help="Path to the fix_val.csv file (default: /Users/wiebkezink/Documents/Uni Münster/MA/dataset/fix_val.csv).")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for predictions (default: 8).")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run predictions on (default: 'cpu').")
    parser.add_argument("--output_dir", 
        default="./predictions", 
        help="Path to save predictions (default: ./predictions).")
    args = parser.parse_args()

    # Load model paths
    model_paths = [os.path.join(args.model_dir, f) for f in os.listdir(args.model_dir) if f.endswith(".pt")]
    assert len(model_paths) > 0, "No model files found in the specified directory."

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    dataset = FixValDataset(data_path=args.data_path, dataframe=args.fixval_csv)

    # Configuration dictionary
    config_dict = {
        "batch_size": args.batch_size,
        "arch": "unet",
        "backbone": "resnet50",
        "use_pretrained_model": False,
        "loss_name": "l2",
        "device": args.device,
        "seed": 42,
        "fp16": False,
        "use_label_rescaling": False,
        "use_weighted_sampler": False,
        "use_weighting_quantile": None,
        "dataset": "default_dataset",
        "num_workers_per_gpu": 8,
        "use_mixup": False,
        "mixup_alpha": 0.2,
        "log_freq": 100,
        "n_iterations": 1000,
        "use_grad_clipping": False,
        "weight_decay": 0.0,
        "optim": "AdamW",
        "model_save_dir": "./models",
        "initial_lr": 1e-3,
        "n_lr_cycles": 0,
        "cyclic_mode": "triangular2",
    }


    # Convert config_dict to Namespace
    config = argparse.Namespace(**config_dict)

    # Load ensemble models
    models = load_ensemble_models(model_paths, config, device=args.device)

    assert os.path.exists(args.data_path), f"Dataset path does not exist: {args.data_path}"
    assert os.path.exists(args.fixval_csv), f"Fixval CSV path does not exist: {args.fixval_csv}"

    print(f"Using dataset path: {args.data_path}")
    print(f"Using fixval CSV: {args.fixval_csv}")

    # Predict with ensemble
    print("Starting ensemble predictions...")
    mean_preds, var_preds, all_preds = predict_with_ensemble(models, dataset, args.batch_size, args.device)

    # Save results
    mean_output_path = os.path.join(args.output_dir, "mean_predictions.pt")
    var_output_path = os.path.join(args.output_dir, "variance_predictions.pt")
    all_preds_output_path = os.path.join(args.output_dir, "all_predictions.pt")

    torch.save(mean_preds, mean_output_path)
    torch.save(var_preds, var_output_path)
    torch.save(all_preds, all_preds_output_path)

    print(f"Mean predictions saved to {mean_output_path}")
    print(f"Variance predictions saved to {var_output_path}")
    print(f"All model predictions saved to {all_preds_output_path}")


if __name__ == "__main__":
    main()