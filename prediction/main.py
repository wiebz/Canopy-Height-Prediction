import os
import sys
import wandb
from config import DATA_PATH, CSV_FILE, MODEL_PATHS, BATCH_SIZE, NUM_WORKERS, MODEL_VARIANT, DEVICE, DEBUG_MODE, LOG_TO_WANDB
from predict import Predictor
from prediction_utils import get_dataloader


if __name__ == "__main__":

    print("🚀 Starting prediction pipeline...")

    # Initialize WandB
    if LOG_TO_WANDB:
        wandb.init(
            mode="offline",  # Add this to prevent network calls
            project="fixval-predictions",
            #name="fixval_run",  # Set a unique run name
            config={
                "model_variant": MODEL_VARIANT,  # Example metadata
                "batch_size": BATCH_SIZE,
                "device": DEVICE,
            },
            settings=wandb.Settings(init_timeout=180)
        )
        print("✅ WandB initialized successfully!")

    
    # ✅ Load dataset
    dataloader = get_dataloader(DATA_PATH, CSV_FILE, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # ✅ Initialize Predictor
    #predictor = Predictor(model_paths=MODEL_PATHS)
    predictor = Predictor(model_paths=MODEL_PATHS, log_to_wandb=LOG_TO_WANDB)

    # ✅ Select prediction function dynamically
    #model_variant = predictor.configs[0].get("model_variant")#, "baseline")
    model_variant = predictor.models[0][1]
    print(f"Model variant: {model_variant}")

    # ✅ Limit dataset if in debug mode
    if DEBUG_MODE:
        limited_data = []
        for i, batch in enumerate(dataloader):
            if i >= 100:  # Process only 20 batches
                break
            limited_data.append(batch)
        dataloader = limited_data  # Replace full dataloader with limited data

        print(f"🛠 DEBUG MODE: Processing only {len(dataloader)} batches!")

    """
    if model_variant == "quantile_regression":
        predictions = predictor.predict_quantiles(dataloader)
    elif model_variant == "gaussian_loss":
        predictions = predictor.predict_gaussian(dataloader)
    #elif len(predictor.models) > 1:  # If more than one model → Ensemble
    elif model_variant == "ensemble":
        predictions = predictor.predict_ensemble(dataloader)
    else:
        predictions = predictor.predict_baseline(dataloader)  # Default = baseline
    """
    # ✅ Run prediction
    predictions = predictor.predict(dataloader, model_variant)

    # ✅ Add debug print
    if DEBUG_MODE: print(f"DEBUG: Type of predictions after prediction = {type(predictions)}")

    # ✅ Ensure predictions is a dict before saving
    """
    if isinstance(predictions, dict):
        predictor.save_predictions(predictions)
    else:
        raise TypeError(f"Expected 'predictions' to be a dict, but got {type(predictions)} instead.")
    """
    # ✅ Save results
    #predictor.save_predictions(predictions)
    #predictor.save_predictions(predictions, model_variant)
    """
    predictor.save_predictions(
        predictions=predictions,  # Explicitly passing predictions
        model_variant=model_variant,  # Explicitly passing model variant
        save_json=True,  # Set to True if you want JSON output
        save_csv=True,  # Set to True if you want CSV output
        save_npz=False  # Set to True if you want NPZ output
    )
    """


    # ✅ Generate visualizations
    if DEBUG_MODE: print(f"DEBUG: Type of predictions before visualization = {type(predictions)}")
    #redictor.visualize_results(predictions, dataloader)
    predictor.visualize_results(predictions, dataloader, model_variant)

    print("🎉 Prediction pipeline completed successfully!")
