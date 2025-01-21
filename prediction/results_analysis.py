import torch
import numpy as np
import pandas as pd

# Load the predictions
mean_predictions_path = "./predictions/mean_predictions.pt"
variance_predictions_path = "./predictions/variance_predictions.pt"
all_predictions_path = "./predictions/all_predictions.pt"

mean_predictions = torch.load(mean_predictions_path, weights_only=True)
variance_predictions = torch.load(variance_predictions_path, weights_only=True)
all_predictions = torch.load(all_predictions_path, weights_only=True)

# Extract some example values
variance_sample = variance_predictions.flatten()[:10].tolist()
mean_sample = mean_predictions.flatten()[:10].tolist()
all_sample = all_predictions.flatten()[:10].tolist()

{
    "variance_sample": variance_sample,
    "mean_sample": mean_sample,
    "all_sample": all_sample
}

#print("Variance Predictions Example:", variance_predictions[0])  # Adjust indexing if necessary
#print("Mean Predictions Example:", mean_predictions[0])  # Adjust indexing if necessary
#print("All Predictions Example:", all_predictions[0])  # Adjust indexing if necessary


# Convert tensors to numpy arrays
mean_predictions_np = mean_predictions.numpy()
variance_predictions_np = variance_predictions.numpy()
all_predictions_np = all_predictions.numpy()

# Save mean predictions to CSV
mean_df = pd.DataFrame(mean_predictions_np.reshape(-1))  # Flatten for CSV
mean_df.to_csv("./predictions/mean_predictions.csv", index=False, header=["Mean Predictions"])

# Save variance predictions to CSV
variance_df = pd.DataFrame(variance_predictions_np.reshape(-1))  # Flatten for CSV
variance_df.to_csv("./predictions/variance_predictions.csv", index=False, header=["Variance Predictions"])

# Save all predictions to CSV
all_predictions_flat = all_predictions_np.reshape(all_predictions_np.shape[0], -1)  # Flatten each set of predictions
all_predictions_df = pd.DataFrame(all_predictions_flat)
#all_predictions_df.to_csv("./predictions/all_predictions.csv", index=False, header=[f"Model_{i}" for i in range(all_predictions_np.shape[0])])
all_predictions_df.iloc[:, :10].to_csv(
    "./predictions/all_predictions_sample.csv", 
    index=False, 
    header=[f"Prediction_{i}" for i in range(10)]
)


# Print sample values for quick inspection
print("Sample Variance Predictions:")
print(variance_df.head())

print("\nSample Mean Predictions:")
print(mean_df.head())

print("\nSample All Predictions:")
print(all_predictions_df.head())