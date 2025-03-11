import torch

checkpoint_path = "models/prediction/gaussian_model.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

print("🔍 Checkpoint Keys:", checkpoint.keys())  # Should contain 'state_dict' and 'config'

if "state_dict" in checkpoint:
    print("🔍 State Dict Keys:", list(checkpoint["state_dict"].keys())[:])  # Print first 20 keys
else:
    print("❌ ERROR: 'state_dict' missing from checkpoint!")

if "config" in checkpoint:
    print("🔍 Model Config:", checkpoint["config"])  # Ensure the config is there
else:
    print("❌ ERROR: 'config' missing from checkpoint!")
