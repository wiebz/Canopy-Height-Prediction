import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
import wandb
from torch.cuda.amp import autocast, GradScaler
from runner import Runner  # Ensures dataset consistency
from config import PreprocessedSatelliteDataset
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ----------------------------
# 🔹 Step 1: Initialize WandB & Load Config
# ----------------------------

wandb.init(project="fine_tune_gaussian", resume="allow")
config = wandb.config  # Load config dynamically

# ✅ Extract key parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASELINE_MODEL_PATH = os.path.join(config.model_save_dir, "baseline_model.pt")
GAUSSIAN_MODEL_PATH = os.path.join(config.model_save_dir, "gaussian_finetuned.pt")
NUM_EPOCHS = config.get("num_finetune_epochs", 10)
LEARNING_RATE = config.get("finetune_lr", 1e-4)

print(f"✅ Using DEVICE: {DEVICE}")
print(f"✅ Loading Baseline Model from: {BASELINE_MODEL_PATH}")
print(f"✅ Fine-tuning for {NUM_EPOCHS} epochs with LR={LEARNING_RATE}\n")

# Track total fine-tuning time
fine_tune_start_time = time.time()


# ----------------------------
# 🔹 Step 2: Load the Pretrained Baseline Model
# ----------------------------

checkpoint = torch.load(BASELINE_MODEL_PATH, map_location=DEVICE)
baseline_state_dict = checkpoint["state_dict"]
baseline_config = checkpoint["config"]

if checkpoint["model_variant"] != "baseline":
    raise ValueError(f"❌ ERROR: Loaded model is not a baseline model! Found: {checkpoint['model_variant']}")

baseline_model = smp.Unet(
    encoder_name=baseline_config["backbone"],  
    encoder_weights=None,  
    in_channels=baseline_config["in_channels"],  
    classes=baseline_config["num_classes"]
)
baseline_model.load_state_dict(baseline_state_dict)
baseline_model.to(DEVICE)
baseline_model.eval()

print("✅ Baseline Model Loaded Successfully!\n")


# ----------------------------
# 🔹 Step 3: Modify the Final Layer for Gaussian Outputs
# ----------------------------

previous_out_channels = baseline_model.segmentation_head[0].in_channels  

baseline_model.segmentation_head = nn.Sequential(
    nn.Conv2d(in_channels=previous_out_channels, out_channels=2, kernel_size=1)
)
gaussian_model = baseline_model.to(DEVICE)

print("✅ Model Updated to Output Mean & Variance!\n")


# ----------------------------
# 🔹 Step 4: Freeze All Layers Except the Last One
# ----------------------------

for param in gaussian_model.parameters():
    param.requires_grad = False  

for param in gaussian_model.segmentation_head.parameters():
    param.requires_grad = True

print("✅ Only the final layer is trainable!\n")


# ----------------------------
# 🔹 Step 5: Prepare for Training
# ----------------------------

def gaussian_nll_loss(pred, target):
    mean, log_var = pred[:, 0, :, :], pred[:, 1, :, :]  
    variance = torch.exp(log_var)  
    loss = 0.5 * torch.mean((target - mean) ** 2 / variance + log_var)
    return loss

print("🔄 Loading Training & Validation Data...")
train_loader, val_loader, _ = Runner.get_dataloaders()  

optimizer = optim.AdamW(gaussian_model.segmentation_head.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
scaler = torch.amp.GradScaler('cuda')

print("✅ Training Setup Completed!\n")


# ----------------------------
# 🔹 Step 6: Training Loop with Time Logging & Validation
# ----------------------------

best_val_loss = float("inf")

print("🚀 Starting Fine-Tuning...")
for epoch in range(NUM_EPOCHS):
    gaussian_model.train()
    epoch_loss = 0
    start_time = time.time()

    for x_input, y_target in train_loader:
        x_input, y_target = x_input.to(DEVICE), y_target.to(DEVICE)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            output = gaussian_model(x_input)  
            loss = gaussian_nll_loss(output, y_target)  

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(gaussian_model.segmentation_head.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

    epoch_duration = time.time() - start_time
    avg_loss = epoch_loss / len(train_loader)

    print(f"📌 Epoch {epoch + 1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f} - Time: {epoch_duration:.2f}s")


    # ----------------------------
    # 🔹 Step 7: Validation & Metrics Logging
    # ----------------------------

    gaussian_model.eval()
    val_loss, mse, mae = 0, 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x_input, y_target in val_loader:
            x_input, y_target = x_input.to(DEVICE), y_target.to(DEVICE)

            with autocast():
                output = gaussian_model(x_input)  
                loss = gaussian_nll_loss(output, y_target)  

            val_loss += loss.item()

            # Extract predictions & targets
            mean_preds = output[:, 0, :, :].cpu().numpy()
            true_targets = y_target.cpu().numpy()
            all_preds.append(mean_preds)
            all_targets.append(true_targets)

            # Compute Metrics
            mse += np.mean((mean_preds - true_targets) ** 2)
            mae += np.mean(np.abs(mean_preds - true_targets))

    val_loss /= len(val_loader)
    mse /= len(val_loader)
    mae /= len(val_loader)
    rmse = np.sqrt(mse)

    print(f"🔎 Validation Results - Loss: {val_loss:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # Log to WandB
    wandb.log({
        "Epoch": epoch + 1,
        "Train Loss": avg_loss,
        "Validation Loss": val_loss,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "Epoch Duration": epoch_duration,
    })

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "state_dict": gaussian_model.state_dict(),
            "config": baseline_config,
            "model_variant": "gaussian"
        }, GAUSSIAN_MODEL_PATH)
        print(f"✅ Best Model Saved! (Validation Loss: {val_loss:.4f})")


# ----------------------------
# 🔹 Step 8: Generate Validation Visualizations
# ----------------------------

# ✅ Function to plot input-output comparison
def plot_input_output(pred, target, filename):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(pred, cmap="viridis")
    axes[0].set_title("Predicted Mean Height")

    axes[1].imshow(target, cmap="viridis")
    axes[1].set_title("Ground Truth Height")

    plt.savefig(filename)
    plt.close()

# ✅ Function to create density scatter plot (like in FixVal visualizations)
def plot_density_scatter(preds, targets, filename):
    plt.figure(figsize=(6, 6))
    sns.kdeplot(x=targets.flatten(), y=preds.flatten(), fill=True, cmap="Blues", thresh=0.05)
    plt.plot([0, 60], [0, 60], "r--", label="x = y")  # Reference line
    plt.xlabel("Ground Truth Height (m)")
    plt.ylabel("Predicted Height (m)")
    plt.title("Density Scatter Plot")
    plt.legend()
    plt.savefig(filename)
    plt.close()

# ✅ Function to create boxplot of error distribution
def plot_boxplot(preds, targets, filename):
    errors = preds.flatten() - targets.flatten()
    plt.figure(figsize=(8, 5))
    sns.boxplot(y=errors)
    plt.axhline(0, color="red", linestyle="--")  # Reference line
    plt.ylabel("Prediction Error (m)")
    plt.title("Error Distribution")
    plt.savefig(filename)
    plt.close()

# ✅ Generate & Log Visualizations for Validation Set
print("\n📊 Generating Validation Visualizations...")
all_preds = np.concatenate(all_preds, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

for i in range(min(3, all_preds.shape[0])):
    plot_input_output(all_preds[i], all_targets[i], f"val_sample_{i}.png")
    wandb.log({f"Validation Sample {i}": wandb.Image(f"val_sample_{i}.png")})

plot_density_scatter(all_preds, all_targets, "val_density_scatter.png")
wandb.log({"Density Scatter Plot": wandb.Image("val_density_scatter.png")})

plot_boxplot(all_preds, all_targets, "val_error_boxplot.png")
wandb.log({"Error Boxplot": wandb.Image("val_error_boxplot.png")})

print("✅ Validation Visualizations Logged to WandB!")


# ----------------------------
# 🔹 Step 9: Save Final Metrics & Model
# ----------------------------

fine_tune_total_time = time.time() - fine_tune_start_time
wandb.log({"Total Fine-Tune Time": fine_tune_total_time})

print(f"\n✅ Fine-Tuning Complete! Total Time: {fine_tune_total_time:.2f}s")
print(f"✅ Final Model Saved at {GAUSSIAN_MODEL_PATH}")

wandb.finish()
