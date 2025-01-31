import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split

# Check current working directory
cwd = os.getcwd()
print(f"Current working directory: {cwd}")

# Load CSV file
csv_path = '/home/ubuntu/work/satellite_data/sentinel_pauls_paper/samples.csv'

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"File not found: {csv_path}")

data = pd.read_csv(csv_path)
print("\n✅ First 5 rows of the CSV:")
print(data.head())

# Ensure required columns exist
print("\n✅ Columns in CSV:", data.columns)
assert 'path' in data.columns, "Error: CSV must have a 'path' column."
assert 'latitudes' in data.columns and 'longitudes' in data.columns, "Error: CSV must have 'latitudes' and 'longitudes' columns for geo splitting."

# Prepend base path to 'path' column (only if needed)
base_path = "/home/ubuntu/work/satellite_data/sentinel_pauls_paper/samples/"
if not data['path'].iloc[0].startswith(base_path):  # Prevent duplication
    data['path'] = base_path + data['path']

print("\n✅ Updated file paths:")
print(data['path'].head())


### 🚀 Custom PyTorch Dataset
class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return self.dataframe.iloc[idx]


### 🚀 Function to create a subset (random sampling)
def create_subset(data, subset_size, seed=42):
    """
    Randomly selects a subset of the dataset.

    Parameters:
        data (DataFrame): Original dataset as a pandas DataFrame.
        subset_size (int): Number of elements to sample.
        seed (int): Seed for reproducibility.

    Returns:
        DataFrame: A randomly sampled subset of the dataset.
    """
    return data.sample(n=subset_size, random_state=seed).reset_index(drop=True)


### 🚀 Function to split dataset (supports both fraction and absolute count)
def split_data_torch(dataset, split_ratios=(0.8, 0.1, 0.1), sample_counts=None, generator=None):
    """
    Splits the data into train, val, and fix_val subsets using torch.utils.data.random_split.

    Parameters:
        dataset (Dataset): The PyTorch dataset.
        split_ratios (tuple): Ratios for train, val, and fix_val splits.
        sample_counts (tuple): Specific number of samples for train, val, and fix_val splits.
        generator (torch.Generator): A torch random generator for reproducibility.

    Returns:
        dict: A dictionary containing train, val, and fix_val subsets.
    """
    total_length = len(dataset)

    if sample_counts:
        assert sum(sample_counts) <= total_length, "Error: Sum of sample counts exceeds dataset size!"
        lengths = sample_counts
    else:
        lengths = [int(r * total_length) for r in split_ratios]
        lengths[-1] = total_length - sum(lengths[:-1])  # Adjust last split for rounding errors

    train, val, fix_val = random_split(dataset, lengths, generator=generator)
    return {'train': train, 'val': val, 'fix_val': fix_val}


# Choose between fraction-based or count-based splitting
USE_FRACTION = True  # Set to False for absolute count-based splitting

split_ratios = (0.8, 0.1, 0.1)  # 80% train, 10% val, 10% fix_val
sample_counts = (100, 20, 20)  # Set the exact count for train, val, fix_val

# Create subset (either full dataset or smaller for testing)
SUBSET_SIZE = len(data)  # Change this if you want a smaller subset
data_subset = create_subset(data, SUBSET_SIZE)

# Convert DataFrame into a PyTorch Dataset
dataset = CustomDataset(data_subset)

# Define a random generator for reproducibility
generator = torch.Generator().manual_seed(42)

# Perform the split (use either fraction-based or absolute count-based)
if USE_FRACTION:
    splits = split_data_torch(dataset, split_ratios=split_ratios, generator=generator)
else:
    splits = split_data_torch(dataset, sample_counts=sample_counts, generator=generator)

# Save split datasets to CSV
data_subset.iloc[splits['train'].indices].to_csv('train.csv', index=False)
data_subset.iloc[splits['val'].indices].to_csv('val.csv', index=False)
data_subset.iloc[splits['fix_val'].indices].to_csv('fix_val.csv', index=False)

print("\n✅ Data successfully saved to train.csv, val.csv, and fix_val.csv.")
