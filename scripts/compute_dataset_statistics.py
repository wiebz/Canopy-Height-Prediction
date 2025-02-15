import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import sys
import os
import numpy as np

# Add the project's root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from training.config import PreprocessedSatelliteDataset
from training.runner import Runner

from tqdm.auto import tqdm

def compute_mean_std(dataset, split):

    #rootPath = Runner.get_dataset_root(dataset_name=dataset)
    #rootPath = '/Users/wiebkezink/Documents/Uni Münster/MA/dataset'
    splitPath = '/home/ubuntu/work/saved_data/Global-Canopy-Height-Map'
    rootPath = '/home/ubuntu/work/satellite_data/sentinel_pauls_paper/samples'

    print(f"Resolved dataset path: {rootPath}") # Debugging

    if split == 'train':
        dataframe = os.path.join(splitPath, 'train.csv')
    elif split == 'val':
        dataframe = os.path.join(splitPath, 'val.csv')
    else:
        raise ValueError("Invalid split value. Expected 'train' or 'val'.")
    
    print(f"Looking for dataframe at: {dataframe}") # Debugging

    # Convert to tensor (this changes the order of the channels)
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = PreprocessedSatelliteDataset(data_path=rootPath, dataframe=dataframe, image_transforms=train_transforms,
                                           use_weighted_sampler=None, use_memmap=True)
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers_default = 4
    num_workers = num_workers_default * torch.cuda.device_count()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    mean = 0.
    std = 0.
    nb_samples = 0.
    with torch.no_grad():
        for data in tqdm(dataloader):
            data, _ = data
            data = data.to(device=device, non_blocking=True)
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std

# Load the dataset
dataset = 'sentinel_pauls_paper'
split = 'train'


# Compute and print the mean and std
mean, std = compute_mean_std(dataset=dataset, split=split)
print(f'Mean: {mean}')
print(f'Std: {std}')


# Define the results directory
results_dir = './results'
os.makedirs(results_dir, exist_ok=True)

# Define the dump path
dump_path = os.path.join(results_dir, f'{dataset}_{split}_mean_std.txt')

print(f"Saving statistics to: {dump_path}")


# Dump the mean and std to a file in the current working directory
#dump_path = os.path.join(os.getcwd(), f'{dataset}_{split}_mean_std.txt') # keine Schreibrechte im scripts folder
with open(dump_path, 'w') as f:
    f.write(f'Mean: {mean}\n')
    f.write(f'Std: {std}\n')