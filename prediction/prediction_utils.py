import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from config import DATA_PATH, CSV_FILE


class FixValDataset(Dataset):
    """Dataset class for FixVal dataset."""
    def __init__(self, data_path, dataframe, image_transforms=None):
        self.data_path = data_path
        self.df = pd.read_csv(os.path.join(data_path,dataframe))
        self.short_filenames = self.df["path"].apply(lambda x: os.path.basename(x).replace("data_", "").replace(".npz", "")).tolist()
        self.files = list(self.df["path"].apply(lambda x: os.path.join(data_path, '/home/ubuntu/work/satellite_data/sentinel_pauls_paper/samples/', x)))
        self.image_transforms = image_transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index].replace("'", "")
        fileName = file[file.rfind('data_') + 5: file.rfind('.npz')]
        data = np.load(file)

        image = data["data"].astype(np.float32)
        image = np.moveaxis(image, 0, -1)

        if self.image_transforms:
            image = self.image_transforms(image)

        return image, fileName


def get_dataloader(data_path, csv_file, batch_size=2, num_workers=4):
    """Returns a DataLoader for the FixValDataset."""
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = FixValDataset(data_path=data_path, dataframe=csv_file, image_transforms=transform)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
