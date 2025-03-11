import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from config import DATA_PATH, CSV_FILE


class old_FixValDataset(Dataset):
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




import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from config import DATA_PATH, CSV_FILE

class FixValDataset(Dataset):
    """Dataset class for FixVal dataset, structured like training dataset."""
    
    def __init__(self, data_path, dataframe, image_transforms=None, label_transforms=None, patch_size=512):
        self.data_path = data_path
        self.df = pd.read_csv(os.path.join(data_path, dataframe))
        self.files = list(self.df["path"].apply(lambda x: os.path.join(data_path, '/home/ubuntu/work/satellite_data/sentinel_pauls_paper/samples/', x)))
        self.image_transforms = image_transforms
        self.label_transforms = label_transforms
        self.patch_size = patch_size  # ✅ Ensure patches are correctly sized

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index].replace("'", "")
        fileName = file[file.rfind('data_') + 5: file.rfind('.npz')]
        data = np.load(file)

        # ✅ Load image
        image = data["data"].astype(np.float32)
        image = np.moveaxis(image, 0, -1)  # Channels last
        image = image[:self.patch_size, :self.patch_size]  # Clip to patch size

        # ✅ Load label (ground truth)
        if "labels" in data:
            label = data["labels"].astype(np.float32)  # Match training format
        else:
            raise KeyError(f"❌ ERROR: No 'labels' found in {file}")

        # 🛠 **Apply same processing as training pipeline**
        #label = label[:3]  # ✅ Sum the first 3 granules along axis 0
        #label = label[:,:self.patch_size, :self.patch_size]  # Match patch size
        label = label[:3].sum(axis=0)  # ✅ Sum the first 3 granules along axis 0
        label = label[:self.patch_size, :self.patch_size]  # Match patch size
        label = label / 100  # Convert cm to m
        label = np.moveaxis(label, 0, -1)  # Convert shape to (H, W, 3)

        # ✅ Apply transformations
        if self.image_transforms:
            image = self.image_transforms(image)
        if self.label_transforms:
            label = self.label_transforms(label)

        return image, label, fileName  # ✅ Return per-pixel labels


def get_dataloader(data_path, csv_file, batch_size=8, num_workers=4):
    """Returns a DataLoader for the FixValDataset."""
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = FixValDataset(data_path=data_path, dataframe=csv_file, image_transforms=transform)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True, pin_memory=torch.cuda.is_available())
