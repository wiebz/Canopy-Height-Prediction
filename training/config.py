from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import glob
import os
import torch
import numpy as np
import pandas as pd
import pdb
from torch.utils.data.dataloader import default_collate
import sys

means = {
    'satellite_data': (7350.2964, 8265.4316, 5197.9922, 4661.1250,  743.4734, 1063.2332,
        1328.1122, 1657.2146, 2194.5916, 2415.3865, 2473.2197, 2572.5078,
        2590.8245, 2032.7953),
}

stds = {
    'satellite_data': (847.7974, 897.7203, 928.9338, 874.8210, 176.5845, 211.8331, 279.8626,
        279.1518, 334.1505, 371.3719, 395.3250, 387.0693, 370.0272, 339.1393),
}

percentiles = {
    'satellite_data': {
        1: (-8956.0, -8706.0, -13676.0, -13721.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        2: (-7930.0, -7629.0, -12630.0, -12847.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        5: (-6502.0, -6262.0, -11293.0, -11611.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        95: (26317.0, 25280.0, 21480.0, 22595.0, 15959.0, 16047.0, 15919.0, 15811.0, 15795.0, 15767.0, 15751.0, 15705.0, 12344.0, 13236.0),
        98: (27685.0, 26830.0, 23002.0, 24265.0, 16231.0, 16143.0, 15945.0, 15891.0, 15824.0, 15807.0, 15896.0, 15729.0, 13504.0, 14253.0),
        99: (28788.0, 27897.0, 24213.0, 25487.0, 16403.0, 16208.0, 16080.0, 16102.0, 15924.0, 15976.0, 16000.0, 15818.0, 13813.0, 15047.0),
    }
}

class FixValDataset(Dataset):
    """
    Dataset class to load the fixval dataset.
    """
    def __init__(self, data_path, dataframe, image_transforms=None):
        self.data_path = data_path
        self.df = pd.read_csv(dataframe, index_col=False)
        #self.files = list(self.df["path"].apply(lambda x: os.path.join(data_path, x)))
        self.files = list(self.df["path"].apply(lambda x: os.path.join(data_path, '/home/ubuntu/work/satellite_data/sentinel_pauls_paper/samples/', x)))
        self.image_transforms = image_transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index].replace(r"'", "")
        fileName = file[file.rfind('data_')+5: file.rfind('.npz')]
        data = np.load(file)

        image = data["data"].astype(np.float32)
        # Move the channel axis to the last position (required for torchvision transforms)
        image = np.moveaxis(image, 0, -1)
        if self.image_transforms:
            image = self.image_transforms(image)

        return image, fileName

class PreprocessedSatelliteDataset(Dataset):
    """
    Dataset class for preprocessed satellite imagery.
    """

    def __init__(self, data_path, dataframe=None, image_transforms=None, label_transforms=None, joint_transforms=None, use_weighted_sampler=False,
                  use_weighting_quantile=None, use_memmap=False, remove_corrupt=True, load_labels=True, patch_size=512):
        self.use_memmap = use_memmap
        self.patch_size = patch_size
        self.load_labels = load_labels  # If False, we only load the images and not the labels
        df = pd.read_csv(dataframe)

        if remove_corrupt:
            if "has_corrupt_s2_channel_flag" in df.columns:
                old_len = len(df)
                df = df[df["has_corrupt_s2_channel_flag"] == False]
                sys.stdout.write(f"Removed {old_len - len(df)} corrupt rows.\n")
            else:
                sys.stdout.write("Warning: Column 'has_corrupt_s2_channel_flag' not found. Proceeding without filtering corrupt rows.\n")

        #self.files = list(df["paths"].apply(lambda x: os.path.join(data_path, x)))
        self.df = df # neu , warum?
        #self.files = list(df["path"].apply(lambda x: os.path.join(data_path, x)))
        self.files = list(self.df["path"].apply(lambda x: os.path.join(data_path, '/home/ubuntu/work/satellite_data/sentinel_pauls_paper/samples/', x)))


        if use_weighted_sampler not in [False, None]:
            assert use_weighted_sampler in ['g5', 'g10', 'g15', 'g20', 'g25', 'g30']
            weighting_quantile = use_weighting_quantile
            assert weighting_quantile in [None, 'None'] or int(weighting_quantile) == weighting_quantile, "weighting_quantile must be an integer."
            if weighting_quantile in [None, 'None']:
                self.weights = (df[use_weighted_sampler] / df["totals"]).values.clip(0., 1.)
            else:
                # We do not clip between 0 and 1, but rather between the weighting_quantile and 1.
                weighting_quantile = float(weighting_quantile)
                self.weights = (df[use_weighted_sampler] / df["totals"]).values

                # Compute the quantiles, ignoring nan values and zero values
                tmp_weights = self.weights.copy()
                tmp_weights[np.isnan(tmp_weights)] = 0.
                tmp_weights = tmp_weights[tmp_weights > 0.]

                quantile_min = np.nanquantile(tmp_weights, weighting_quantile / 100)
                sys.stdout.write(f"Computed weighting {weighting_quantile}-quantile-lower bound: {quantile_min}.\n")

                # Clip the weights
                self.weights = self.weights.clip(quantile_min, 1.0)

            # Set the nan values to 0.
            self.weights[np.isnan(self.weights)] = 0.

        else:
            self.weights = None
        self.image_transforms, self.label_transforms, self.joint_transforms = image_transforms, label_transforms, joint_transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        if self.use_memmap:
            item = self.getitem_memmap(index)
        else:
            item = self.getitem_classic(index)

        return item

    def getitem_memmap(self, index):
        file = self.files[index]
        with np.load(file, mmap_mode='r') as npz_file:
            image = npz_file['data'].astype(np.float32)
            # Move the channel axis to the last position (required for torchvision transforms)
            image = np.moveaxis(image, 0, -1)
            if self.image_transforms:
                image = self.image_transforms(image)
            if self.load_labels:
                label = npz_file['labels'].astype(np.float32)

                # Process label
                label = label[:3]  # Everything after index/granule 3 is irrelevant
                label = label / 100  # Convert from cm to m
                label = np.moveaxis(label, 0, -1)

                if self.label_transforms:
                    label = self.label_transforms(label)
                if self.joint_transforms:
                    image, label = self.joint_transforms(image, label)
                return image, label

        return image

    def getitem_classic(self, index):
        file = self.files[index]
        data = np.load(file)

        image = data["data"].astype(np.float32)
        # Move the channel axis to the last position (required for torchvision transforms)
        image = np.moveaxis(image, 0, -1)[:self.patch_size,:self.patch_size]
        if self.image_transforms:
            image = self.image_transforms(image)
        if self.load_labels:
            label = data["labels"].astype(np.float32)

            # Process label
            label = label[:3]  # Everything after index 3 is irrelevant
            label = label[:,:self.patch_size, :self.patch_size]
            label = label / 100  # Convert from cm to m
            label = np.moveaxis(label, 0, -1)

            if self.label_transforms:
                label = self.label_transforms(label)
            if self.joint_transforms:
                image, label = self.joint_transforms(image, label)
            return image, label

        return image
