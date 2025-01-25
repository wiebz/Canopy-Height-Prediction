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
    'dataset': (7299.1479, 6796.8154, 4115.0771, 4183.9517,  366.5581,  547.1146,
         479.1360,  892.2798, 1931.0896, 2284.8613, 2331.2368, 2503.7361,
        1584.9974,  927.7599),    # Not the true values, change for your dataset
}

stds = {
    'dataset': (774.4051, 676.9658, 698.3646, 789.2250, 101.2454, 124.4731, 140.2093,
        155.5658, 309.7619, 368.0429, 397.1309, 387.4051, 239.8134, 194.8254),  # Not the true values, change for your dataset
}

percentiles = {
    'dataset': {
        1: (-7542.0, -8126.0, -16659.0, -14187.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        2: (-6834.0, -7255.0, -14468.0, -13537.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        5: (-5694.0, -5963.0, -12383.0, -12601.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        95: (24995.0, 24556.0, 22124.0, 20120.0, 15016.0, 15116.0, 15212.0, 15181.0, 14946.0, 14406.0, 14660.0, 13810.0, 12082.0, 13041.0),
        98: (25969.0, 26078.0, 23632.0, 21934.0, 15648.0, 15608.0, 15487.0, 15449.0, 15296.0, 15155.0, 15264.0, 14943.0, 13171.0, 14064.0),
        99: (27044.0, 27349.0, 24868.0, 23266.0, 15970.0, 15680.0, 15548.0, 15494.0, 15432.0, 15368.0, 15385.0, 15219.0, 13590.0, 14657.0),
    }  # Not the true values, change for your dataset
}

"""
means = {
    'ai4forest_camera': (10782.3223,  3304.7444,  1999.6086,  7276.4209,  1186.4460,  1884.6165,
         2645.6113,  3128.2588,  3806.2808,  4134.6855,  4113.4883,  4259.1885,
         4683.5879,  3838.2222),    # Not the true values, change for your dataset
}

stds = {
    'ai4forest_camera': (907.7484,  472.1412,  423.8558, 1086.0916,  175.0936,  226.6303,
         299.4834,  313.0911,  388.1186,  434.4579,  455.7314,  455.0303,
         388.5127,  374.1260),  # Not the true values, change for your dataset
}

percentiles = {
    'ai4forest_camera': {
        1: (-7542.0, -8126.0, -16659.0, -14187.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        2: (-6834.0, -7255.0, -14468.0, -13537.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        5: (-5694.0, -5963.0, -12383.0, -12601.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        95: (24995.0, 24556.0, 22124.0, 20120.0, 15016.0, 15116.0, 15212.0, 15181.0, 14946.0, 14406.0, 14660.0, 13810.0, 12082.0, 13041.0),
        98: (25969.0, 26078.0, 23632.0, 21934.0, 15648.0, 15608.0, 15487.0, 15449.0, 15296.0, 15155.0, 15264.0, 14943.0, 13171.0, 14064.0),
        99: (27044.0, 27349.0, 24868.0, 23266.0, 15970.0, 15680.0, 15548.0, 15494.0, 15432.0, 15368.0, 15385.0, 15219.0, 13590.0, 14657.0),
    }  # Not the true values, change for your dataset
}
"""

class FixValDataset(Dataset):
    """
    Dataset class to load the fixval dataset.
    """
    def __init__(self, data_path, dataframe, image_transforms=None):
        self.data_path = data_path
        self.df = pd.read_csv(dataframe, index_col=False)
        self.files = list(self.df["path"].apply(lambda x: os.path.join(data_path, x)))
        self.image_transforms = image_transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index].replace(r"'", "")
        fileName = file[file.rfind('data_')+5: file.rfind('.npz')]
        data = np.load(file)

        image = data["data"].astype(np.float32)
        print(f"original image shape: {image.shape}")
        # Move the channel axis to the last position (required for torchvision transforms) -> (H, W, C)
        image = np.moveaxis(image, 0, -1)
        print(f"image shape for transformations: {image.shape}")
        if self.image_transforms:
            image = self.image_transforms(image)
        image = np.moveaxis(image, -1, 0)  # Move channels first for the model -> (C, H, W)
        print(f"image shape after transformations: {image.shape}")

        return image, fileName

class PreprocessedSatelliteDataset(Dataset):
    """
    Dataset class for preprocessed satellite imagery, adaptable for training, validation, and prediction.
    """

    def __init__(
            self, 
            data_path, 
            dataframe=None, 
            image_transforms=None, 
            label_transforms=None, 
            joint_transforms=None, 
            use_weighted_sampler=False,
            use_weighting_quantile=None, 
            use_memmap=False, 
            remove_corrupt=True, 
            load_labels=True, 
            patch_size=512
        ):
        self.use_memmap = use_memmap
        self.patch_size = patch_size
        self.load_labels = load_labels  # If False, we only load the images and not the labels
        self.data_path = data_path # neu , warum?

        # Load the dataframe and remove corrupt data is necessary
        df = pd.read_csv(dataframe)
        """ kann wieder rein, wenn entsprechende spalte im dataset vorhanden
        if remove_corrupt:
            old_len = len(df)
            df = df[df["has_corrupt_s2_channel_flag"] == False]
            sys.stdout.write(f"Removed {old_len - len(df)} corrupt rows.\n")
        """

        if remove_corrupt:
            if "has_corrupt_s2_channel_flag" in df.columns:
                old_len = len(df)
                df = df[df["has_corrupt_s2_channel_flag"] == False]
                sys.stdout.write(f"Removed {old_len - len(df)} corrupt rows.\n")
            else:
                sys.stdout.write("Warning: Column 'has_corrupt_s2_channel_flag' not found. Proceeding without filtering corrupt rows.\n")


        self.df = df # neu , warum?
        self.files = list(df["path"].apply(lambda x: os.path.join(data_path, x)))

        # Handle weighted sampling
        self.weights = None
        if use_weighted_sampler:
            self.weights = self._compute_weights(df, use_weighted_sampler, use_weighting_quantile)

        """ nicht mehr nötig, da in self._compute_weights(df, use_weighted_sampler, use_weighting_quantile)

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
        """
        
        self.image_transforms = image_transforms
        self.label_transforms = label_transforms
        self.joint_transforms = joint_transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        if self.use_memmap:
            item = self._getitem_memmap(index)
        else:
            item = self._getitem_classic(index)

        return item

    def _getitem_memmap(self, index):
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

    def _getitem_classic(self, index):
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


    def _compute_weights(self, df, weighted_sampler_column, weighting_quantile):
        """Compute sample weights for weighted sampling."""
        assert weighted_sampler_column in df.columns, f"Column {weighted_sampler_column} not found in dataframe."
        weights = df[weighted_sampler_column].values

        if weighting_quantile:
            tmp_weights = weights[weights > 0]  # Ignore zeros
            quantile_min = np.nanquantile(tmp_weights, weighting_quantile / 100)
            print(f"Computed {weighting_quantile}-quantile lower bound: {quantile_min}.")
            weights = weights.clip(quantile_min, 1.0)

        weights[np.isnan(weights)] = 0  # Replace NaNs with 0
        return weights
