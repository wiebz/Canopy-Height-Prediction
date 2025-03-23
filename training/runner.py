import os
import platform
import shutil
import sys
import time
from collections import OrderedDict
from typing import Optional, Any
from collections import defaultdict

import numpy as np
import segmentation_models_pytorch as smp
import torch
import wandb
from torch.cuda.amp import autocast
from torch.utils.data import WeightedRandomSampler
from torchmetrics import MeanMetric
from torchvision.transforms import transforms
from tqdm.auto import tqdm
"""
from training import visualization
from training.config import PreprocessedSatelliteDataset, FixValDataset
from training.config import means as meanDict
from training.config import percentiles as percentileDict
from training.config import stds as stdDict
from training.metrics import MetricsClass
from training.utilities import JointRandomRotationTransform
from training.utilities import SequentialSchedulers
"""
import visualization
from config import PreprocessedSatelliteDataset, FixValDataset
from config import means as meanDict
from config import percentiles as percentileDict
from config import stds as stdDict
from metrics import MetricsClass
from utilities import JointRandomRotationTransform
from utilities import SequentialSchedulers



class Runner:
    """Base class for all runners, defines the general functions"""

    def __init__(self, config: Any, tmp_dir: str, debug: bool):
        """
        Initialize useful variables using config.
        :param config: wandb run config
        :type config: wandb.config.Config
        :param debug: Whether we are in debug mode or not
        :type debug: bool
        """
        self.config = config
        self.debug = debug

        n_gpus = torch.cuda.device_count()
        if n_gpus > 0:
            config.update(dict(device='cuda:0'))
        else:
            config.update(dict(device='cpu'))

        self.dataParallel = (torch.cuda.device_count() > 1)
        if not self.dataParallel:
            self.device = torch.device(config.device)
            if 'gpu' in config.device:
                torch.cuda.set_device(self.device)
        else:
            # Use all visible GPUs
            self.device = torch.device("cuda:0")
            torch.cuda.device(self.device)
        torch.backends.cudnn.benchmark = True

        # Variant setting
        self.model_variant = self.config.get('model_variant', 'baseline')
        
        # Ensemble settings
        if self.model_variant == 'ensemble':
            self.ensemble_size = self.config.get('ensemble_size', 5)  # Für Ensemble
        if self.model_variant == 'quantiles':
            self.quantiles = self.config.get('quantiles', [0.1, 0.5, 0.9])

        # Set a couple useful variables
        self.seed = int(self.config.seed)
        self.loss_name = self.config.loss_name or 'l2' # vorher: shift_l1
        sys.stdout.write(f"Using loss: {self.loss_name}.\n")
        self.use_amp = self.config.fp16
        self.tmp_dir = tmp_dir
        print(f"Using temporary directory {self.tmp_dir}.")
        self.label_rescaling_factor = 1.
        if self.config.use_label_rescaling:
            self.label_rescaling_factor = 60.
            sys.stdout.write(f"Using label rescaling of {self.label_rescaling_factor} - this is hardcoded.\n")

        # Variables to be set
        self.loader = {loader_type: None for loader_type in ['train', 'val']}
        self.loss_criteria = {loss_name: self.get_loss(loss_name=loss_name) for loss_name in ['shift_l1', 'shift_l2', 'shift_huber', 'l1', 'l2', 'huber', 'gaussian_loss', 'pinball_loss']} #'l1', 'l2'
        for threshold in [15, 20, 25, 30]:
            self.loss_criteria[f"l1_{threshold}"] = self.get_loss(loss_name=f"l1", threshold=threshold)

        self.optimizer = None
        self.scheduler = None
        self.model = None
        self.artifact = None
        self.model_paths = {model_type: None for model_type in ['initial', 'trained']}
        self.model_metrics = {  # Organized way of saving metrics needed for retraining etc.
        }

        self.metrics = {mode: {'loss': MeanMetric().to(device=self.device),
                               'shift_l1': MeanMetric().to(device=self.device),
                               'shift_l2': MeanMetric().to(device=self.device),
                               'shift_huber': MeanMetric().to(device=self.device),
                               'l1': MeanMetric().to(device=self.device),
                               'l2': MeanMetric().to(device=self.device),
                               'huber': MeanMetric().to(device=self.device),
                               'gaussian_loss': MeanMetric().to(device=self.device),
                               'pinball_loss': MeanMetric().to(device=self.device),
                               }
                        for mode in ['train', 'val']}

        for mode in ['train', 'val']:
            for threshold in [15, 20, 25, 30]:
                self.metrics[mode][f"l1_{threshold}"] = MeanMetric().to(device=self.device)

        self.metrics['train']['ips_throughput'] = MeanMetric().to(device=self.device)

        # ✅ Initialize loss history dictionary to store intermediate metrics
        self.loss_history = {}


    @staticmethod
    def set_seed(seed: int):
        """
        Sets the seed for the current run.
        :param seed: seed to be used
        """
        # Set a unique random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Remark: If you are working with a multi-GPU model, this function is insufficient to get determinism. To seed all GPUs, use manual_seed_all().
        torch.cuda.manual_seed(seed)  # This works if CUDA not available

        print(f"Using seed: {seed}")  # Log the seed
 

    def reset_averaged_metrics(self):
        """Resets all metrics"""
        for mode in self.metrics.keys():
            for metric in self.metrics[mode].values():
                metric.reset()

    def get_metrics(self) -> dict:
        """
        Returns the metrics for the current epoch.
        :return: dict containing the metrics
        :rtype: dict
        """
        with torch.no_grad():
            loggingDict = dict(
                # Model metrics
                n_params=MetricsClass.get_parameter_count(model=self.model),

                # Optimizer metrics
                learning_rate=float(self.optimizer.param_groups[0]['lr']),
            )
            # Add metrics
            for split in ['train', 'val']:
                for metric_name, metric in self.metrics[split].items():
                    try:
                        # Catch case where MeanMetric mode not set yet
                        loggingDict[f"{split}/{metric_name}"] = metric.compute()
                    except Exception as e:
                        continue

        return loggingDict

    @staticmethod
    def get_dataset_root(dataset_name: str) -> str:
        """Copies the dataset and returns the rootpath."""
        # Determine where the data lies
        for root in ['/home/ubuntu/work/saved_data/Global-Canopy-Height-Map/']:#'/home/ubuntu/work/satellite_data/sentinel_pauls_paper/', '/home/htc/mzimmer/SCRATCH/', './datasets_pytorch/', '/home/jovyan/work/scratch/']:  # SCRATCHAIS2T, local, scratch_jan
            rootPath = f"{root}{dataset_name}"
            if os.path.isdir(rootPath):
                break
        
        return rootPath

    def get_dataloaders(self):
        rootPath = self.get_dataset_root(dataset_name=self.config.dataset)
        print(f"Loading {self.config.dataset} dataset from {rootPath}.")

        data_path = rootPath
        train_dataframe = os.path.join(rootPath, 'train.csv')
        val_dataframe = os.path.join(rootPath, 'val.csv')
        fix_val_dataframe = os.path.join(rootPath, 'fix_val.csv')

        transformDict = {split: None for split in ['train', 'val']}
        base_transform = transforms.ToTensor()
        transforms_list = [base_transform]   # Convert to tensor (this changes the order of the channels)
        """
        if self.config.use_standardization:
            assert self.config.use_input_clipping not in [False, None, 'None'], "Mutually exclusive options: use_standardization and use_input_clipping."
            assert self.config.dataset in meanDict.keys(), f"Mean of Dataset {self.config.dataset} not implemented."
            assert self.config.dataset in stdDict.keys(), f"Std of Dataset {self.config.dataset} not implemented."
            mean, std = meanDict[self.config.dataset], stdDict[self.config.dataset]
            normalize_transform = transforms.Normalize(mean=mean, std=std)
            transforms_list.append(normalize_transform)
        elif self.config.use_input_clipping not in [False, None, 'None']:
            assert not self.config.use_standardization, "Mutually exclusive options: use_standardization and use_input_clipping."
            use_input_clipping = int(self.config.use_input_clipping)
            assert use_input_clipping in [1, 2, 5], "use_input_clipping must be in [False, None, 1, 2, 5]."
            sys.stdout.write(f"Using input clipping of in the {use_input_clipping}-{100-use_input_clipping}-range.\n")
            input_clipping_lower_bound = percentileDict[self.config.dataset][use_input_clipping]
            input_clipping_upper_bound = percentileDict[self.config.dataset][100 - use_input_clipping]

            # Convert the bounds to tensors
            input_clipping_lower_bound = torch.tensor(input_clipping_lower_bound, dtype=torch.float).view(-1, 1, 1) # View to make it a 3D tensor
            input_clipping_upper_bound = torch.tensor(input_clipping_upper_bound, dtype=torch.float).view(-1, 1, 1) # View to make it a 3D tensor

            # Define the clipping transform over the channels, i.e. each channel is clipped individually using the bounds from percentileDict
            clipping_transform = transforms.Lambda(lambda x: torch.clamp(x, min=input_clipping_lower_bound, max=input_clipping_upper_bound))
            transforms_list.append(clipping_transform)
        """


        transformDict['train'] = transforms.Compose(transforms_list)
        transformDict['val'] = transforms.Compose(transforms_list)

        # Create the label transform to rescale the labels
        label_transforms = transforms.Compose([base_transform, lambda x: x * (1./self.label_rescaling_factor)])

        joint_transforms = None # Train transforms that are both applied to the image and the label
        if self.config.use_augmentation:
            sys.stdout.write(f"Using JointRandomRotationTransform.\n")
            joint_transforms = JointRandomRotationTransform()

        use_weighted_sampler = self.config.use_weighted_sampler or False

        remove_corrupt = not self.debug
        trainData = PreprocessedSatelliteDataset(data_path=data_path, dataframe=train_dataframe, image_transforms=transformDict['train'], label_transforms=label_transforms, joint_transforms=joint_transforms,
                                                 use_weighted_sampler=use_weighted_sampler, use_weighting_quantile=self.config.use_weighting_quantile, use_memmap=self.config.use_memmap, remove_corrupt=remove_corrupt)
        valData = PreprocessedSatelliteDataset(data_path=data_path, dataframe=val_dataframe,
                                               image_transforms=transformDict['val'], label_transforms=label_transforms, use_memmap=self.config.use_memmap, remove_corrupt=remove_corrupt)
        fixvalData = FixValDataset(data_path=data_path, dataframe=fix_val_dataframe,
                                               image_transforms=transformDict['val'])

        sys.stdout.write(f"Length of train and val splits: {len(trainData)}, {len(valData)}.\n")
        cut_off = 3000
        if len(valData) >= cut_off:
            sys.stdout.write(f"Validation dataset is large, reducing to a maximum of {cut_off} samples.\n")
            # Reduce the size of the validation dataset using self.seed as the random seed
            # Perform a random split using a generator with self.seed as the seed
            valData, _ = torch.utils.data.random_split(valData, [cut_off, len(valData) - cut_off], generator=torch.Generator().manual_seed(self.seed))
        sys.stdout.write(f"New length of train and val splits: {len(trainData)}, {len(valData)}.\n")


        num_workers_default = self.config.num_workers_per_gpu if self.config.num_workers_per_gpu is not None else 8
        num_workers = num_workers_default * torch.cuda.device_count() * int(not self.debug)
        sys.stdout.write(f"Using {num_workers} workers.\n")
        train_sampler = None
        shuffle = True
        if use_weighted_sampler:
            train_sampler = WeightedRandomSampler(trainData.weights, len(trainData))
            shuffle = False

        trainLoader = torch.utils.data.DataLoader(trainData, batch_size=self.config.batch_size, sampler=train_sampler,
                                                  pin_memory=torch.cuda.is_available(), num_workers=num_workers,
                                                  shuffle=shuffle)
        valLoader = torch.utils.data.DataLoader(valData, batch_size=self.config.batch_size, shuffle=False,
                                                pin_memory=torch.cuda.is_available(), num_workers=num_workers)
        fixvalLoader = torch.utils.data.DataLoader(fixvalData, batch_size=2, shuffle=False, # This only works with batch_size=2
                                                pin_memory=torch.cuda.is_available(), num_workers=num_workers, drop_last=True) # ADDED: drop_last=True

        return trainLoader, valLoader, fixvalLoader

    def get_model(self, reinit: bool, model_path: Optional[str] = None) -> torch.nn.Module:
        """
        Returns the model.
        :param reinit: If True, the model is reinitialized.
        :type reinit: bool
        :param model_path: Path to the model.
        :type model_path: Optional[str]
        :return: The model.
        :rtype: torch.nn.Module
        """
        print(
            f"Loading model - reinit: {reinit} | path: {model_path if model_path else 'None specified'}.")
        if reinit:
            # Define the model
            arch = self.config.arch or 'unet'
            backbone = self.config.backbone or 'resnet50'

            if self.model_variant == "gaussian":
                num_classes = 2  # Gaussian output: 2 channels #
            elif self.model_variant == 'quantiles':
                num_classes = len(self.quantiles)
            else:
                num_classes = 1

            assert isinstance(num_classes, int) and num_classes > 0, "Error: num_classes must be a positive integer!"                
            sys.stdout.write(f"Using architecture {arch} with {num_classes} output channels.\n")

            assert arch in ['unet', 'unetpp', 'manet', 'linknet', 'fpn', 'pspnet', 'pan', 'deeplabv3', 'deeplabv3p']
            network_config = {
                "encoder_name": backbone,
                "encoder_weights": None if not self.config.use_pretrained_model else 'imagenet',
                "in_channels": 14,
                "classes": num_classes,
            }
            if arch == 'unet':
                model = smp.Unet(**network_config)
            elif arch == 'unetpp':
                model = smp.UnetPlusPlus(**network_config)
            elif arch == 'manet':
                model = smp.MAnet(**network_config)
            elif arch == 'linknet':
                model = smp.Linknet(**network_config)
            elif arch == 'fpn':
                model = smp.FPN(**network_config)
            elif arch == 'pspnet':
                model = smp.PSPNet(**network_config)
            elif arch == 'pan':
                model = smp.PAN(**network_config)
            elif arch == 'deeplabv3':
                model = smp.DeepLabV3(**network_config)
            elif arch == 'deeplabv3p':
                model = smp.DeepLabV3Plus(**network_config)
        else:
            # The model has been initialized already
            model = self.model

        if model_path is not None:
            # Load the model
            state_dict = torch.load(model_path, map_location=self.device)

            new_state_dict = OrderedDict()
            require_DP_format = isinstance(model,
                                           torch.nn.DataParallel)  # If true, ensure all keys start with "module."
            for k, v in state_dict.items():
                is_in_DP_format = k.startswith("module.")
                if require_DP_format and is_in_DP_format:
                    name = k
                elif require_DP_format and not is_in_DP_format:
                    name = "module." + k  # Add 'module' prefix
                elif not require_DP_format and is_in_DP_format:
                    name = k[7:]  # Remove 'module.'
                elif not require_DP_format and not is_in_DP_format:
                    name = k
                new_state_dict[name] = v
            # Load the state_dict
            model.load_state_dict(new_state_dict)

        if self.dataParallel and reinit and not isinstance(model, torch.nn.DataParallel):
            # Only apply DataParallel when re-initializing the model!
            # We use DataParallelism
            model = torch.nn.DataParallel(model)
        model = model.to(device=self.device)
        return model

    def get_loss(self, loss_name: str, threshold: float = None):
        assert loss_name in ['shift_l1', 'shift_l2', 'shift_huber', 'l1', 'l2', 'huber', 'gaussian_loss', 'pinball_loss'], f"Loss {loss_name} not implemented." #'l1', 'l2'
        if threshold is not None:
            assert loss_name == 'l1', f"Threshold only implemented for l1 loss, not {loss_name}."
        # Dim 1 is the channel dimension, 0 is batch.
        # Sums up to get average height, could be mean without zeros
        remove_sub_track = lambda out, target: (out, torch.sum(target, dim=1))
        """
        if loss_name == 'l1':
            from losses.l1_loss import L1Loss
            # Rescale the threshold to account for the label rescaling
            if threshold is not None:
                threshold = threshold / self.label_rescaling_factor
            loss = L1Loss(ignore_value=0, pre_calculation_function=remove_sub_track, lower_threshold=threshold)
        elif loss_name == 'l2':
            from losses.l2_loss import L2Loss
            loss = L2Loss(ignore_value=0, pre_calculation_function=remove_sub_track)

        """
        if loss_name == 'shift_l1':
            from losses.shift_l1_loss import ShiftL1Loss
            loss = ShiftL1Loss(ignore_value=0)
        elif loss_name == 'shift_l2':
            from losses.shift_l2_loss import ShiftL2Loss
            loss = ShiftL2Loss(ignore_value=0)
        elif loss_name == 'shift_huber':
            from losses.shift_huber_loss import ShiftHuberLoss
            loss = ShiftHuberLoss(ignore_value=0)
        elif loss_name == 'l1':
            from losses.l1_loss import L1Loss
            # Rescale the threshold to account for the label rescaling
            if threshold is not None:
                threshold = threshold / self.label_rescaling_factor
            loss = L1Loss(ignore_value=0, pre_calculation_function=remove_sub_track, lower_threshold=threshold)
        elif loss_name == 'l2':
            from losses.l2_loss import L2Loss
            loss = L2Loss(ignore_value=0, pre_calculation_function=remove_sub_track)
        elif loss_name == 'huber':
            from losses.huber_loss import HuberLoss
            loss = HuberLoss(ignore_value=0, pre_calculation_function=remove_sub_track, delta=3.0)
        elif loss_name == 'gaussian_loss':
            from losses.gaussian_loss import GaussianLoss
            loss = GaussianLoss(ignore_value=0, pre_calculation_function=remove_sub_track)
        elif loss_name == 'pinball_loss':
            from losses.pinball_loss import PinballLoss
            quantiles = self.config.get("quantiles", [0.1, 0.5, 0.9])  # Default: [0.1, 0.5, 0.9]
            loss = PinballLoss(quantiles=quantiles, ignore_value=0, pre_calculation_function=remove_sub_track)
        
        loss = loss.to(device=self.device)
        return loss

    def get_visualization(self, viz_name: str, inputs, labels, outputs):
        assert viz_name in ['input_output', 'density_scatter_plot',
                            'boxplot'], f"Visualization {viz_name} not implemented."
        
        # 🛠 Ensure `outputs` is always 4D (B, C, H, W)
        if outputs.dim() == 3:  # If shape is (B, H, W), add missing channel dimension
            outputs = outputs.unsqueeze(1)  # (B, 1, H, W)

        # 🛠 Extract correct outputs for Gaussian model
        if self.model_variant == "gaussian":
            outputs = outputs[:, 0, :, :]  # Extract only mean predictions → (B, H, W)
            outputs = outputs.unsqueeze(1)  # Ensure it is always (B, 1, H, W)

        # 🛠 Extract correct outputs for Quantile Regression model
        elif self.model_variant == "quantiles":
            num_quantiles = outputs.shape[1]  # Get number of quantiles (e.g., 3 for τ = [0.1, 0.5, 0.9])
            median_index = num_quantiles // 2  # Index for the median quantile (τ = 0.5)
            outputs = outputs[:, median_index, :, :]  # Extract median prediction (B, H, W)
            outputs = outputs.unsqueeze(1)  # Ensure it is always (B, 1, H, W)


        labels = labels.detach().clone()
        

        # Detach and copy the labels and outputs, then undo the rescaling
        #labels, outputs = labels.detach().clone(), outputs.detach().clone()

        # Undo the rescaling
        labels, outputs = labels * self.label_rescaling_factor, outputs * self.label_rescaling_factor
        #print(f"DEBUG: outputs shape after rescaling in get_viz(): {outputs.shape}")

        def remove_sub_track_vis(inputs, labels, outputs):
            return inputs, labels.sum(
                axis=1), outputs  # Same as remove_sub_track, but for visualization (i.e. has outputs as well)
        
        #print(f"DEBUG: outputs shape before passed to viz functions: {outputs.shape}")

        if viz_name == 'input_output':
            viz_fn = visualization.get_input_output_visualization(rgb_channels=[6, 5, 4],
                                                                  process_variables=remove_sub_track_vis)
        elif viz_name == 'density_scatter_plot':
            viz_fn = visualization.get_density_scatter_plot_visualization(ignore_value=0,
                                                                          process_variables=remove_sub_track_vis)
        elif viz_name == 'boxplot':
            viz_fn = visualization.get_visualization_boxplots(ignore_value=0, process_variables=remove_sub_track_vis)
        return viz_fn(inputs=inputs, labels=labels, outputs=outputs)

    def get_optimizer(self, initial_lr: float) -> torch.optim.Optimizer:
        """
        Returns the optimizer.
        :param initial_lr: The initial learning rate
        :type initial_lr: float
        :return: The optimizer.
        :rtype: torch.optim.Optimizer
        """
        wd = self.config['weight_decay'] or 0.
        optim_name = self.config.optim or 'AdamW'
        if optim_name == 'SGD':
            optimizer = torch.optim.SGD(params=self.model.parameters(), lr=initial_lr,
                                        momentum=0.9,
                                        weight_decay=wd,
                                        nesterov=wd > 0.)
        elif optim_name == 'AdamW':
            optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=initial_lr,
                                          weight_decay=wd)
        else:
            raise NotImplementedError

        return optimizer

    def save_model(self, model_identifier: str, sync: bool = False) -> str:
        """
        Saves the model's state_dict to a file.
        """
        fName = f"{model_identifier}.pt"
        fPath = os.path.join(self.tmp_dir, fName)

        try:
            # ✅ Get model state dict
            model_state_dict = self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()

            # ✅ Convert config to a dictionary safely
            config_dict = vars(self.config) if hasattr(self.config, '__dict__') else dict(self.config)

            # ✅ Add number of output classes (segmentation head)
            config_dict["num_classes"] = self.model.segmentation_head._modules["0"].out_channels if hasattr(self.model, "segmentation_head") else 1
            #num_classes = self.model.segmentation_head._modules["0"].out_channels if hasattr(self.model, "segmentation_head") else 1
            #print(f"config_dict num_classes: {config_dict["num_classes"]}")

            # ✅ Save model with metadata
            model_data = {
                "state_dict": model_state_dict,
                "config": config_dict,
                "model_variant": self.model_variant,
                #"num_classes": num_classes,
                "seed": self.seed
            }

            #print(f"model data: {model_data["num_classes"]}")
            #print(f"num_classes: {model_data.get('num_classes', 'N/A')}")
            print(f"fPath before saving: {fPath}")

            # print("\n🔍 Model architecture before saving:")
            # print(self.model)
            # print("\n🔍 Model state_dict keys before saving:", model_state_dict.keys())


            torch.save(model_data, fPath)  # ✅ Save full model data

            # ✅ Sync to WandB if needed
            if sync:
                wandb.save(fPath)

                # Ensure config update only when necessary
                for key, value in config_dict.items():
                    if key not in wandb.config:  
                        wandb.config[key] = value  # Update WandB config safely


            print(f"✅ Model successfully saved to: {fPath}")

            return fPath
        
        except Exception as e:
            print(f"❌ Model saving failed: {e}")
            return None



    def old_save_model(self, model_identifier: str, sync: bool = False) -> str:
        """
        Saves the model's state_dict to a file.
        :param model_identifier: Name of the file type.
        :type model_identifier: str
        :param sync: Whether to sync the file to wandb.
        :type sync: bool
        :return: Path to the saved model.
        :rtype: str
        """
        fName = f"{model_identifier}.pt"
        fPath = os.path.join(self.tmp_dir, fName)

        try:
            # Get model state dict
            model_state_dict = self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()

            # Convert config to a dictionary safely
            #config_dict = dict(self.config) if isinstance(self.config, dict) else self.config.to_dict()
            config_dict = vars(self.config) if hasattr(self.config, '__dict__') else dict(self.config)

            # Save model with metadata
            model_data = {
                "state_dict": model_state_dict,
                "config": config_dict,  # Save config inside the file!
                "model_variant": self.model_variant,
                "seed": self.seed
            }

            torch.save(model_data, fPath)  # Save full model data

            # ✅ Log config explicitly in WandB if syncing
            if sync:
                wandb.save(fPath)  # Save model file to WandB
                
                # Ensure config update only when necessary
                for key, value in config_dict.items():
                    if key not in wandb.config:  
                        wandb.config[key] = value  # Update WandB config safely


            print(f"✅ Model successfully saved to: {fPath}")
            
            return fPath
        
        except Exception as e:
            print(f"Model saving failed: {e}")
            return None
        

    def log(self, step: int, phase_runtime: float):
        """
        Logs the current training status.
        :param phase_runtime: The wall-clock time of the current phase.
        :type phase_runtime: float
        """
        loggingDict = self.get_metrics()
        loggingDict.update({
            'phase_runtime': phase_runtime,
            'iteration': step,
            'samples_seen': step * self.config.batch_size,
        })

        # Log and push to Wandb
        for metric_type, val in loggingDict.items():
            wandb.run.summary[f"{metric_type}"] = val

        if wandb.run is not None:
            wandb.log(loggingDict)
            # wandb.join()
        else:
            print("⚠️ Warning: WandB run is not initialized! Skipping logging.")
        
        #wandb.log(loggingDict)

    def define_optimizer_scheduler(self):
        # Define the optimizer
        initial_lr = self.config.initial_lr
        self.optimizer = self.get_optimizer(initial_lr=initial_lr)

        # We define a scheduler. All schedulers work on a per-iteration basis
        n_total_iterations = self.config.n_iterations
        n_lr_cycles = self.config.n_lr_cycles or 0
        cyclic_mode = self.config.cyclic_mode or 'triangular2'
        n_warmup_iterations = int(0.1 * n_total_iterations) if n_lr_cycles == 0 else 0

        # Set the initial learning rate
        for param_group in self.optimizer.param_groups: param_group['lr'] = initial_lr

        # Define the warmup scheduler if needed
        warmup_scheduler, milestone = None, None
        if n_warmup_iterations > 0:
            # As a start factor we use 1e-20, to avoid division by zero when putting 0.
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self.optimizer,
                                                                 start_factor=1e-20, end_factor=1.,
                                                                 total_iters=n_warmup_iterations)
            milestone = n_warmup_iterations + 1

        n_remaining_iterations = n_total_iterations - n_warmup_iterations
        if n_lr_cycles > 0:
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=self.optimizer, base_lr=0.,
                                                            max_lr=initial_lr, step_size_up=n_remaining_iterations // (2*n_lr_cycles),
                                                            mode=cyclic_mode,
                                                            cycle_momentum=False)
        else:
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self.optimizer,
                                                      start_factor=1.0, end_factor=0.,
                                                      total_iters=n_remaining_iterations)

        # Reset base lrs to make this work
        scheduler.base_lrs = [initial_lr if warmup_scheduler else 0. for _ in self.optimizer.param_groups]

        # Define the Sequential Scheduler
        if warmup_scheduler is None:
            self.scheduler = scheduler
        else:
            self.scheduler = SequentialSchedulers(optimizer=self.optimizer, schedulers=[warmup_scheduler, scheduler],
                                                  milestones=[milestone])

    @torch.no_grad()
    def eval(self, data: str):
        """
        Evaluates the model on the given data set.
        :param data: string indicating the data set to evaluate on. Can be 'train' or 'val'.
        :type data: str
        """
        sys.stdout.write(f"Evaluating on {data} split.\n")
        for step, (x_input, y_target) in enumerate(tqdm(self.loader[data]), 1):
                
            x_input = x_input.to(self.device, non_blocking=True)
            y_target = y_target.to(self.device, non_blocking=True)


            # with autocast(enabled=self.use_amp): old
            with torch.amp.autocast('cuda',enabled=self.use_amp):
                output = self.model.eval()(x_input)

                # DEBUG: ✅ Clip outputs to avoid extreme values
                # output = torch.clamp(output, min=-1000, max=1000)
                # Check if the model is producing NaNs
                # if torch.isnan(output).any():
                #     print(f"⚠️ NaN detected in model output at step {step}")

                # ✅ Compute the selected training loss
                loss = self.loss_criteria[self.loss_name](output, y_target)
                if torch.isnan(loss).any():
                    sys.stdout.write(f"⚠️ Warning: {self.loss_name} loss is NaN in {data}.\n")
                else:
                    self.metrics[data]['loss'](value=loss, weight=len(y_target))


                # ✅ Handle different model variants
                if self.model_variant == "gaussian":
                    mean_output = output[:, 0, :, :]  # Extract mean for standard loss
                elif self.model_variant == "quantiles":
                    median_index = len(self.config.quantiles) // 2  # Find the middle quantile (e.g., q50)
                    mean_output = output[:, median_index, :, :]  # Extract median quantile
                else:
                    mean_output = output  # Baseline & ensemble models use full output


                for loss_type in self.loss_criteria.keys():
                    # Skip Gaussian loss for non-Gaussian models
                    if loss_type == "gaussian_loss":
                        if self.model_variant != "gaussian":
                            continue
                        metric_loss = self.loss_criteria["gaussian_loss"](output, y_target)

                    # Skip Pinball loss for non-Quantile models
                    elif loss_type == "pinball_loss":
                        if self.model_variant != "quantiles":
                            continue
                        metric_loss = self.loss_criteria["pinball_loss"](output, y_target)

                    # Compute all other losses (l1, l2, huber, etc.)
                    else:
                        metric_loss = self.loss_criteria[loss_type](mean_output, y_target)

                    # Ensure we don't log NaN values
                    if not torch.isnan(metric_loss):
                        self.metrics[data][loss_type](value=metric_loss, weight=len(y_target))

                """
                # ✅ Compute losses
                for loss_type in self.loss_criteria.keys():
                    if loss_type == "gaussian_loss" and self.model_variant == "gaussian":
                        metric_loss = self.loss_criteria["gaussian_loss"](output, y_target)  # Use full output
                    elif loss_type == "gaussian_loss" and self.model_variant != "gaussian":
                        continue  # Skip Gaussian loss if model is not Gaussian
                    elif loss_type == "pinball_loss" and self.model_variant == "quantiles":
                        metric_loss = self.loss_criteria["pinball_loss"](output, y_target)  # Use full output
                    elif loss_type == "pinball_loss" and self.model_variant != "quantiles":
                        continue  # Skip Pinball loss if model is not Quantile
                    elif loss_type not in ["gaussian_loss", "pinball_loss"]:
                        metric_loss = self.loss_criteria[loss_type](mean_output, y_target)  # Use only median quantile or mean


                    # Check for NaNs
                    #if metric_loss is not None and not torch.isnan(metric_loss):
                    if not torch.isnan(metric_loss):
                        self.metrics[data][loss_type](value=metric_loss, weight=len(y_target))
                """

            #print(f"DEBUG: output shape after loss calc: {mean_output.shape}")
            # ✅ Ensure shape is always (B, 1, H, W) for visualization
            viz_output = mean_output.unsqueeze(1) if mean_output.dim() == 3 else mean_output
            #print(f"DEBUG: output shape before visualization: {viz_output.shape}")

            if step <= 4:
                # Create the visualizations for the first 4 batches
                for viz_func in ['input_output', 'density_scatter_plot', 'boxplot']:
                    viz = self.get_visualization(viz_name=viz_func, inputs=x_input, labels=y_target, outputs=viz_output)
                    wandb.log({data + '/' + viz_func + "_" + str(step): wandb.Image(viz)}, commit=False)
                    # wandb.join()


    @torch.no_grad()
    def eval_fixval(self):
        """Creates the fixval plots and logs them to wandb."""
        sys.stdout.write(f"Creating fixval plots.\n")

        def remove_sub_track_vis_wout_labels(inputs, labels, outputs):
            # 🛠 Handle Gaussian Model: Extract only the mean prediction
            if outputs.dim() == 4 and outputs.shape[1] == 2:  # Gaussian model (B, 2, H, W)
                outputs = outputs[:, 0, :, :]  # Extract mean prediction → (B, H, W)

            # 🛠 Handle Quantile Regression Model: Extract the median quantile prediction
            elif outputs.dim() == 4 and self.model_variant == "quantiles":
                num_quantiles = outputs.shape[1]  # Get number of quantiles (e.g., 3 for τ = [0.1, 0.5, 0.9])
                median_index = num_quantiles // 2  # Index for the median quantile (τ = 0.5)
                outputs = outputs[:, median_index, :, :]  # Extract median prediction → (B, H, W)

            # ✅ Ensure outputs always have shape (B, 1, H, W)
            if outputs.dim() == 3:  # If shape is (B, H, W), add channel dimension
                outputs = outputs.unsqueeze(1)  # (B, 1, H, W)

            #print(f"DEBUG: output shape for eval_fixval: {outputs.shape}")

            return inputs, None, outputs

        viz_fn = visualization.get_input_output_visualization(rgb_channels=[6, 5, 4],
                                                            process_variables=remove_sub_track_vis_wout_labels)

        loggingDict = dict()
        for x_input, fileNames in tqdm(self.loader['fix_val']):
            x_input = x_input.to(self.device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                output = self.model.eval()(x_input)

            # ✅ Handle different model variants
            if self.model_variant == "gaussian":
                mean_output = output[:, 0, :, :]  # Extract mean
            elif self.model_variant == "quantiles":
                median_index = len(self.config.quantiles) // 2  # Find median quantile (q50)
                mean_output = output[:, median_index, :, :]
            else:
                mean_output = output  # Standard output

            # ✅ Ensure correct shape for visualization
            viz_output = mean_output.unsqueeze(1) if mean_output.dim() == 3 else mean_output

            # ✅ Generate FixVal visualization
            viz = viz_fn(inputs=x_input, labels=None, outputs=viz_output)
            jointName = "__".join(fileNames)
            wandb.log({'fixval' + '/' + 'input_output' + '/' + jointName: wandb.Image(viz)}, commit=False)
            # wandb.join()

            # ✅ Log min/max values
            flattened_output = viz_output.flatten(start_dim=1)
            min_values = flattened_output.min(dim=1).values
            max_values = flattened_output.max(dim=1).values


            for i, (min_val, max_val) in enumerate(zip(min_values, max_values)):
                # Log the min and max values as numeric values with the appropriate filename
                name = fileNames[i]
                loggingDict[f"fixval/min_{name}"] = min_val.item()
                loggingDict[f"fixval/max_{name}"] = max_val.item()

        if wandb.run is not None:
            wandb.log(loggingDict, commit=False)
            # wandb.join()
        else:
            print("⚠️ Warning: WandB run is not initialized! Skipping logging.")



    def train_ensemble(self):
        # self.model_paths['ensemble'] = []  # make sure the list is empty
        # ensemble = []
        self.model_paths['ensemble'] = {}

        for i in range(self.ensemble_size):
            seed = i
            self.set_seed(seed=seed)  # Set the seed

            # ✅ Create a new config dictionary with the updated seed
            config_dict = dict(self.config)
            config_dict["seed"] = seed  # Update seed value safely
            #self.config.update({"seed": seed}, allow_val_change=True)
            
            sys.stdout.write(f"Training model {i+1}/{self.ensemble_size} with seed {seed}\n")

            # Start a new WandB run for each ensemble member
            wandb.init(
                #config=dict(self.config),
                config=config_dict,
                project='test-002',
                name=f'ensemble_model_{i}',
                entity=None,
                reinit=True,
                resume="allow"
            )

            if hasattr(self, "model") and self.model is not None:
                del self.model  # Free previous model
                torch.cuda.empty_cache()  # Clear unused memory

            # Reinitialize and train the ensemble member
            self.model = self.get_model(reinit=True)
            self.train()

            # if wandb.run:
                #wandb.finish()
                # wandb.join()


        """ *** NOT NEEDED ANYMORE ***
            # Save the trained model
            model_path = self.save_model(f'ensemble_model_{self.config.n_iterations}_{i}', sync=True)  # Save model and push to WandB
            ensemble.append(model_path)

            # Finish WandB run to ensure separation
            if wandb.run is not None:
                wandb.finish()
            #wandb.finish()
    
        self.model_paths['ensemble'] = ensemble


        # Save the trained ensemble models
        trained_model_dir = self.config.get('model_save_dir', './models/trained/')
        os.makedirs(trained_model_dir, exist_ok=True)

        for idx, model_path in enumerate(self.model_paths['ensemble']):
                save_path = os.path.join(trained_model_dir, f'ensemble_model_{idx}.pt')
                shutil.copy(model_path, save_path)
                print(f"✅ Saved ensemble model {idx+1} to {save_path}")
        """


    def train(self):
        log_freq, n_iterations = self.config.log_freq, self.config.n_iterations
        ampGradScaler = torch.amp.GradScaler('cuda',enabled=self.use_amp)

        self.reset_averaged_metrics()
        phase_start = time.time()
        intermediate_runtime_start = time.time()

        # Define the Stochastic Weight Averaging model
        swa_model = None
        if self.config.use_swa:
            swa_model = torch.optim.swa_utils.AveragedModel(self.model)
            swa_start = max(1, int(0.75 * n_iterations))
            sys.stdout.write(f"SWA model will be updated from iteration {swa_start} onwards.\n")

        # Define the distribution of mixup
        if self.config.use_mixup:
            alpha = 0.2
            beta_distribution = torch.distributions.beta.Beta(alpha, alpha)
            sys.stdout.write(f"Using mixup with alpha={alpha}. Example sample: {beta_distribution.sample()}\n")

        for step in tqdm(range(1, n_iterations + 1, 1), desc="Training progress"):
            # debug
            #print(f"Training iteration: {step}/{n_iterations}")

            # Reinitialize the train iterator if it reaches the end
            if step == 1 or (step - 1) % len(self.loader['train']) == 0:
                train_iterator = iter(self.loader['train'])

            # Move to CUDA if possible
            x_input, y_target = next(train_iterator)
            x_input = x_input.to(device=self.device, non_blocking=True)
            y_target = y_target.to(device=self.device, non_blocking=True)

            if self.config.use_mixup:
                # Sample from the beta distribution with alpha=0.2
                with torch.no_grad():
                    lam = beta_distribution.sample().to(self.device)
                    index = torch.randperm(x_input.size(0)).to(self.device)
                    x_input = lam * x_input + (1 - lam) * x_input[index, :]
                    y_target = lam * y_target + (1 - lam) * y_target[index, :]

            self.optimizer.zero_grad()

            itStartTime = time.time()
            with torch.amp.autocast('cuda',enabled=self.use_amp):
                output = self.model.train()(x_input)

                # print(f"DEBUG: output shape: {output.shape}")

                # ✅ Handle different model variants
                if self.model_variant == "gaussian":
                    mean_output = output[:, 0, :, :]  # Extract mean prediction
                elif self.model_variant == "quantiles":
                    median_index = len(self.config.quantiles) // 2  # Find median quantile (q50)
                    mean_output = output[:, median_index, :, :]
                else:
                    mean_output = output  # Standard for baseline/ensemble

                # ✅ Compute correct loss
                if self.model_variant == "gaussian":
                    loss = self.loss_criteria["gaussian_loss"](output, y_target)  # Full output for Gaussian loss
                elif self.model_variant == "quantiles":
                    loss = self.loss_criteria["pinball_loss"](output, y_target)  # Full output for quantile loss
                else:
                    loss = self.loss_criteria[self.loss_name](mean_output, y_target)  # Other losses use mean or median


                #loss = self.loss_criteria[self.loss_name](output, y_target)

                ampGradScaler.scale(loss).backward()  # Scaling + Backpropagation
                # Unscale the weights manually, normally this would be done by ampGradScaler.step(), but since
                # we might use gradient clipping, this has to be split
                ampGradScaler.unscale_(self.optimizer)
                if self.config.use_grad_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                ampGradScaler.step(self.optimizer)
                ampGradScaler.update()  # This should happen only once per iteration
                self.scheduler.step()
                self.metrics['train']['loss'](value=loss, weight=len(y_target))

                # ✅ Compute other loss metrics
                with torch.no_grad():
                    for loss_type in self.loss_criteria.keys():
                        # Skip Gaussian loss for non-Gaussian models
                        if loss_type == "gaussian_loss":
                            if self.model_variant != "gaussian":
                                continue
                            metric_loss = self.loss_criteria["gaussian_loss"](output, y_target)

                        # Skip Pinball loss for non-Quantile models
                        elif loss_type == "pinball_loss":
                            if self.model_variant != "quantiles":
                                continue
                            metric_loss = self.loss_criteria["pinball_loss"](output, y_target)

                        # Compute all other losses (l1, l2, huber, etc.)
                        else:
                            metric_loss = self.loss_criteria[loss_type](mean_output, y_target)

                        # Ensure we don't log NaN values
                        if not torch.isnan(metric_loss):
                            self.metrics['train'][loss_type](value=metric_loss, weight=len(y_target))

                    """
                    for loss_type in self.loss_criteria.keys():
                        if loss_type == "gaussian_loss" and self.model_variant == "gaussian":
                            metric_loss = self.loss_criteria["gaussian_loss"](output, y_target)
                        elif loss_type == "gaussian_loss" and self.model_variant != "gaussian":
                            continue  # Skip Gaussian loss for non-Gaussian models
                        elif loss_type == "pinball_loss" and self.model_variant == "quantiles":
                            metric_loss = self.loss_criteria["pinball_loss"](output, y_target)
                        elif loss_type == "pinball_loss" and self.model_variant != "quantiles":
                            continue  # Skip Pinball loss for non-Quantile models
                        elif loss_type not in ["gaussian_loss", "pinball_loss"]:
                            metric_loss = self.loss_criteria[loss_type](mean_output, y_target)

                        if not torch.isnan(metric_loss):
                            self.metrics['train'][loss_type](value=metric_loss, weight=len(y_target))
                        """

                            
                itEndTime = time.time()
                n_img_in_iteration = int(self.config.batch_size)
                ips = n_img_in_iteration / (itEndTime - itStartTime)  # Images processed per second
                self.metrics['train']['ips_throughput'](ips)

            if swa_model is not None:
                if step >= swa_start:
                    swa_model.update_parameters(self.model)
                    if step == n_iterations:
                        sys.stdout.write(f"Last step reached. Setting model to SWA model.\n")
                        # Set the model to the SWA model by copying the parameters
                        with torch.no_grad():
                            for param1, param2 in zip(self.model.parameters(), swa_model.parameters()):
                                param1.copy_(param2)
                            swa_model = None

            if step % log_freq == 0 or step == n_iterations:
                phase_runtime = time.time() - phase_start

                #print(f"DEBUG: output shape in train() before reshaping: {output.shape}")

                if self.model_variant == "gaussian":
                    output_viz = output[:, 0, :, :]  # Extract only the mean prediction (Shape: B, H, W)
                    output_viz = output_viz.unsqueeze(1)  # Add missing channel dimension -> (B, 1, H, W)

                elif self.model_variant == "quantiles":
                    num_quantiles = output.shape[1]  # Extract number of quantiles (e.g., 3 for τ = [0.1, 0.5, 0.9])
                    median_index = num_quantiles // 2  # Index for the median quantile (τ = 0.5)
                    output_viz = output[:, median_index, :, :]  # Extract the median prediction (Shape: B, H, W)
                    output_viz = output_viz.unsqueeze(1)  # Add missing channel dimension -> (B, 1, H, W)

                else:
                    output_viz = output  # Use unchanged output for baseline & ensemble models

                # ✅ Ensure outputs always have 4 dimensions
                if output_viz.dim() == 3:  # If shape is (B, H, W), add channel dimension
                    output_viz = output_viz.unsqueeze(1)  # (B, 1, H, W)

                #print(f"DEBUG: output shape in train() after reshaping: {output_viz.shape}")

                
                # Create the visualizations
                for viz_func in ['input_output', 'density_scatter_plot', 'boxplot']:
                    viz = self.get_visualization(viz_name=viz_func, inputs=x_input, labels=y_target, outputs=output_viz)
                    wandb.log({'train/' + viz_func: wandb.Image(viz)}, commit=False)
                    #  wandb.join()

                # Evaluate the validation dataset
                if not self.debug:
                    self.eval(data='val')

                # Create the fixval plots
                """
                if not self.debug:
                    self.eval_fixval()
                """

                self.log(step=step, phase_runtime=phase_runtime)
                if step in range(5000, n_iterations+1, 5000):
                #if step in {10000, 25000, 40000, 50000, 60000, 65000, n_iterations}:
                    intermediate_runtime = time.time() - intermediate_runtime_start

                    print("📊 Calculate metrics for intermediate saving \n")
                    metricsDict = self.get_metrics()
                    metricsDict.update({
                        'phase_runtime': intermediate_runtime,
                        'iteration': step,
                        })

                    for metric_type, value in metricsDict.items():
                        print(f"{metric_type}: {value}")


                    # Save the trained ensemble models
                    trained_model_dir = self.config.get('model_save_dir', './models/trained/')
                    os.makedirs(trained_model_dir, exist_ok=True)

                    if self.model_variant == 'ensemble':
                        model_path = self.save_model(f'ensemble_model_{step}_{self.config.seed}', sync=True)  # Save model and push to WandB
                        self.model_paths.setdefault(f'{self.model_variant}', defaultdict(dict))[f'{step}'][self.config.seed] = model_path

                        save_path = os.path.join(trained_model_dir, f'ensemble_model_{step//1000}k_{self.config.seed}.pt')
                        
                    else:
                        model_path = self.save_model(f'{self.model_variant}', sync=True)  # Save model and push to WandB
                        self.model_paths.setdefault(f'{self.model_variant}', defaultdict(dict))[f'{step}'] = model_path

                        save_path = os.path.join(trained_model_dir, f'{self.model_variant}_model_{step//1000}k.pt')
                        
                    shutil.copy(model_path, save_path)

                self.reset_averaged_metrics()
                phase_start = time.time()

            """
            if step in {10000, 25000, 50000, n_iterations}:

                # Save the trained ensemble models
                trained_model_dir = self.config.get('model_save_dir', './models/trained/')
                os.makedirs(trained_model_dir, exist_ok=True)

                if self.model_variant == 'ensemble':
                    model_path = self.save_model(f'ensemble_model_{step}_{self.config.seed}', sync=True)  # Save model and push to WandB
                    #ensemble.append(model_path)
                    # self.model_paths[f'{self.model_variant}'][f'{step//1000}k'][self.config.seed] = model_path
                    #self.model_paths.setdefault(f'{self.model_variant}', {}).setdefault(f'{step//1000}k', {})[self.config.seed] = model_path
                    self.model_paths.setdefault(f'{self.model_variant}', defaultdict(dict))[f'{step}'][self.config.seed] = model_path

                    #for idx, model_path in enumerate(self.model_paths['ensemble']):
                    save_path = os.path.join(trained_model_dir, f'ensemble_model_{step//1000}k_{self.config.seed}.pt')
                    # shutil.copy(model_path, save_path)
                    # print(f"✅ Saved ensemble model {self.config.seed + 1} with {step} iterations to {save_path}")


                else:
                    # Save the trained model and upload it to wandb
                    model_path = self.save_model(f'{self.model_variant}', sync=True)  # Speichere das Modell
                    # self.model_paths[f'{self.model_variant}'][f'{step//1000}k'] = model_path
                    #self.model_paths.setdefault(f'{self.model_variant}', {})[f'{step//1000}k'] = model_path
                    self.model_paths.setdefault(f'{self.model_variant}', defaultdict(dict))[f'{step}'] = model_path

                    save_path = os.path.join(trained_model_dir, f'{self.model_variant}_model_{step//1000}k.pt')
                    
                shutil.copy(model_path, save_path)
                print(f"✅ Saved {self.model_variant} model with {step} iterations to {save_path}")
                
                """

                # if wandb.run:
                #     wandb_dir_path = wandb.run.dir
                #     wandb.join()


    def run(self):
        """Controls the execution of the script."""
        # We start training from scratch
        if not self.model_variant == 'ensemble': 
            self.set_seed(seed=self.seed)  # Set the seed
        loaders = self.get_dataloaders()
        self.loader['train'], self.loader['val'], self.loader['fix_val'] = loaders
        self.model = self.get_model(reinit=True, model_path=self.model_paths['initial'])  # Load the model

        self.define_optimizer_scheduler()  # This was moved before define_strategy to have the optimizer available

        if self.model_variant == 'ensemble':
            self.train_ensemble()  # Train the ensemble model

        else:
            self.train()

        # Close wandb run
            wandb_dir_path = wandb.run.dir
            wandb.join()

            # Delete the local files
            if os.path.exists(wandb_dir_path):
                shutil.rmtree(wandb_dir_path)

        """" *** NOT NEEDED ANYMORE ***
            # Save the trained model and upload it to wandb
            model_path = self.save_model(f'{self.model_variant}', sync=True)  # Speichere das Modell
            self.model_paths[f'{self.model_variant}'] = model_path

        # Save the trained ensemble models
        trained_model_dir = self.config.get('model_save_dir', './models/trained/')
        os.makedirs(trained_model_dir, exist_ok=True)

        if self.model_variant == 'ensemble':
            for idx, model_path in enumerate(self.model_paths['ensemble']):
                save_path = os.path.join(trained_model_dir, f'ensemble_model_{idx}.pt')
                shutil.copy(model_path, save_path)
                print(f"✅ Saved ensemble model {idx+1} to {save_path}")
        
        else:
            model_variant = self.config.model_variant
            model_path = self.model_paths[f'{model_variant}']
            save_path = os.path.join(trained_model_dir, f'{model_variant}_model.pt')
            shutil.copy(model_path, save_path)
            print(f"✅ Saved {model_variant} model to {save_path}")

            # Close wandb run
            wandb_dir_path = wandb.run.dir
            wandb.join()

            # Delete the local files
            if os.path.exists(wandb_dir_path):
                shutil.rmtree(wandb_dir_path)
        """