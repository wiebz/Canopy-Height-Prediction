import argparse
import getpass
import os
import shutil
import socket
import sys
import tempfile
import warnings
from contextlib import contextmanager

# Add the project's root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import wandb

from training.runner import Runner
from training.utilities import GeneralUtility

# Set logging levels for external libraries
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('wandb').setLevel(logging.WARNING)

warnings.filterwarnings('ignore')

"""
# Parse command-line arguments
parser = argparse.ArgumentParser(description="Training script for canopy height prediction.")
parser.add_argument("--debug", action="store_true", help="Run in debug mode.")
parser.add_argument("--model_save_dir", type=str, default="./models", help="Directory to save trained models.")
args = parser.parse_args()

debug = args.debug
model_save_dir = args.model_save_dir

# Ensure the model save directory exists
os.makedirs(model_save_dir, exist_ok=True)
"""

# Configuration parameters
debug = False  # Set this manually to enable/disable debug mode
MODEL_SAVE_DIR = "./models"  # Change this as needed

# Ensure the model save directory exists
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


defaults = dict(
    # Model save directory
    model_save_dir=MODEL_SAVE_DIR,

    # System
    seed=1,

    # Data
    dataset='',#'ai4forest_debug', 'sentinel_pauls_paper'
    batch_size=5,

    # Model variant specifics
    ensemble_size=3,
    loss_name='l2',#'shift_huber',  # Defaults to shift_l1

    # Architecture
    arch='unet',  # Defaults to unet
    backbone='resnet50',  # Defaults to resnet50
    use_pretrained_model=False,

    # Optimization
    optim='AdamW',  # Defaults to AdamW
    n_iterations=100, #100
    log_freq=5,
    initial_lr=1e-3,
    weight_decay=1e-2,
    use_standardization=False,
    use_augmentation=False,
    use_label_rescaling=False,

    # Efficiency
    fp16=False,
    use_memmap=False,
    num_workers_per_gpu=8,   # Defaults to 8

    # Other
    use_weighted_sampler='g10',
    use_weighting_quantile=10,
    use_swa=False,
    use_mixup=False,
    use_grad_clipping=False,
    use_input_clipping=False,   # Must be in [False, None, 1, 2, 5]
    n_lr_cycles=0,
    cyclic_mode='triangular2',
    )
"""
if not debug:
    # Set everything to None recursively
    defaults = GeneralUtility.fill_dict_with_none(defaults)
"""
if not debug:
    # Only set `None` for missing keys, NOT overwrite everything
    for key in defaults:
        if defaults[key] is None:
            defaults[key] = None  # Only change missing keys

# Add the hostname to the defaults
defaults['computer'] = socket.gethostname()

print("Defaults before wandb.init:", defaults)

# Configure wandb logging
wandb.init(
    config=defaults,
    project='test-000',  # automatically changed in sweep
    entity=None,  # automatically changed in sweep
)

print("Wandb config after init:", dict(wandb.config))  # Check what wandb is storing

config = wandb.config
config = GeneralUtility.update_config_with_default(config, defaults)

print("Config after update:", dict(config))  # Final check

# Ensure None values are replaced with actual defaults
for key, value in defaults.items():
    if getattr(config, key, None) is None:
        setattr(config, key, value)

# Debug output
print(f"Final config after update: {dict(config)}")


@contextmanager
def tempdir():
    username = getpass.getuser()
    tmp_root = '/scratch/local/' + username
    tmp_path = os.path.join(tmp_root, 'tmp')
    if os.path.isdir('/scratch/local/') and not os.path.isdir(tmp_root):
        os.mkdir(tmp_root)
    if os.path.isdir(tmp_root):
        if not os.path.isdir(tmp_path): os.mkdir(tmp_path)
        path = tempfile.mkdtemp(dir=tmp_path)
    else:
        assert 'htc-' not in os.uname().nodename, "Not allowed to write to /tmp on htc- machines."
        path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        try:
            shutil.rmtree(path)
            sys.stdout.write(f"Removed temporary directory {path}.\n")
        except IOError:
            sys.stderr.write('Failed to clean up temp dir {}'.format(path))


with tempdir() as tmp_dir:
    # Check if we are running on the GCP cluster, if so, mark as potentially preempted
    is_htc = 'htc-' in os.uname().nodename
    is_gcp = 'gpu' in os.uname().nodename and not is_htc
    if is_gcp:
        print('Running on GCP, marking as preemptable.')
        wandb.mark_preempting()  # Note: This potentially overwrites the config when a run is resumed -> problems with tmp_dir

    runner = Runner(config=config, tmp_dir=tmp_dir, debug=debug)
    runner.run()

    # Save the trained ensemble models
    for idx, model_path in enumerate(runner.model_paths['ensemble']):
        permanent_path = os.path.join(config.get('model_save_dir', './models'), f'ensemble_model_{idx}.pt')
        shutil.copy(model_path, permanent_path)
        print(f"Saved ensemble model {idx+1} to {permanent_path}")

    # Close wandb run
    wandb_dir_path = wandb.run.dir
    wandb.join()

    # Delete the local files
    if os.path.exists(wandb_dir_path):
        shutil.rmtree(wandb_dir_path)
