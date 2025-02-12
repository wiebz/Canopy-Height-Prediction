
import getpass
import os
import shutil
import socket
import sys
import tempfile
import warnings
from contextlib import contextmanager

import wandb

from runner import Runner
from utilities import GeneralUtility

warnings.filterwarnings('ignore')

debug = "--debug" in sys.argv

defaults = dict(
    # Model save directory
    model_save_dir="./models",

    # System
    seed=1, # bei ensemble rausnehmen!!

    # Data
    dataset='', #ai4forest_debug
    batch_size=16,

    # Architecture
    arch='unet',  # Defaults to unet
    backbone='resnet50',  # Defaults to resnet50
    use_pretrained_model=False,

    # Model variant specifics
    variant='baseline', # 'ensemble', None
    ensemble_size=3,
    loss_name='l2',  # Defaults to shift_l1

    # Optimization
    optim='AdamW',  # Defaults to AdamW
    n_iterations=50, #100, 25000
    log_freq=50,
    initial_lr=1e-3,
    weight_decay=1e-2,
    use_standardization=False, #default: False
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

print(f"Defaults: {defaults}")

"""
if not debug:
    # Set everything to None recursively
    defaults = GeneralUtility.fill_dict_with_none(defaults)
"""

# Add the hostname to the defaults
defaults['computer'] = socket.gethostname()
print(f"Defaults: {defaults}")

# Configure wandb logging
wandb.init(
    config=defaults,
    project='base-001',  # automatically changed in sweep
    entity=None,  # automatically changed in sweep
)
config = wandb.config
config = GeneralUtility.update_config_with_default(config, defaults)

# Ensure defaults are applied properly
for key, value in defaults.items():
    if getattr(config, key, None) is None:
        setattr(config, key, value)  # Apply default if None

print(f"WandB Config: {config.items}")

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
    if config.variant == 'ensemble':
        for idx, model_path in enumerate(runner.model_paths['ensemble']):
            permanent_path = os.path.join(config.get('model_save_dir', './models'), f'ensemble_model_{idx}.pt')
            shutil.copy(model_path, permanent_path)
            print(f"Saved ensemble model {idx+1} to {permanent_path}")
    else:
        variant = config.variant
        model_path = runner.model_paths[f'{variant}']
        permanent_path = os.path.join(config.get('model_save_dir', './models'), f'{variant}_model.pt')
        shutil.copy(model_path, permanent_path)
        print(f"Saved {variant} model to {permanent_path}")            

    # Close wandb run
    wandb_dir_path = wandb.run.dir
    wandb.join()

    # Delete the local files
    if os.path.exists(wandb_dir_path):
        shutil.rmtree(wandb_dir_path)
