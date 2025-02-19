import getpass
import os
import shutil
import socket
import sys
import tempfile
import warnings
from contextlib import contextmanager

import wandb
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from runner import Runner
from utilities import GeneralUtility

warnings.filterwarnings('ignore')

debug = "--debug" in sys.argv

defaults = dict(
    # Model save directory
    model_save_dir="./models/trained/",  # Train models go here

    # System
    seed=1,
    n_iterations=50,
    log_freq=25,

    # Data
    dataset='satellite_data', #ai4forest_debug 'satellite_data'
    batch_size=16,

    # Architecture
    arch='unet',
    backbone='resnet50',
    use_pretrained_model=False,

    # Model variant specifics
    model_variant='ensemble', # 'ensemble', 'baseline', 'gaussian', 'quantiles'
    ensemble_size=3,
    quantiles=[0.1, 0.5, 0.9],
    loss_name='l2', #'pinball', #'gaussian'

    # Optimization
    optim='AdamW',
    initial_lr=1e-3,
    weight_decay=1e-2,
    use_standardization=False,
    use_augmentation=False,
    use_label_rescaling=False,

    # Efficiency
    fp16=False,
    use_memmap=False,
    num_workers_per_gpu=8,

    # Other
    use_weighted_sampler='g10',
    use_weighting_quantile=10,
    use_swa=False,
    use_mixup=False,
    use_grad_clipping=False,
    use_input_clipping=False,
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
    project='base-001',
    entity=None,
)
config = wandb.config
config = GeneralUtility.update_config_with_default(config, defaults)

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
    # Check if we are running on the GCP cluster
    is_htc = 'htc-' in os.uname().nodename
    is_gcp = 'gpu' in os.uname().nodename and not is_htc
    if is_gcp:
        print('Running on GCP, marking as preemptable.')
        wandb.mark_preempting()

    runner = Runner(config=config, tmp_dir=tmp_dir, debug=debug)
    runner.run()

    # Save the trained ensemble models
    trained_model_dir = config.get('model_save_dir', './models/trained/')
    os.makedirs(trained_model_dir, exist_ok=True)

    if config.model_variant == 'ensemble':
        for idx, model_path in enumerate(runner.model_paths['ensemble']):
            save_path = os.path.join(trained_model_dir, f'ensemble_model_{idx}.pt')
            shutil.copy(model_path, save_path)
            print(f"✅ Saved ensemble model {idx+1} to {save_path}")
    
    else:
        model_variant = config.model_variant
        model_path = runner.model_paths[f'{model_variant}']
        save_path = os.path.join(trained_model_dir, f'{model_variant}_model.pt')
        shutil.copy(model_path, save_path)
        print(f"✅ Saved {model_variant} model to {save_path}")

        # Close wandb run
        wandb_dir_path = wandb.run.dir
        wandb.join()

        # Delete the local files
        if os.path.exists(wandb_dir_path):
            shutil.rmtree(wandb_dir_path)

    # Move Trained Models to Prediction Directory
    prediction_model_dir = "./models/prediction/"
    os.makedirs(prediction_model_dir, exist_ok=True)

    for model_filename in os.listdir(trained_model_dir):
        src_path = os.path.join(trained_model_dir, model_filename)
        dst_path = os.path.join(prediction_model_dir, model_filename)
        shutil.copy(src_path, dst_path)

    print("🚀 All trained models copied to 'models/prediction/' for inference.")
