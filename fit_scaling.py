import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["AUTOGRAPH_VERBOSITY"] = "1"
import logging

# Set up logger
logger = logging.getLogger()
logger.handlers = []
ch = logging.StreamHandler()
formatter = logging.Formatter(
    fmt="%(asctime)s (%(levelname)s): %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel("INFO")

import torch

from gemnet.model.gemnet import GemNet
from gemnet.training.trainer import Trainer
from gemnet.training.metrics import Metrics
from gemnet.training.data_container import DataContainer
from gemnet.training.data_provider import DataProvider

import yaml
from tqdm import trange
from seml.config import generate_configs

from gemnet.model.utils import write_json
from gemnet.model.layers.scaling import AutomaticFit


def get_configs(path):
    with open(path, "r") as stream:
        try:
            all_configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    configs = generate_configs(all_configs)
    lengths = [len(config) for config in configs]
    max_length = max(lengths)
    configs = [config for i, config in enumerate(configs) if lengths[i] >= max_length]
    out = []
    for config in configs:
        if config not in out:
            out.append(config)
    return out


def run(
    nBatches,
    num_spherical,
    num_radial,
    num_blocks,
    emb_size_atom,
    emb_size_edge,
    emb_size_trip,
    emb_size_quad,
    emb_size_rbf,
    emb_size_cbf,
    emb_size_sbf,
    num_before_skip,
    num_after_skip,
    num_concat,
    num_atom,
    emb_size_bil_quad,
    emb_size_bil_trip,
    triplets_only,
    forces_coupled,
    direct_forces,
    mve,
    cutoff,
    int_cutoff,
    envelope_exponent,
    extensive,
    output_init,
    scale_file,
    data_seed,
    val_dataset,
    tfseed,
    batch_size,
    comment,
    overwrite_mode=3,
    **kwargs,
):
    """
    Run this function to automatically fit all scaling factors in the network.
    """
    torch.manual_seed(tfseed)

    def init(scale_file):
        # initialize file
        # same for all models
        preset = {"comment": comment}
        write_json(scale_file, preset)

    if os.path.exists(scale_file):
        print(f"Already found existing file: {scale_file}")
        if str(overwrite_mode) == "1":
            print("Selected: Overwrite the current file.")
            init(scale_file)
        elif str(overwrite_mode) == "2":
            print("Selected: Only fit unfitted variables.")
        else:
            print("Selected: Exit script")
            return
    else:
        init(scale_file)

    AutomaticFit.set2fitmode()

    logging.info("Initialize model")
    model = GemNet(
        num_spherical=num_spherical,
        num_radial=num_radial,
        num_blocks=num_blocks,
        emb_size_atom=emb_size_atom,
        emb_size_edge=emb_size_edge,
        emb_size_trip=emb_size_trip,
        emb_size_quad=emb_size_quad,
        emb_size_rbf=emb_size_rbf,
        emb_size_cbf=emb_size_cbf,
        emb_size_sbf=emb_size_sbf,
        num_before_skip=num_before_skip,
        num_after_skip=num_after_skip,
        num_concat=num_concat,
        num_atom=num_atom,
        emb_size_bil_quad=emb_size_bil_quad,
        emb_size_bil_trip=emb_size_bil_trip,
        num_targets=2 if mve else 1,
        cutoff=cutoff,
        int_cutoff=int_cutoff,
        envelope_exponent=envelope_exponent,
        forces_coupled=forces_coupled,
        direct_forces=True,  # evaluates faster
        triplets_only=triplets_only,
        activation="swish",
        extensive=extensive,
        output_init=output_init,
        scale_file=scale_file,
    )

    logging.info("Load dataset")
    # Initialize validation datasets
    val_data_container = DataContainer(
        val_dataset, cutoff=cutoff, int_cutoff=int_cutoff, triplets_only=triplets_only
    )
    val_data_provider = DataProvider(
        val_data_container,
        0,
        nBatches * batch_size,
        batch_size,
        seed=data_seed,
        shuffle=True,
        random_split=True,
    )

    # Initialize datasets
    dataset_iter = val_data_provider.get_dataset("val")
    logging.info("Prepare training")

    # Initialize trainer
    trainer = Trainer(model, mve=mve)
    metrics = Metrics("train", trainer.tracked_metrics, None)

    # Training loop
    logging.info("Start training")

    while not AutomaticFit.fitting_completed():
        for step in trange(0, nBatches, desc="Training..."):
            trainer.test_on_batch(dataset_iter, metrics)

        current_var = AutomaticFit.activeVar
        if current_var is not None:
            current_var.fit()  # fit current variable
        else:
            print("Found no variable to fit. Something went wrong!")

    logging.info(f"\n Fitting done. Results saved to: {scale_file}")


if __name__ == "__main__":

    config_path = "config.yaml"

    nBatches = 25  ## number of batches to use to fit a single variable
    config = get_configs(config_path)[0]

    config["scale_file"] = "scaling_factors.json"
    config["batch_size"] = 32
    config["direct_forces"] = True
    config["triplets_only"] = False
    run(nBatches, **config)
