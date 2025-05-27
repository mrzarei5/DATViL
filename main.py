import os

import torch
import yaml
from trainers import run_datvil, run_datvilc
from utils.arguments import get_arguments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # Load config file
    args = get_arguments()
    assert os.path.exists(args.config)

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    cfg["shots"] = args.shots
    cfg["root_path"] = args.root_path
    cfg["per_sample_train"] = args.per_sample_train
    cfg["plus_residual"] = args.plus_residual
    cfg["plus_transform"] = args.plus_transform
    cfg["model_name"] = args.model_name

    if cfg["model_name"] not in ["datvil", "datvilc"]:
        raise ValueError(
            "Parameter model_name should be set to either datvil or datvilc."
        )

    # Determine number of training epochs
    if cfg["dataset"] in ["imagenet", "food101"]:
        cfg["train_epoch"] = 10  # Always 10 for these datasets, regardless of shots
    elif args.shots == 16:
        cfg["train_epoch"] = 150
    else:
        cfg["train_epoch"] = 100

    if cfg["model_name"] == "datvil":
        run_datvil(cfg)
    elif cfg["model_name"] == "datvilc":
        run_datvilc(cfg)
    else:
        raise ValueError("model_name must be either 'datvil' or 'datvilc'")


if __name__ == "__main__":
    main()
