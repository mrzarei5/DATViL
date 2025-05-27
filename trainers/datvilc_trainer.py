import os
import random
import json
import csv

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import clip
from datasets import build_dataset
from datasets.utils import build_data_loader
from datasets.imagenet import ImageNet
from utils.metrics import cls_acc
from utils.clip_utils import (
    clip_attributes,
    extract_candidates,
    pre_load_features,
    process_attributes_imagenet,
)
from models import DATVIL


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_datvilc(
    cfg, test_features, test_labels, att_weights, clip_model, train_loader
):
    """
    Train DATViL-C model.

    Args:
        cfg: Configuration dictionary.
        test_features: Precomputed test set features.
        test_labels: Test set labels.
        att_weights: Attribute weights.
        clip_model: CLIP model.
        train_loader: Dataloader for training.

    Returns:
        Trained network of DATVIL-C.
    """
    alpha = cfg["alpha"]
    network = DATVIL(
        att_weights,
        alpha,
        clip_model.dtype,
        cfg["plus_residual"],
        cfg["plus_transform"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": network.text_attributes_weights, "lr": cfg["lr_transformer"]},
            {"params": network.text_attributes_residuals, "lr": cfg["lr_residual"]},
        ],
        eps=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg["train_epoch"] * len(train_loader)
    )

    for train_idx in range(1, cfg["train_epoch"] + 1):
        network.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print("Train Epoch: {:} / {:}".format(train_idx, cfg["train_epoch"]))

        for images, target in tqdm(train_loader):
            images, target = images.to(device), target.to(device)
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            updated_classifiers = network()
            updated_classifiers = updated_classifiers / updated_classifiers.norm(
                dim=-1, keepdim=True
            )
            logits = 100.0 * image_features @ updated_classifiers.t()

            loss = F.cross_entropy(logits, target)
            acc = cls_acc(logits, target)
            correct_samples += acc / 100 * len(logits)
            all_samples += len(logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"LR: {current_lr:.6f}, Acc: {correct_samples / all_samples:.4f} ({correct_samples:.0f}/{all_samples:.0f}), "
            f"Loss: {sum(loss_list) / len(loss_list):.4f}"
        )

    network.eval()
    with torch.no_grad():
        updated_classifiers = network()
        updated_classifiers /= updated_classifiers.norm(dim=-1, keepdim=True)
        logits = 100.0 * test_features @ updated_classifiers.t()
        acc = cls_acc(logits, test_labels)

    print("**** DATViL-C test accuracy: {:.2f}. ****\n".format(acc))

    log_file = cfg["log_file"]
    with open(log_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                cfg["dataset"],
                str(cfg["shots"]),
                str(alpha),
                str(cfg["train_epoch"]),
                str(acc),
            ]
        )

    torch.save(
        {
            "model_state_dict": network.state_dict(),
        },
        os.path.join(
            cfg["save_dir"], "network_" + str(cfg["train_epoch"]) + ".pth.tar"
        ),
    )
    return network


def run_datvilc(cfg):
    """
    Run training or loading of DATViL-C model and export results.
    """

    save_dir = os.path.join(
        "./checkpoints",
        cfg["model_name"],
        cfg["dataset"],
        f"{cfg['shots']}shot_{cfg['alpha']}alpha",
    )
    os.makedirs(save_dir, exist_ok=True)
    cfg["save_dir"] = save_dir

    print("\nRunning configs.")
    print(cfg, "\n")

    # Load CLIP model
    clip_model, preprocess = clip.load(cfg["backbone"])
    clip_model = clip_model.to(torch.float32)
    clip_model.eval()

    # Set random seed
    random.seed(1)
    torch.manual_seed(1)

    print("Preparing dataset.")

    if cfg["dataset"] == "imagenet":
        dataset = ImageNet(cfg["root_path"], cfg["shots"], preprocess)
        attributes_weights = process_attributes_imagenet(
            dataset.classnames, dataset.attributes, clip_model
        )
        test_loader = torch.utils.data.DataLoader(
            dataset.test, batch_size=64, num_workers=8, shuffle=False
        )
        train_loader = torch.utils.data.DataLoader(
            dataset.train, batch_size=256, num_workers=8, shuffle=True
        )
    else:
        dataset = build_dataset(cfg["dataset"], cfg["root_path"], cfg["shots"])
        attributes_weights = clip_attributes(
            dataset.classnames, dataset.attributes, clip_model
        )
        test_loader = build_data_loader(
            data_source=dataset.test,
            batch_size=64,
            is_train=False,
            tfm=preprocess,
            shuffle=False,
        )
        train_tranform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=224,
                    scale=(0.5, 1),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        train_loader = build_data_loader(
            data_source=dataset.train_x,
            batch_size=256,
            tfm=train_tranform,
            is_train=True,
            shuffle=True,
        )

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)

    log_file = os.path.join(cfg["save_dir"], "log.csv")
    if not os.path.isfile(log_file):
        with open(log_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["dataset", "shot", "alpha", "iter", "acc"])

    cfg["log_file"] = log_file

    datvilc_path = os.path.join(
        cfg["save_dir"], "network_" + str(cfg["train_epoch"]) + ".pth.tar"
    )

    if os.path.isfile(datvilc_path):
        network = DATVIL(
            attributes_weights,
            cfg["alpha"],
            clip_model.dtype,
            cfg["plus_residual"],
            cfg["plus_transform"],
        ).to(device)
        network_saved = torch.load(datvilc_path)
        network.load_state_dict(network_saved["model_state_dict"])
        network.to(device)
    else:
        network = train_datvilc(
            cfg,
            test_features,
            test_labels,
            attributes_weights,
            clip_model,
            train_loader,
        )

    candidate_classes = extract_candidates(
        cfg, network, test_features, dataset.classnames
    )

    with open(
        os.path.join(cfg["save_dir"], f"{cfg['dataset']}_{cfg['shots']}_labels.json"),
        "w",
    ) as outfile:
        json.dump(candidate_classes, outfile, indent=2)
