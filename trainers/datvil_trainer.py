import os
import random
import json
import csv

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import clip
from datasets import build_dataset
from datasets.utils import build_data_loader, FewShotData, FewShotDataImagenet
from datasets.imagenet import ImageNet
from models import DATVIL
from utils.clip_utils import (
    clip_attributes,
    pre_load_features,
    clip_attributes_discriminative,
    process_attributes_imagenet,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_datvil(
    cfg,
    test_logits,
    test_labels,
    test_features,
    clip_model,
    datvilc,
    few_shot_data_class,
    attributes_discriminative,
    topk=5,
):
    pred = test_logits.topk(topk, 1, True, True)[1]

    epoch_number = cfg["per_sample_train"]
    beta = cfg["beta"]
    alpha = cfg["alpha"]

    for param in datvilc.parameters():
        param.requires_grad = False

    classifiers_datvilc = datvilc()

    correct_sample_counter = 0
    counter_all = 0

    samples_num = len(pred)

    for i in range(samples_num):
        candidates = pred[i]
        samples, labels, labels_names = few_shot_data_class.getitem(
            candidates.to("cpu").numpy()
        )

        labels_names_distinct = [
            item
            for index, item in enumerate(labels_names)
            if item not in labels_names[:index]
        ]

        if len(labels_names_distinct) == 1:
            labels_names_distinct.append(labels_names_distinct[0])

        label_key1 = labels_names_distinct[0] + ":" + labels_names_distinct[1]
        label_key2 = labels_names_distinct[1] + ":" + labels_names_distinct[0]

        if label_key1 in attributes_discriminative.keys():
            attributes_dicriminative_weights = clip_attributes_discriminative(
                attributes_discriminative[label_key1], clip_model
            )
        else:
            try:
                attributes_dicriminative_weights = clip_attributes_discriminative(
                    [
                        attributes_discriminative[label_key2][1],
                        attributes_discriminative[label_key2][0],
                    ],
                    clip_model,
                )
            except (KeyError, IndexError):
                print(
                    "Context-aware descriptions for {}:{} not found. Please update the descriptions with the information of the new classes.".format(
                        label_key1[0], label_key1[1]
                    )
                )
                continue

        classifiers_datvilc_selected = classifiers_datvilc[candidates]

        classifiers_datvilc_selected = (
            classifiers_datvilc_selected
            / classifiers_datvilc_selected.norm(dim=-1, keepdim=True)
        )

        datvil = DATVIL(
            attributes_dicriminative_weights,
            alpha,
            clip_model.dtype,
            cfg["plus_residual"],
            cfg["plus_transform"],
        ).to(device)
        optimizer = torch.optim.AdamW(
            [
                {"params": datvil.text_attributes_weights, "lr": cfg["lr_transformer"]},
                {"params": datvil.text_attributes_residuals, "lr": cfg["lr_residual"]},
            ],
            eps=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch_number)

        images, target = samples.to(device), labels.to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        for train_idx in range(epoch_number):
            # Train
            classifiers_datvil = datvil()
            classifiers_datvil = classifiers_datvil / classifiers_datvil.norm(
                dim=-1, keepdim=True
            )

            final_weights = classifiers_datvilc_selected + beta * classifiers_datvil

            final_weights = final_weights / final_weights.norm(dim=-1, keepdim=True)

            logits = 100.0 * image_features @ final_weights.t()

            loss = F.cross_entropy(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Eval
        with torch.no_grad():
            classifiers_datvil = datvil()
            classifiers_datvil = classifiers_datvil / classifiers_datvil.norm(
                dim=-1, keepdim=True
            )

            final_weights = classifiers_datvilc_selected + beta * classifiers_datvil

            final_weights = final_weights / final_weights.norm(dim=-1, keepdim=True)

            logits = 100.0 * test_features[i : i + 1] @ final_weights.t()

            pred_this = logits.topk(1, 1, True, True)[1].t()

            if candidates[pred_this[0].item()].item() == test_labels[i]:
                correct_sample_counter += 1
            counter_all += 1
        print("Sample number {:} / {:} done!".format(i, samples_num))
    return correct_sample_counter / counter_all * 100


def run_datvil(cfg):
    save_dir = os.path.join(
        "./checkpoints",
        cfg["model_name"],
        cfg["dataset"],
        f"{cfg['shots']}shot_{cfg['alpha']}alpha_{cfg['beta']}beta",
    )

    os.makedirs(save_dir, exist_ok=True)

    cfg["save_dir"] = save_dir

    datvilc_dir = os.path.join(
        "./checkpoints",
        "datvilc",
        cfg["dataset"],
        str(cfg["shots"]) + "shot" + "_" + str(cfg["alpha"]) + "alpha",
    )

    datvilc_net_dir = os.path.join(
        datvilc_dir, "network_" + str(cfg["train_epoch"]) + ".pth.tar"
    )

    print("\nRunning configs.")
    print(cfg, "\n")

    # CLIP
    clip_model, preprocess = clip.load(cfg["backbone"])
    clip_model = clip_model.to(torch.float32)

    clip_model.eval()

    random.seed(1)
    torch.manual_seed(1)

    print("Preparing dataset.")

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

    if cfg["dataset"] == "imagenet":
        dataset = ImageNet(cfg["root_path"], cfg["shots"], preprocess)
        attributes_weights = process_attributes_imagenet(
            dataset.classnames, dataset.attributes, clip_model
        )
        test_loader = torch.utils.data.DataLoader(
            dataset.test, batch_size=64, num_workers=8, shuffle=False
        )
        few_shot_data_class = FewShotDataImagenet(
            dataset.train,
            dataset.attributes,
            attributes_weights,
            dataset.classnames,
            clip_model,
            input_size=224,
            transform=train_tranform,
            is_train=True,
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
        few_shot_data_class = FewShotData(
            dataset.train_x,
            dataset.attributes,
            attributes_weights,
            dataset.classnames,
            clip_model,
            input_size=224,
            transform=train_tranform,
            is_train=True,
        )

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)

    assert os.path.isfile(datvilc_net_dir)

    datvilc = DATVIL(
        attributes_weights,
        cfg["alpha"],
        clip_model.dtype,
        cfg["plus_residual"],
        cfg["plus_transform"],
    ).to(device)
    network_saved = torch.load(datvilc_net_dir)

    datvilc.load_state_dict(network_saved["model_state_dict"])
    datvilc.to(device)

    attribute_discriminative_path = os.path.join(
        cfg["root_path"], "attributes_discriminative_gpt4", cfg["dataset"] + ".json"
    )

    with open(attribute_discriminative_path, "r") as f:
        attributes_discriminative = json.load(f)

    with torch.no_grad():
        classifiers = datvilc()
        classifiers /= classifiers.norm(dim=-1, keepdim=True)

        logits = 100.0 * test_features @ classifiers.t()

    acc_revised = train_datvil(
        cfg,
        logits,
        test_labels,
        test_features,
        clip_model,
        datvilc,
        few_shot_data_class,
        attributes_discriminative,
        topk=2,
    )

    log_file = os.path.join(cfg["save_dir"], "log.csv")
    if not os.path.isfile(log_file):
        with open(log_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["dataset", "shot", "alpha", "beta", "iter", "acc"])
    with open(log_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                cfg["dataset"],
                str(cfg["shots"]),
                str(cfg["alpha"]),
                str(cfg["beta"]),
                str(cfg["per_sample_train"]),
                str(acc_revised),
            ]
        )
