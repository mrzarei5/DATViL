import torch
import os

import clip


def text_encoder(text, clip_model):
    with torch.no_grad():
        texts = clip.tokenize(text).cuda()
        embeddings = clip_model.encode_text(texts)
        embeddings /= embeddings.norm(dim=-1, keepdim=True)
    return embeddings


def clip_attributes_discriminative(attributes, clip_model):
    with torch.no_grad():
        texts = clip.tokenize(attributes).cuda()

        attribute_embeddings = clip_model.encode_text(texts)

    return attribute_embeddings.cuda()


def clip_attributes(classnames, attributes, clip_model):
    with torch.no_grad():
        attributes_weights = []

        for classname in classnames:
            # Tokenize the prompts
            texts = clip.tokenize(attributes[classname]).cuda()

            attribute_embeddings = clip_model.encode_text(texts)

            attribute_embedding = attribute_embeddings.mean(dim=0)

            attributes_weights.append(attribute_embedding)

        attributes_weights = torch.stack(attributes_weights, dim=0).cuda()
    return attributes_weights


def extract_candidates(cfg, datvil, test_features, classnames, topk=2):
    datvil.eval()
    with torch.no_grad():
        updated_clip_weights = datvil()
        updated_clip_weights /= updated_clip_weights.norm(dim=-1, keepdim=True)

        logits = 100.0 * test_features @ updated_clip_weights.t()

    pred = logits.topk(topk, 1, True, True)[1]

    label_pairs = []
    for i in range(len(pred)):
        candidates = pred[i]

        labels_names = [classnames[index] for index in candidates]

        labels_names = list(set(labels_names))

        if len(labels_names) == 1:  # if the classnames have the exact same names.
            label_tuple1 = (labels_names[0], labels_names[0])
            label_tuple2 = (labels_names[0], labels_names[0])
        else:
            label_tuple1 = (labels_names[0], labels_names[1])
            label_tuple2 = (labels_names[1], labels_names[0])

        if (label_tuple1 in label_pairs) or (label_tuple2 in label_pairs):
            continue
        else:
            label_pairs.append(label_tuple1)

    return label_pairs


def process_attributes_imagenet(classnames, attributes, clip_model):
    imagenet_templates = [
        "itap of a {}",
        "a bad photo of the {}",
        "a origami {}",
        "a photo of the large {}",
        "a {} in a video game",
        "art of the {}",
        "a photo of the small {}",
        "a photo of a {}",
    ]
    with torch.no_grad():
        attributes_new = []
        for class_name in classnames:
            attributes_this = attributes[class_name]

            attribute_features = []
            for attribute in attributes_this:
                attribute_single_revised = []
                position = attribute.find(class_name)
                if attribute.startswith(class_name):
                    rest_of_sentence = attribute[position + len(class_name) :].strip()

                    if rest_of_sentence.startswith("with"):
                        for template in imagenet_templates:
                            attribute_new = (
                                template.format(class_name) + " " + rest_of_sentence
                            )
                            if not attribute_new.endswith("."):
                                attribute_new += "."
                            attribute_single_revised.append(attribute_new)
                    else:
                        # print(f"The rest of sentence for attribute '{attribute}' does not start with with.")
                        continue
                else:
                    # print(f"The attribute '{attribute}' does not start with '{class_name}'.")
                    continue

                if attribute_single_revised:
                    texts = clip.tokenize(attribute_single_revised).cuda()
                    attribute_embeddings = clip_model.encode_text(texts)
                    attribute_embedding = attribute_embeddings.mean(dim=0)
                    attribute_features.append(attribute_embedding)

            attribute_features = (
                torch.stack(attribute_features, dim=0).cuda().mean(dim=0)
            )
            attributes_new.append(attribute_features)

        attributes_new = torch.stack(attributes_new, dim=0).cuda()

    return attributes_new


def pre_load_features(cfg, split, clip_model, loader):
    datvil_dir = os.path.join(
        "./checkpoints",
        "datvilc",
        cfg["dataset"],
        str(cfg["shots"]) + "shot" + "_" + str(cfg["alpha"]) + "alpha",
    )

    features_path = datvil_dir + "/" + split + "_f.pt"
    label_path = datvil_dir + "/" + split + "_l.pt"

    if os.path.exists(features_path):
        features = torch.load(features_path)
        labels = torch.load(label_path)
    else:
        features, labels = [], []
        with torch.no_grad():
            for i, (images, target) in enumerate(loader):
                images, target = images.cuda(), target.cuda()
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)
        features, labels = torch.cat(features), torch.cat(labels)
        torch.save(features, features_path)
        torch.save(labels, label_path)
    return features, labels
