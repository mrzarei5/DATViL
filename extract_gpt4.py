import argparse
import json
import os

from openai import OpenAI


# should be replaced by OpenAI API key
api_key = "api_key"

dataset_to_message_mapper = {
    "dtd": "I am facing an image classification problem where I have to assign an image of a textual pattern into either the class {} or the class {}. Provide me with visual information that distinguishes these two textures and helps me classify the image. Please present a discriminative one-sentence description for each of these two textures. First provide sentence for {} and then provide sentence for {} without any additional preceding and separated by ‘*’.",
    "fgvc": "I am facing an image classification problem where I have to assign an image of an aircraft into either the class {} or the class {}. Provide me with visual information that distinguishes these two aircraft models and helps me classify the image. Please present a discriminative one-sentence description for each of these two aircraft models. First provide sentence for {} and then provide sentence for {} without any additional preceding and separated by ‘*’.",
    "caltech101": "I am facing an image classification problem where I have to assign an image into either the class {} or the class {}. Provide me with visual information that distinguishes these two classes and helps me classify the image. Please present a discriminative one-sentence description for each of these two classes. First provide sentence for {} and then provide sentence for {} without any additional preceding and separated by ‘*’.",
    "food101": "I am facing an image classification problem where I have to assign an image into either the class {} or the class {}. Provide me with visual information that distinguishes these two food categories and helps me classify the image. Please present a discriminative one-sentence description for each of these two food categories. First provide sentence for {} and then provide sentence for {} without any additional preceding and separated by ‘*’.",
    "imagenet": "I am facing an image classification problem where I have to assign an image into either the class {} or the class {}. Provide me with visual information that distinguishes these two classes and helps me classify the image. Please present a discriminative one-sentence description for each of these two classes. First provide sentence for {} and then provide sentence for {} without any additional preceding and separated by ‘*’.",
    "oxford_flowers": "I am facing an image classification problem where I have to assign an image into either the class {} or the class {}. Provide me with visual information that distinguishes these two flowers and helps me classify the image. Please present a discriminative one-sentence description for each of these two flower categories. First provide sentence for {} and then provide sentence for {} without any additional preceding and separated by ‘*’.",
    "oxford_pets": "I am facing an image classification problem where I have to assign an image into either the class {} or the class {}, two types of pets. Provide me with visual information that distinguishes these two classes and helps me classify the image. Please present a discriminative one-sentence description for each of these two classes. First provide sentence for {} and then provide sentence for {} without any additional preceding and separated by ‘*’.",
    "stanford_cars": "I am facing an image classification problem where I have to assign an image into either the class {} or the class {}. Provide me with visual information that distinguishes these two car models and helps me classify the image. Please present a discriminative one-sentence description for each of these two car categories. First provide sentence for {} and then provide sentence for {} without any additional preceding and separated by ‘*’.",
    "sun397": "I am facing an image classification problem where I have to assign an image of a scene into either the class {} or the class {}. Provide me with visual information that distinguishes these two scenes and helps me classify the image. Please present a discriminative one-sentence description for each of each of these two scenes. First provide sentence for {} and then provide sentence for {} without any additional preceding and separated by ‘*’.",
    "ucf101": "I am facing an image classification problem where I have to assign an image into either the activity {} or the activity {}. Provide me with visual information that distinguishes these two activities and helps me classify the image. Please present a discriminative one-sentence description for each of each of these two activities. First provide sentence for {} and then provide sentence for {} without any additional preceding and separated by ‘*’.",
    "eurosat": "I am facing an image classification problem where I have to assign a satellite image into either the class {} or the class {}. Provide me with visual information that distinguishes these two classes and helps me classify the image. Please present a discriminative one-sentence description for each of each of these two classes. First provide sentence for {} and then provide sentence for {} without any additional preceding and separated by ‘*’.",
}


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--description_file", type=str, help="Path to the descriptions file"
    )
    parser.add_argument("--labels_file", type=str, help="Path to the labels file")
    parser.add_argument("--dataset", type=str, help="Dataset name")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()
    description_file = args.description_file
    labels_file = args.labels_file
    dataset = args.dataset

    attributes_discriminative = {}

    if os.path.exists(description_file):
        with open(description_file, "r") as f:
            attributes_discriminative = json.load(f)

    with open(labels_file, "r") as f:
        labels_pairs = json.load(f)

    client = OpenAI(api_key=api_key)

    for label_pair in labels_pairs:
        label_key1 = label_pair[0] + ":" + label_pair[1]
        label_key2 = label_pair[1] + ":" + label_pair[0]

        if (label_key1 in attributes_discriminative.keys()) or (
            label_key2 in attributes_discriminative.keys()
        ):
            continue

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": dataset_to_message_mapper[dataset].format(
                        " ".join(label_pair[0].split("_")),
                        " ".join(label_pair[1].split("_")),
                        " ".join(label_pair[0].split("_")),
                        " ".join(label_pair[1].split("_")),
                    ),
                },
                {
                    "role": "user",
                    "content": "Here is an example for classes bengal and bombay to understand the structure of the answer that I expect:",
                },
                {
                    "role": "user",
                    "content": "Bengals have distinctive large spots, rosettes, and a more muscular appearance compared to other domestic cats. * Bombays are known for their sleek black coat, copper or gold eyes, and a sturdy, medium-sized body with a striking resemblance to a mini-panther.",
                },
                {
                    "role": "user",
                    "content": "Now provide what I first asked for {} and {} with the same structure. Just provide the sentence for {}, then the sentence for {}, seperated by *. Start the sentences with the class names. Do not say anything else in response.".format(
                        " ".join(label_pair[0].split("_")),
                        " ".join(label_pair[1].split("_")),
                        " ".join(label_pair[0].split("_")),
                        " ".join(label_pair[1].split("_")),
                    ),
                },
            ],
            temperature=0.0,
        )

        info = response.choices[0].message.content.split("*")
        attributes_discriminative[label_pair[0] + ":" + label_pair[1]] = [
            info[0].strip(),
            info[1].strip(),
        ]

        with open(description_file, "w") as outfile:
            json.dump(attributes_discriminative, outfile, indent=2)
    with open(description_file, "w") as outfile:
        json.dump(attributes_discriminative, outfile, indent=2)
