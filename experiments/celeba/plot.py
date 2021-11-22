from __future__ import print_function
import argparse
import warnings

import sys
import logging
import json
import pandas as pd
import torch

import torch.multiprocessing
from torchvision import transforms

from gradcam import GradCAM
from tqdm import tqdm

from datasets.CelebADataset import CelebADataset
from models.celeba_classifier import CelebAClassifier
from experiments.celeba.cluster_utils import plot_both, get_cluster_details

torch.multiprocessing.set_sharing_strategy("file_system")
warnings.simplefilter("ignore", UserWarning)

IMAGE_SIZE = 224


def load_model(args):
    model = CelebAClassifier()
    model.eval()

    model = model.cuda()

    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint, strict=False)

    return model


def plot(args, logger):
    # Load dataset

    torch.cuda.empty_cache()

    dataset = CelebADataset(
        args.csv_file,
        args.dataroot,
        transform=transforms.Compose(
            [
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
            ]
        ),
    )

    logger.info(f"Plotting cluster {args.cluster_id}..")

    logger.info("Loaded dataset.")

    indices = list(range(len(dataset)))
    test_indices = indices[182638:]

    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # Load dataset annotations
    df = pd.read_csv(args.csv_file, delimiter="\\s+", header=0)
    df.replace(to_replace=-1, value=0, inplace=True)

    test_set_df = df.copy()
    test_set_df = test_set_df[182638:]
    test_set_sum = test_set_df.iloc[:, 1:].sum() / len(test_set_df)

    model = load_model(args)

    gradcam_generator = GradCAM.from_config(
        model_type=args.model_type, arch=model.model_ft, layer_name=args.gradcam_layer
    )

    test_set_df.reset_index(drop=True, inplace=True)

    with open(args.tree_json, "r") as fp:
        data = json.load(fp)

    def parse_json(root, cluster_id):
        if root is None:
            return

        if root["leaf"] == "True" and int(root["id"]) == cluster_id:
            return root["images"]

        if "children" in root:
            for child in root["children"]:
                result = parse_json(child, int(args.cluster_id))
                if result is not None:
                    return result

    logger.info(f"Parsing JSON..")
    cluster_to_plot = parse_json(data, int(args.cluster_id))

    cluster_inames = [x[len(args.dataroot) :] for x in cluster_to_plot]
    image_indices = list(test_set_df[test_set_df["Image_Name"].isin(cluster_inames)].index)

    cluster_dataset = torch.utils.data.Subset(test_dataset, image_indices)
    cluster_loader = torch.utils.data.DataLoader(cluster_dataset, batch_size=1, shuffle=False, num_workers=4)

    cluster_df = test_set_df[test_set_df["Image_Name"].isin(cluster_inames)]
    cluster_distribution = cluster_df.iloc[:, 1:].sum() / (len(cluster_df))
    cluster_distribution = cluster_distribution.drop(
        labels=["No_Beard", "Oval_Face", "High_Cheekbones", "Male", "Young", "Smiling", "Attractive", "Heavy_Makeup"]
    )
    cluster_distribution = cluster_distribution.sort_values(ascending=False)

    image_details = {}
    for _, batch in tqdm(enumerate(cluster_loader)):
        for i in range(batch["image"].shape[0]):
            image_details[batch["image_name"][i]] = {}
            image_details[batch["image_name"][i]]["label"] = int(batch["attributes"][i])

    logger.info(f"Retrieving cluster details..")

    (
        _,
        _,
        inc_images,
        _,
        _,
    ) = get_cluster_details(model, cluster_loader, image_details)

    plot_both(
        args.output_path, inc_images, model, gradcam_generator, cluster_loader, image_details, cluster_distribution
    )

    logger.info(f"Plotted cluster.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bias Discovery Plot")

    parser.add_argument("--dataroot", type=str, help="dataset folder", required=True)
    parser.add_argument("--csv_file", type=str, help="annotations", required=True)
    parser.add_argument("--model", type=str, help="model name", required=True)
    parser.add_argument("--gradcam_layer", type=str, help="features from layer", required=True)
    parser.add_argument("--batch_size", type=int, help="batch size", required=True)
    parser.add_argument("--model_type", type=str, help="model type", required=True)
    parser.add_argument("--tree_json", type=str, help="tree json", required=True)
    parser.add_argument("--cluster_id", type=int, help="cluster id", required=True)
    parser.add_argument("--output_path", type=str, help="output directory", required=True)

    a = parser.parse_args()

    l = logging.getLogger(__name__)
    l.setLevel(logging.INFO)

    c_handler = logging.StreamHandler(sys.stdout)
    c_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)

    l.addHandler(c_handler)

    plot(a, l)
