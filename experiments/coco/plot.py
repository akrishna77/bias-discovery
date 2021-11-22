import os
import argparse
import warnings
import sys

import logging
import json

from gradcam import GradCAM

import torch
import torch.nn as nn
import torch.multiprocessing
from torchvision import transforms

from experiments.coco.multilabel_classify import CocoDetection
import experiments.coco.models as models
from experiments.coco.utils import add_flops_counting_methods
from experiments.coco.cluster_utils import plot_both, init_helpers, get_cluster_details, COCO_ROOT

warnings.simplefilter("ignore", UserWarning)
torch.multiprocessing.set_sharing_strategy("file_system")


def load_model(args):
    model = models.__dict__[args.model_specific_type](num_classes=80)
    model = add_flops_counting_methods(model)
    model.eval()

    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    return model


def plot(args, logger):
    _, category_id_to_coco_label, _, cat_ids = init_helpers()

    torch.cuda.empty_cache()

    val_dataset = CocoDetection(
        os.path.join(COCO_ROOT, "val2017"),
        os.path.join(COCO_ROOT, "annotations/instances_val2017.json"),
        transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        ),
    )

    coco_img_id_to_index = {id: i for i, id in enumerate(val_dataset.ids)}

    class Identity(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

        def forward(self, x):
            return x

    model = load_model(args)

    gradcam_generator = GradCAM.from_config(model_type=args.model_type, arch=model, layer_name=args.gradcam_layer)
    gradcam_category = args.biased_category

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

    indices = [coco_img_id_to_index[i] for i in cluster_to_plot]
    cluster_dataset = torch.utils.data.Subset(val_dataset, indices)
    cluster_loader = torch.utils.data.DataLoader(cluster_dataset, batch_size=1, shuffle=False, num_workers=16)

    logger.info(f"Retrieving cluster details..")

    (
        _,
        _,
        _,
        _,
        inc_images,
    ) = get_cluster_details(model, cluster_loader, category_id_to_coco_label[cat_ids[gradcam_category]])

    plot_both(
        args.cluster_id,
        args.output_path,
        model,
        cluster_loader,
        gradcam_generator,
        inc_images,
        gradcam_idx=category_id_to_coco_label[cat_ids[gradcam_category]],
    )

    logger.info(f"Plotted cluster.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bias Discovery Plot")

    parser.add_argument("--model", type=str, help="model name", required=True)
    parser.add_argument("--model_type", type=str, help="model type", required=True)
    parser.add_argument("--model_specific_type", type=str, help="model specific type", required=True)
    parser.add_argument("--gradcam_layer", type=str, help="gradcam layer", required=True)
    parser.add_argument("--biased_category", type=str, help="category to check", required=True)
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
