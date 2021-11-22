"""Unsupervised discovery of biases in models on a test dataset (multi-label classification).

"""

import os
import argparse
import warnings
import json
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing
from torchvision import transforms

from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering

from experiments.coco.multilabel_classify import CocoDetection
import experiments.coco.models as models
from experiments.coco.utils import add_flops_counting_methods
from experiments.celeba.cluster_utils import find_maximum
from experiments.coco.cluster_utils import get_cluster_details, init_helpers, evaluate, COCO_ROOT

torch.multiprocessing.set_sharing_strategy("file_system")
warnings.simplefilter("ignore", UserWarning)


def load_model(args):
    """Returns the requested model loaded from file.

    Args:
        args: arguments containing the location of the model at args.model.

    Returns:
        model : Requested model.
    """
    model = models.__dict__[args.model_specific_type](num_classes=80)
    model = add_flops_counting_methods(model)
    model.eval()

    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    return model


def sorted_clusters(tree_file):
    """Returns the clusters sorted by their losses.

    Args:
        tree_file: JSON file location containing the clustering tree data.

    Returns:
        losses : list of lists containing clusters sorted by their losses.
    """
    with open(tree_file, "r") as fp:
        data = json.load(fp)

    losses = []

    def sort_losses(root):
        if root is None:
            return
        if root["leaf"] == "True":
            losses.append(
                (
                    "cluster" + str(root["id"]),
                    root["loss"],
                    root["acc"],
                    root["f1"],
                    root["f1_c"],
                    int(len(root["images"])),
                )
            )

        if "children" in root:
            for child in root["children"]:
                sort_losses(child)

    sort_losses(data)

    return losses


def convert_to_dict(tree_file, accuracy, threshold=0.66):
    """Returns the clusters sorted by their losses in dict format.

    Args:
        tree_file: JSON file location containing the clustering tree data.

    Returns:
        data : list of dicts containing clusters sorted by their losses.
        columns : list of dicts containing attributes of the clusters as columns.
    """
    columns = [
        {
            "field": "index",
            "title": "Index",
            "sortable": True,
        },
        {
            "field": "name",
            "title": "Name",
            "sortable": False,
        },
        {
            "field": "loss",
            "title": "Loss",
            "sortable": True,
        },
        {
            "field": "acc",
            "title": "Accuracy",
            "sortable": True,
        },
        {
            "field": "f1",
            "title": "F1",
            "sortable": True,
        },
        {
            "field": "f1_c",
            "title": "F1_class",
            "sortable": True,
        },
        {
            "field": "count",
            "title": "Count",
            "sortable": True,
        },
    ]

    losses = sorted_clusters(tree_file)
    new_losses = sorted(losses, key=lambda x: x[2])

    idx = 1
    data = []
    for i in new_losses:
        if float(i[2]) < float(accuracy) * threshold:
            data.append(
                {
                    "index": idx,
                    "name": i[0],
                    "loss": round(float(i[1]), 2),
                    "acc": round(float(i[2]), 2),
                    "f1": round(float(i[3]), 2),
                    "f1_c": round(float(i[4]), 2),
                    "count": int(i[5]),
                }
            )
            idx += 1

    return data, columns


def discover(args, logger):
    """Generates tree JSON of the obtained clustering.

    Args:
        args: arguments to the program.
        logger: logger object.

    Returns:
        None.
    """
    _, category_id_to_coco_label, _, cat_ids = init_helpers()

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

    logger.info("Loaded dataset.")

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=16)

    coco_img_id_to_index = {id: i for i, id in enumerate(val_dataset.ids)}

    class Identity(nn.Module):
        def forward(self, x):
            return x

    if not os.path.exists(f"{args.clustering_output_dir}"):
        os.makedirs(f"{args.clustering_output_dir}")

    image_details = {}
    image_list = []
    for i, (index, image, target) in tqdm(enumerate(val_loader)):
        target = target.cuda(non_blocking=True)
        target = target.max(dim=1)[0]
        image_list.extend(index)
        for j in range(image.shape[0]):
            x = int(index[j])
            image_details[x] = {}
            image_details[x]["label"] = target[j].cpu().detach().numpy()

    model = load_model(args)
    evaluate(model, val_loader, category_id_to_coco_label[cat_ids[args.biased_category]], image_details, verbose=False)

    cat_preds = []
    for i in image_details:
        if image_details[i]["pred"][category_id_to_coco_label[cat_ids[args.biased_category]]] == 1:
            cat_preds.append(i)

    indices = [coco_img_id_to_index[i] for i in cat_preds]

    # Clustering predicted attribute
    cat_pred_set = torch.utils.data.Subset(val_dataset, indices)
    cat_pred_loader = torch.utils.data.DataLoader(cat_pred_set, batch_size=1, shuffle=False, num_workers=16)

    logger.info(f"Loaded {len(cat_pred_set)} images for category=={args.biased_category}.")

    evaluate(model, cat_pred_loader, category_id_to_coco_label[cat_ids[args.biased_category]])

    # Feature Extraction

    model = load_model(args)

    model.module.classifier = Identity()

    feature_vector_size = 1920
    feature_vectors = np.empty((0, feature_vector_size))

    for _, (index, image, target) in tqdm(enumerate(cat_pred_loader)):
        with torch.no_grad():
            temp = model(image.cuda())

        feature_vectors = np.vstack((feature_vectors, temp.cpu().detach().numpy()))

    logger.info("Computed FC feature vectors.")

    class TreeNode:
        def __init__(self, i, left, right, images):
            self.id = i
            self.images = images
            self.right = right
            self.left = left
            self.loss = 0
            self.acc = 0
            self.f1 = 0
            self.f1_c = 0
            self.inc_images = None
            self.leaf = False

        def flatten(self, leaf_nodes):
            if self.id in leaf_nodes or self.acc == 100:
                print("LEAFNODE: ", self.id, "-", len(self.images))
                self.leaf = True

                indices = [coco_img_id_to_index[i] for i in self.images]
                data_subset = torch.utils.data.Subset(val_dataset, indices)
                loader = torch.utils.data.DataLoader(data_subset, batch_size=1, shuffle=False, num_workers=4)

                self.loss, self.f1, self.f1_c, self.acc, self.inc_images = get_cluster_details(
                    model, loader, category_id_to_coco_label[cat_ids[args.biased_category]]
                )

                temp = {
                    "id": int(self.id),
                    "images": self.images,
                    "inc_images": self.inc_images,
                    "loss": float(self.loss),
                    "acc": float(self.acc),
                    "f1": float(self.f1),
                    "f1_c": float(self.f1_c),
                    "leaf": str(self.leaf),
                }
                return temp
            else:
                return {
                    "id": int(self.id),
                    "leaf": str(self.leaf),
                    "children": [
                        self.left.flatten(leaf_nodes) if self.left else None,
                        self.right.flatten(leaf_nodes) if self.right else None,
                    ],
                }

    num_samples = len(feature_vectors)
    clustering = AgglomerativeClustering().fit(feature_vectors)

    logger.info("Clustered feature vectors.")

    image_list = cat_preds
    model = load_model(args)

    tree = []
    for i in range(num_samples):
        tree.append(TreeNode(i, None, None, [image_list[i]]))

    for i in range(num_samples, 2 * num_samples - 1):
        tree.append(
            TreeNode(
                i,
                tree[clustering.children_[i - num_samples][0]],
                tree[clustering.children_[i - num_samples][1]],
                tree[clustering.children_[i - num_samples][0]].images
                + tree[clustering.children_[i - num_samples][1]].images,
            )
        )

    reversed_tree = list(reversed(tree))

    def max_leaf_node_size_check(leaf_nodes):
        for i in leaf_nodes:
            if len(leaf_nodes[i]) < 5:
                return False

        return True

    def min_leaf_node_size_check(leaf_nodes):
        for i in leaf_nodes:
            if len(leaf_nodes[i]) > 100:
                return True

        return False

    leaf_nodes = {}
    leaf_nodes_dict = {}

    for i in range(num_samples):
        leaf_nodes[i] = [i]

    min_leaf_node_size = 2
    max_leaf_node_size = None
    k = num_samples

    for i, j in clustering.children_:
        leaf_nodes[k] = leaf_nodes[i] + leaf_nodes[j]
        del leaf_nodes[i], leaf_nodes[j]
        leaf_nodes_dict[len(leaf_nodes)] = leaf_nodes.copy()

        if max_leaf_node_size_check(leaf_nodes) and max_leaf_node_size is None:
            max_leaf_node_size = len(leaf_nodes)

        if min_leaf_node_size_check(leaf_nodes) and len(leaf_nodes) > min_leaf_node_size:
            min_leaf_node_size = len(leaf_nodes)

        k += 1

    logger.info(f"Min leaf node size = {min_leaf_node_size}.")
    logger.info(f"Max leaf node size = {max_leaf_node_size}.")

    if min_leaf_node_size <= max_leaf_node_size:
        _, max_length = find_maximum(leaf_nodes_dict, min_leaf_node_size, max_leaf_node_size, feature_vectors)
    else:
        _, max_length = find_maximum(leaf_nodes_dict, max_leaf_node_size, min_leaf_node_size, feature_vectors)

    final_leaf_nodes = leaf_nodes_dict[max_length]

    logger.info(f"Leaf nodes = {len(final_leaf_nodes)}.")

    logger.info(f"Generating JSON..")

    with open(f"{args.clustering_output_dir}{args.biased_category.replace(' ', '_')}_fc_treeData.json", "w") as fp:
        json.dump(
            reversed_tree[0].flatten(final_leaf_nodes),
            fp,
        )

    logger.info(
        f"JSON generated at {args.clustering_output_dir}{args.biased_category.replace(' ', '_')}_fc_treeData.json."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bias Discovery")

    parser.add_argument("--model", type=str, help="model path", required=True)
    parser.add_argument("--model_type", type=str, help="model type", required=True)
    parser.add_argument("--model_specific_type", type=str, help="model specific type", required=True)
    parser.add_argument("--gradcam_layer", type=str, help="penultimate layer for gradcam", required=True)
    parser.add_argument("--biased_category", type=str, help="category to check gradcam against", required=True)
    parser.add_argument("--clustering_output_dir", type=str, help="output dir", required=True)

    a = parser.parse_args()

    l = logging.getLogger(__name__)
    l.setLevel(logging.INFO)

    c_handler = logging.StreamHandler(sys.stdout)
    c_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)

    l.addHandler(c_handler)

    discover(a, l)
