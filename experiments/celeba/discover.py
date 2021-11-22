"""Unsupervised discovery of biases in models on a test dataset (binary classification).

"""

from __future__ import print_function
from pathlib import Path
import argparse
import warnings

import os
import sys
import logging
import json
import pickle
import scipy
import numpy as np
import pandas as pd
from torchvision import transforms
from sklearn.cluster import AgglomerativeClustering

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.multiprocessing

from datasets.CelebADataset import CelebADataset
from models.celeba_classifier import CelebAClassifier
from experiments.celeba.cluster_utils import get_cluster_details, evaluate, find_maximum

IMAGE_SIZE = 224
PROJ_ROOT = Path(__file__).resolve().parent.parent.parent

torch.multiprocessing.set_sharing_strategy("file_system")
warnings.simplefilter("ignore", UserWarning)


def load_model(args):
    """Returns the requested model loaded from file.

    Args:
        args: arguments containing the location of the model at args.model.

    Returns:
        model : Requested model.
    """
    model = CelebAClassifier()
    model.eval()

    model = model.cuda()

    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint, strict=False)

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
        if root["leaf"] == "True" and int(root["count"]) > 5:
            losses.append(
                (
                    "cluster" + str(root["id"]),
                    float(root["loss"]),
                    int(root["name"].split("-")[-1]),
                    int(root["count"]),
                    root["top_attr"],
                    float(root["conf"]),
                    root["images"],
                    root["avg_vec"],
                    root["dist_vec"],
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
            "field": "count",
            "title": "Count",
            "sortable": True,
        },
        {
            "field": "conf",
            "title": "Confidence",
            "sortable": True,
        },
        {
            "field": "top_attr",
            "title": "Top Attributes",
            "sortable": False,
        },
    ]

    losses = sorted_clusters(tree_file)
    new_losses = sorted(losses, key=lambda x: (x[2], -x[1]))

    data = []
    idx = 1
    for i in new_losses:
        if float(i[2]) < float(accuracy) * threshold:
            data.append(
                {
                    "index": idx,
                    "name": i[0],
                    "loss": round(i[1], 2),
                    "acc": i[2],
                    "count": i[3],
                    "conf": round(i[5], 2),
                    "top_attr": i[4],
                }
            )
            idx += 1

    return data, columns


def celeba_nn(tree_file, c_id):
    losses = sorted_clusters(tree_file)
    new_losses = sorted(losses, key=lambda x: (x[2], -x[1]))

    accuracy_cutoff = 99999999
    name_list = []
    vector_list = []
    idx = 0
    for i in new_losses:
        if i[2] == 100.0 and idx < accuracy_cutoff:
            accuracy_cutoff = idx
        name_list.append(i[0])
        vector_list.append(np.asarray(i[7]))
        idx += 1

    vector_list = np.array(vector_list)
    dm = scipy.spatial.distance_matrix(vector_list, vector_list)

    arglist = np.argsort(dm, axis=1)
    for j in arglist[name_list.index("cluster" + str(c_id))]:
        if j >= accuracy_cutoff:
            return int(name_list[j][7:])


def celeba_dist_nn(tree_file, c_id):
    losses = sorted_clusters(tree_file)
    new_losses = sorted(losses, key=lambda x: (x[2], -x[1]))

    accuracy_cutoff = 99999999
    name_list = []
    vector_list = []
    idx = 0
    for i in new_losses:
        if i[2] == 100.0 and idx < accuracy_cutoff:
            accuracy_cutoff = idx
        name_list.append(i[0])
        vector_list.append(np.asarray(i[8]))
        idx += 1

    vector_list = np.array(vector_list)
    dm = scipy.spatial.distance_matrix(vector_list, vector_list)

    arglist = np.argsort(dm, axis=1)
    for j in arglist[name_list.index("cluster" + str(c_id))]:
        if j >= accuracy_cutoff:
            return int(name_list[j][7:])


def discover(args, logger):
    """Generates tree JSON of the obtained clustering.

    Args:
        args: arguments to the program.
        logger: logger object.

    Returns:
        None.
    """

    # Load dataset
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

    logger.info("Loaded dataset.")

    indices = list(range(len(dataset)))
    test_indices = indices[182638:]

    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    class Identity(nn.Module):
        def forward(self, x):
            return x

    # Load dataset annotations
    df = pd.read_csv(args.csv_file, delimiter="\\s+", header=0)
    df.replace(to_replace=-1, value=0, inplace=True)

    test_set_df = df.copy()
    test_set_df = test_set_df[182638:]

    if not os.path.exists(f"{args.clustering_output_dir}"):
        os.makedirs(f"{args.clustering_output_dir}")

    # Populating image info - labels and preds
    image_details = {}
    image_names = []
    image_paths = []

    for _, batch in tqdm(enumerate(test_loader)):
        image_paths.extend(batch["image_path"])
        image_names.extend(batch["image_name"])
        for i in range(batch["image"].shape[0]):
            image_details[batch["image_name"][i]] = {}
            image_details[batch["image_name"][i]]["label"] = int(batch["attributes"][i])

    model = load_model(args)
    model = model.cuda()
    evaluate(model, test_loader, image_details, verbose=True)

    test_set_df["Smiling_Pred"] = [image_details[f"{i}"]["pred"] for i in list(test_set_df["Image_Name"])]

    test_set_df.reset_index(drop=True, inplace=True)

    # Grouping based on predicted attribute
    # image_indices = list(test_set_df[test_set_df["Smiling_Pred"] != test_set_df["Smiling"]].index)
    # cat_pred_subset = torch.utils.data.Subset(test_dataset, image_indices)
    # cat_pred_loader = torch.utils.data.DataLoader(
    #     cat_pred_subset, batch_size=args.batch_size, shuffle=False, num_workers=4
    # )

    subset = test_dataset  # test_dataset
    loader = test_loader  # test_loader

    logger.info(f"Loaded {len(subset)} images.")

    # Computing feature vectors

    if not os.path.exists(f"{args.clustering_output_dir}fc_feature_vectors.pickle"):
        model.model_ft.fc = Identity()
        feature_vector_size = 2048
        feature_vectors = np.empty((0, feature_vector_size))
        for _, batch in tqdm(enumerate(loader)):
            with torch.no_grad():
                temp = model(batch["image"].cuda())
            for i in range(batch["image"].shape[0]):
                image_details[batch["image_name"][i]]["feat"] = temp[i].cpu().detach().numpy()
            feature_vectors = np.vstack((feature_vectors, temp.cpu().detach().numpy()))

        logger.info(f"Computed FC feature vectors = {feature_vectors.shape}.")

        pickle.dump(feature_vectors, file=open(f"{args.clustering_output_dir}fc_feature_vectors.pickle", "wb"))
    else:
        feature_vectors = pickle.load(file=open(f"{args.clustering_output_dir}fc_feature_vectors.pickle", "rb"))

    count = 0
    model = load_model(args)
    model = model.cuda()

    class TreeNode:
        def __init__(self, i, left, right, image_names, image_paths):
            """Initialize variables."""
            self.id = i
            self.acc = None
            self.avg_vec = None
            self.dist_vec = None
            self.image_names = image_names
            self.image_paths = image_paths
            self.inc_images = None
            self.right = right
            self.left = left
            self.incorrect_conf = None
            self.loss = 0
            self.conf = 0
            self.top_attr = None
            self.leaf = False

        def flatten(self, leaf_nodes):
            """Generates tree JSON of the node.

            Args:
                None.
            Returns:
                dict of cluster details.
            """

            image_indices = list(test_set_df[test_set_df["Image_Name"].isin(self.image_names)].index)

            cluster_dataset = torch.utils.data.Subset(test_dataset, image_indices)
            cluster_loader = torch.utils.data.DataLoader(cluster_dataset, batch_size=32, shuffle=False, num_workers=4)

            (
                self.loss,
                self.acc,
                self.inc_images,
                self.incorrect_conf,
                self.conf,
            ) = get_cluster_details(model, cluster_loader, verbose=True)

            if self.id in leaf_nodes or self.acc == 100:
                self.leaf = True

                nonlocal count
                count += 1
                print(f"LEAFNODE #{count}: ", self.id, "-", len(self.image_names))

                cluster_df = test_set_df[test_set_df["Image_Name"].isin(self.image_names)]
                cluster_distribution = cluster_df.iloc[:, 1:].sum() / (len(cluster_df))
                cluster_distribution = cluster_distribution.drop(
                    labels=[
                        "No_Beard",
                        "Smiling_Pred",
                        "Oval_Face",
                        "High_Cheekbones",
                        "Male",
                        "Young",
                        "Smiling",
                        "Attractive",
                        "Heavy_Makeup",
                        "Wearing_Lipstick",
                        "Mouth_Slightly_Open",
                    ]
                )

                self.dist_vec = cluster_distribution.values
                cluster_distribution = cluster_distribution.sort_values(ascending=False)

                cluster_feat_vecs = []
                for j in self.image_names:
                    cluster_feat_vecs.append(image_details[j]["feat"])

                self.avg_vec = np.mean(np.array(cluster_feat_vecs), axis=0)

                self.top_attr = cluster_distribution[:5].to_string()
                self.top_attr = [j.split(" ")[0] for j in self.top_attr.split("\n")]

                return {
                    "id": int(self.id),
                    "name": str(self.id) + "-" + str(int(self.acc)),
                    "images": self.image_paths,
                    "inc_images": self.inc_images,
                    "count": int(len(self.image_names)),
                    "inc_count": int(len(self.inc_images)),
                    "conf": self.conf,
                    "top_attr": self.top_attr,
                    "avg_vec": self.avg_vec.tolist(),
                    "dist_vec": self.dist_vec.tolist(),
                    "loss": float(self.loss),
                    "leaf": str(self.leaf),
                }
            else:
                return {
                    "id": int(self.id),
                    "name": str(self.id) + "-" + str(int(self.acc)),
                    "count": int(len(self.image_names)),
                    "leaf": str(self.leaf),
                    "children": [
                        self.left.flatten(leaf_nodes) if self.left else None,
                        self.right.flatten(leaf_nodes) if self.right else None,
                    ],
                }

    num_samples = len(feature_vectors)
    if not os.path.exists(f"{args.clustering_output_dir}fc_clustering.pickle"):
        clustering = AgglomerativeClustering().fit(feature_vectors)

    else:
        clustering = pickle.load(file=open(f"{args.clustering_output_dir}fc_clustering.pickle", "rb"))

    pickle.dump(clustering, file=open(f"{args.clustering_output_dir}fc_clustering.pickle", "wb"))
    logger.info("Clustered feature vectors.")

    tree = []
    for i in range(num_samples):
        tree.append(TreeNode(i, None, None, [image_names[i]], [image_paths[i]]))

    for i in range(num_samples, 2 * num_samples - 1):
        tree.append(
            TreeNode(
                i,
                tree[clustering.children_[i - num_samples][0]],
                tree[clustering.children_[i - num_samples][1]],
                tree[clustering.children_[i - num_samples][0]].image_names
                + tree[clustering.children_[i - num_samples][1]].image_names,
                tree[clustering.children_[i - num_samples][0]].image_paths
                + tree[clustering.children_[i - num_samples][1]].image_paths,
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

    if not os.path.exists(f"{args.clustering_output_dir}fc_final_leaf_nodes.pickle"):
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

    else:
        final_leaf_nodes = pickle.load(file=open(f"{args.clustering_output_dir}fc_final_leaf_nodes.pickle", "rb"))

    pickle.dump(final_leaf_nodes, file=open(f"{args.clustering_output_dir}fc_final_leaf_nodes.pickle", "wb"))
    logger.info(f"Leaf nodes = {len(final_leaf_nodes)}.")

    test_set_df.reset_index(drop=True, inplace=True)
    model = load_model(args)

    logger.info(f"Generating JSON..")

    with open(f"{args.clustering_output_dir}fc_treeData.json", "w") as fp:
        json.dump(reversed_tree[0].flatten(final_leaf_nodes), fp)

    logger.info(f"Leaf nodes = {count}.")
    logger.info(f"JSON generated at {args.clustering_output_dir}fc_treeData.json.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bias Discovery")

    parser.add_argument("--dataroot", type=str, help="dataset folder", required=True)
    parser.add_argument("--csv_file", type=str, help="annotations", required=True)
    parser.add_argument("--model", type=str, help="model path", required=True)
    parser.add_argument("--gradcam_layer", type=str, help="penultimate layer for gradcam", required=True)
    parser.add_argument("--batch_size", type=int, help="batch size", required=True)
    parser.add_argument("--model_type", type=str, help="model type", required=True)
    parser.add_argument("--clustering_output_dir", type=str, help="output directory", required=True)

    a = parser.parse_args()

    l = logging.getLogger(__name__)
    l.setLevel(logging.INFO)

    c_handler = logging.StreamHandler(sys.stdout)
    c_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)

    l.addHandler(c_handler)

    discover(a, l)
