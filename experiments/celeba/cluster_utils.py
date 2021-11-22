import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from collections import defaultdict
import seaborn as sns

from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
from gradcam.utils import visualize_cam
from sklearn.metrics import confusion_matrix, silhouette_score


@torch.no_grad()
def evaluate(model, loader, image_details=defaultdict(lambda: defaultdict(int)), verbose=False):
    """
    Compute loss on val or test data.
    """
    model.eval()

    loss = 0
    correct = 0
    incorrect = 0
    confidence = 0
    incorrect_list = []
    n_examples = 0
    incorrect_conf = 0

    for _, batch in enumerate(loader):
        data, names, target = batch["image"], batch["image_name"], batch["attributes"]
        data, target = data.cuda(), target.cuda()

        output = model(data)
        loss += F.cross_entropy(output, target, reduction="sum").data

        # predict the argmax of the log-probabilities
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        incorrect += pred.ne(target.data.view_as(pred)).cpu().sum()
        incorrect_idx = np.argwhere(np.asarray(pred.ne(target.data.view_as(pred)).cpu().flatten())).flatten()
        incorrect_list.extend([names[i] for i in incorrect_idx])

        scores = F.softmax(output, dim=1)
        temp = torch.max(scores, dim=1)
        for i in range(batch["image"].shape[0]):
            image_details[batch["image_name"][i]]["pred"] = int(temp.indices[i])
            image_details[batch["image_name"][i]]["score"] = str(float(temp.values[i]))
            confidence += float(temp.values[i])

        for i in incorrect_idx:
            incorrect_conf += scores[i].max()

        n_examples += pred.size(0)

    if len(incorrect_list) != 0:
        incorrect_conf /= len(incorrect_list)

    confidence /= n_examples
    loss /= n_examples
    acc = 100.0 * correct / n_examples

    if verbose == True:
        print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(loss, correct, n_examples, acc))
    return loss, acc, incorrect_list, incorrect_conf, confidence


def gc(model, generator, loader, class_idx=None):
    model.eval()
    img = []
    for _, batch in enumerate(loader):
        mask, _ = generator(batch["image"].cuda(), class_idx)
        _, result = visualize_cam(mask, batch["image"])
        img.append(result.permute(1, 2, 0).detach().numpy())

    return img


def get_cluster_details(model, cluster_loader, image_details=defaultdict(lambda: defaultdict(int)), verbose=True):

    loss, acc, inc, inc_conf, conf = evaluate(model, cluster_loader, image_details, verbose)

    return loss, acc, inc, inc_conf, conf


def plot_both(
    path,
    cluster_incorrect,
    model,
    generator,
    loader,
    image_details,
    cluster_distribution,
    gradcam_idx=None,
):
    image_names = []
    image_paths = []
    labels = []
    preds = []
    for _, batch in enumerate(loader):
        image_paths.extend(batch["image_path"])
        image_names.extend(batch["image_name"])
        labels.extend(batch["attributes"].cpu().numpy())

        with torch.no_grad():
            output = model(batch["image"].cuda())
            pred = output.data.max(1, keepdim=True)[1]
            preds.append(int(pred.cpu()))

    cimages = [Image.open(j) for j in image_paths]
    gradcam_images = gc(model, generator, loader, class_idx=gradcam_idx)

    fig, axs = plt.subplots(3, 2, figsize=(30, 60))

    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05)

    for axi in axs.ravel():
        axi.set_axis_off()

    if len(gradcam_images) < 24:
        r_c = (3, 8)
        f = 24
    else:
        r_c = (len(gradcam_images) // 11 + 1, 11)
        f = 22

    grid = ImageGrid(
        fig,
        (3, 2, (1, 2)),
        nrows_ncols=r_c,
        axes_pad=0.5,
    )

    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])

    for axi in grid:
        axi.set_axis_off()

    j = 0
    for ax, im in zip(grid, cimages):
        ax.imshow(im)
        if image_details != {}:
            # ax.set_title(
            #     f"{image_details[image_names[j]]['pred']}, {image_details[image_names[j]]['score'][:4]}",
            #     fontdict={"fontsize": 24},
            # )

            if image_details[image_names[j]]["label"] == 0:
                ax.set_title(f"GT: Not Smiling", fontdict={"fontsize": f})
            else:
                ax.set_title(f"GT: Smiling", fontdict={"fontsize": f})

        if image_names[j] in cluster_incorrect:
            (x0, y0, w, h) = ax.dataLim.bounds
            rect = plt.Rectangle((x0, y0), w, h, fill=False, color="red", linewidth=10)
            ax.add_patch(rect)
        j = j + 1

    grid1 = ImageGrid(
        fig,
        (3, 2, (3, 4)),
        nrows_ncols=r_c,
        axes_pad=0.5,
    )
    grid1[0].get_yaxis().set_ticks([])
    grid1[0].get_xaxis().set_ticks([])

    for axi in grid1:
        axi.set_axis_off()

    j = 0
    for ax1, im in zip(grid1, gradcam_images):
        ax1.imshow(im)
        if image_details != {}:
            # ax1.set_title(
            #     f"{image_details[image_names[j]]['pred']}, {image_details[image_names[j]]['score'][:4]}",
            #     fontdict={"fontsize": 24},
            # )

            if image_details[image_names[j]]["pred"] == 0:
                ax1.set_title(f"P: Not Smiling", fontdict={"fontsize": f})
            else:
                ax1.set_title(f"P: Smiling", fontdict={"fontsize": f})

        if image_names[j] in cluster_incorrect:
            (x0, y0, w, h) = ax1.dataLim.bounds
            rect = plt.Rectangle((x0, y0), w, h, fill=False, color="red", linewidth=13)
            ax1.add_patch(rect)
        j = j + 1

    c = cluster_distribution.take([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    c.plot.bar(fontsize=22, ax=axs[2, 0])
    axs[2, 0].set_axis_on()

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    cm = np.flipud(np.fliplr(cm))

    sns.set(font_scale=3)
    g = sns.heatmap(
        cm,
        annot=True,
        xticklabels=["Smiling", "Not Smiling"],
        yticklabels=["Smiling", "Not Smiling"],
        cmap="viridis",
        ax=axs[2, 1],
        cbar=False,
    )
    g.set_xticklabels(g.get_xmajorticklabels(), fontsize=22)
    g.set_yticklabels(g.get_ymajorticklabels(), fontsize=22)

    axs[2, 1].set_axis_on()

    plt.close(fig)
    fig.savefig(f"{path}")
    print(f"Plotted {path}")


def calculate_labels(leaf_nodes, num_samples):
    labels = np.zeros(num_samples)

    k = 0
    for i in leaf_nodes:
        for j in leaf_nodes[i]:
            labels[j] = k
        k += 1

    return labels


def get_score(leaf_nodes_dict, length, features):
    labels = calculate_labels(leaf_nodes_dict[length], len(features))
    return silhouette_score(features, labels)


def find_maximum(leaf_nodes_dict, low, high, features):
    s_low = get_score(leaf_nodes_dict, low, features)
    s_high = get_score(leaf_nodes_dict, high, features)

    if low == high:
        return s_low, low

    if high == low + 1 and s_low >= s_high:
        return s_low, low

    if high == low + 1 and s_low < s_high:
        return s_high, high

    mid = (low + high) // 2

    s_mid = get_score(leaf_nodes_dict, mid, features)
    s_midplus = get_score(leaf_nodes_dict, mid + 1, features)
    s_midminus = get_score(leaf_nodes_dict, mid - 1, features)

    if s_mid > s_midplus and s_mid > s_midminus:
        return s_mid, mid

    if s_mid > s_midplus and s_mid < s_midminus:
        return find_maximum(leaf_nodes_dict, low, mid - 1, features)
    else:
        return find_maximum(leaf_nodes_dict, mid + 1, high, features)
