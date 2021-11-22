import os
from collections import defaultdict

import torch
import torch.nn as nn

import scipy
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pycocotools.coco import COCO
from gradcam.utils import visualize_cam

from experiments.coco.utils import AverageMeter
from sklearn.metrics import multilabel_confusion_matrix

COCO_ROOT = "/srv/share/datasets/coco"

coco = COCO(os.path.join(f"{COCO_ROOT}/annotations/", "instances_train2017.json"))


@torch.no_grad()
def evaluate(model, loader, category, image_details=defaultdict(lambda: defaultdict(int)), verbose=True):
    """
    Compute loss on val or test data.
    """
    criterion = nn.BCEWithLogitsLoss().cuda()
    losses = AverageMeter()
    inc_images = []
    tp, fp, fn, tn, count = 0, 0, 0, 0, 0
    inc_count = 0

    model.eval()

    for _, (index, image, target) in enumerate(loader):
        target = target.cuda(non_blocking=True)
        target = target.max(dim=1)[0]
        # compute output
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, target.float())

        # measure accuracy and record loss
        pred = output.data.gt(0.0).long()

        tp += (pred + target).eq(2).sum(dim=0)
        fp += (pred - target).eq(1).sum(dim=0)
        fn += (pred - target).eq(-1).sum(dim=0)
        tn += (pred + target).eq(0).sum(dim=0)

        p_o = tp.sum().float() / (tp + fp).sum().float() * 100.0
        r_o = tp.sum().float() / (tp + fn).sum().float() * 100.0
        f_o = 2 * p_o * r_o / (p_o + r_o)

        p_c = [float(tp[i].float() / (tp[i] + fp[i]).float()) * 100.0 if tp[i] > 0 else 0.0 for i in range(len(tp))]
        r_c = [float(tp[i].float() / (tp[i] + fn[i]).float()) * 100.0 if tp[i] > 0 else 0.0 for i in range(len(tp))]
        f_c = [2 * p_c[i] * r_c[i] / (p_c[i] + r_c[i]) if tp[i] > 0 else 0.0 for i in range(len(tp))]

        mean_f_c = sum(f_c) / len(f_c)

        for j in range(image.shape[0]):
            image_details[int(index[j])]["pred"] = pred[j].cpu().detach().numpy()

            if target[j][category] == 0:
                inc_count += 1
                inc_images.append(int(index[j]))

            # if image_details[int(index[j])]["pred"][category] == 0:
            #     inc_count += 1
            #     inc_images.append(int(index[j]))

        count += image.size(0)

        losses.update(float(loss.item()), image.size(0))

    if count != 0:
        acc = 100 * (count - inc_count) / count
    else:
        acc = -999
    if verbose:
        print(" * Loss {:.2f}, Accuracy {:.2f}".format(losses.avg, acc))

    return losses.avg, f_o, mean_f_c, acc, inc_images


def init_helpers():
    cat_ids = coco.getCatIds()
    categories = coco.loadCats(cat_ids)
    categories.sort(key=lambda x: x["id"])

    category_id_to_coco_label = {category["id"]: i for i, category in enumerate(categories)}
    coco_label_to_category_id = {v: k for k, v in category_id_to_coco_label.items()}

    cat_classes = {i["id"]: i["name"] for i in categories}
    cat_ids = {v: k for k, v in cat_classes.items()}

    return coco_label_to_category_id, category_id_to_coco_label, cat_classes, cat_ids


def print_annots(annots):
    coco_label_to_category_id, _, cat_classes, _ = init_helpers()

    y = np.where(annots.cpu() == 1)[1]
    return [cat_classes[coco_label_to_category_id[int(i)]] for i in y]


def gc(model, generator, loader, class_idx=None):
    img = []
    pred_classes = []
    gt_classes = []
    masks = []
    preds = np.empty((0, 80))
    gts = np.empty((0, 80))

    for _, (_, image, target) in enumerate(loader):
        target = target.cuda(non_blocking=True)
        target = target.max(dim=1)[0]

        gts = np.vstack((gts, target.cpu().detach().numpy()))

        with torch.no_grad():
            output = model(image)

        pred = output.data.gt(0.0).long()
        preds = np.vstack((preds, pred.cpu().detach().numpy()))

        p = print_annots(pred)
        gt = print_annots(target)
        mask, _ = generator(image.cuda(), class_idx)
        _, result = visualize_cam(mask, image)

        img.append(result.permute(1, 2, 0).detach().numpy())
        pred_classes.append(p)
        gt_classes.append(gt)
        masks.append(mask[0])

    return img, preds, gts, pred_classes, gt_classes, torch.cat(masks, dim=0)


def get_cluster_details(model, loader, category):
    loss, f1, f1_c, acc, inc_images = evaluate(model, loader, category)
    return loss, f1, f1_c, acc, inc_images


def get_coocc(tags, window_size=2):
    vocabulary = {}
    data = []
    row = []
    col = []

    for tag in tags:
        for pos, token in enumerate(tag):
            i = vocabulary.setdefault(token, len(vocabulary))
            start = max(0, pos - window_size)
            end = min(len(tag), pos + window_size + 1)
            for pos2 in range(start, end):
                if pos2 == pos:
                    continue
                j = vocabulary.setdefault(tag[pos2], len(vocabulary))
                data.append(1)
                row.append(i)
                col.append(j)

    if len(row) == 0 or len(col) == 0:
        return None

    cooccurrence_matrix_sparse = scipy.sparse.coo_matrix((data, (row, col)))

    try:
        df_co_occ = pd.DataFrame(
            cooccurrence_matrix_sparse.todense(), index=vocabulary.keys(), columns=vocabulary.keys()
        )
    except ValueError:
        return None

    df_co_occ = df_co_occ.sort_index()[sorted(vocabulary.keys())]

    return df_co_occ


def plot_both(cluster_id, path, model, loader, generator, cluster_incorrect, gradcam_idx):

    print(f"Plotting cluster {cluster_id}")

    pred_classes = []
    for _, (_, image, target) in enumerate(loader):
        target = target.cuda(non_blocking=True)
        target = target.max(dim=1)[0]

        with torch.no_grad():
            output = model(image)

        pred = output.data.gt(0.0).long()
        p = print_annots(pred)
        pred_classes.append(p)

    df_co_occ = get_coocc(pred_classes)
    coco_label_to_category_id, _, cat_classes, _ = init_helpers()

    if df_co_occ is not None:
        column = cat_classes[coco_label_to_category_id[gradcam_idx]]
        m = df_co_occ.loc[df_co_occ[column].idxmax()][column]
        row = df_co_occ[column].idxmax()

    gradcam_images1, preds, gts, pred_classes, gt_classes, _ = gc(model, generator, loader, class_idx=gradcam_idx)

    images = []
    image_names = []
    for _, (index, image, _) in enumerate(loader):
        images.append(image[0])
        image_names.append(index)

    images = [i.permute(1, 2, 0) for i in images]
    gradcam_images1 = [i for i in gradcam_images1]

    fig, axs = plt.subplots(3, 2, figsize=(30, 30))

    for axi in axs.ravel():
        axi.set_axis_off()

    if len(gradcam_images1) < 24:
        r_c = (3, 8)
    else:
        r_c = (5, 10)

    grid = ImageGrid(
        fig,
        (3, 2, (1, 2)),
        nrows_ncols=r_c,
        axes_pad=0.4,
    )
    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])

    for axi in grid:
        axi.set_axis_off()

    j = 0
    for ax1, im in zip(grid, images):
        ax1.imshow(im)
        ax1.set_title(f"GT: {gt_classes[j]}", fontdict={"fontsize": 16})
        if image_names[j] in cluster_incorrect:
            (x0, y0, w, h) = ax1.dataLim.bounds
            rect = plt.Rectangle((x0, y0), w, h, fill=False, color="red", linewidth=10)
            ax1.add_patch(rect)
        j = j + 1

    grid1 = ImageGrid(
        fig,
        (3, 2, (3, 4)),
        nrows_ncols=r_c,
        axes_pad=0.4,
    )

    grid1[0].get_yaxis().set_ticks([])
    grid1[0].get_xaxis().set_ticks([])

    for axi in grid1:
        axi.set_axis_off()

    j = 0
    for ax1, im in zip(grid1, gradcam_images1):
        ax1.imshow(im)
        ax1.set_title(f"P: {pred_classes[j]}", fontdict={"fontsize": 16})
        if image_names[j] in cluster_incorrect:
            (x0, y0, w, h) = ax1.dataLim.bounds
            rect = plt.Rectangle((x0, y0), w, h, fill=False, color="red", linewidth=10)
            ax1.add_patch(rect)
        j = j + 1

    cm = multilabel_confusion_matrix(gts, preds, labels=[gradcam_idx])[0]

    cm = np.flipud(np.fliplr(cm))

    sns.set(font_scale=3)
    g = sns.heatmap(
        cm,
        annot=True,
        xticklabels=[
            cat_classes[coco_label_to_category_id[gradcam_idx]],
            "not " + cat_classes[coco_label_to_category_id[gradcam_idx]],
        ],
        yticklabels=[
            cat_classes[coco_label_to_category_id[gradcam_idx]],
            "not " + cat_classes[coco_label_to_category_id[gradcam_idx]],
        ],
        cmap="viridis",
        ax=axs[2, 0],
        cbar=False,
    )
    g.set_xticklabels(g.get_xmajorticklabels(), fontsize=24)
    g.set_yticklabels(g.get_ymajorticklabels(), fontsize=24)

    axs[2, 0].set_axis_on()

    if df_co_occ is not None:
        sns.set(font_scale=3)
        g = sns.heatmap(
            df_co_occ,
            cmap="viridis",
            annot=True,
            linewidths=0.75,
            ax=axs[2, 1],
            fmt="g",
            cbar=False,
        )
        g.set_xticklabels(g.get_xmajorticklabels(), fontsize=24)
        g.set_yticklabels(g.get_ymajorticklabels(), fontsize=24)

        axs[2, 1].set_axis_on()

    fig.savefig(f"{path}")
    print(f"Plotted {path}")
    plt.close()
