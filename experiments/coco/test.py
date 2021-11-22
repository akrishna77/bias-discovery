"""
Script with Pytorch's dataloader class
"""

import os
import glob
from typing import Dict, List, Tuple

import torch
import numpy as np
import torchvision
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
from experiments.coco.utils import add_flops_counting_methods
import experiments.coco.models as models
from experiments.coco.cluster_utils import print_annots


class ImageLoader(data.Dataset):
    """Class for data loading"""

    train_folder = "train"
    test_folder = "test"

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: torchvision.transforms.Compose = None,
    ):
        """Initialize the dataloader and set `curr_folder` for the corresponding data split.

        Args:
            root_dir: the dir path which contains the train and test folder
            split: 'test' or 'train' split
            transform: the composed transforms to be applied to the data
        """
        self.root = os.path.expanduser(root_dir)
        self.transform = transform
        self.split = split

        if split == "train":
            self.curr_folder = os.path.join(root_dir, self.train_folder)
        elif split == "test":
            self.curr_folder = os.path.join(root_dir, self.test_folder)

        self.class_dict = self.get_classes()
        self.dataset = self.load_imagepaths_with_labels(self.class_dict)

    def load_imagepaths_with_labels(self, class_labels: Dict[str, int]) -> List[Tuple[str, int]]:
        """Fetches all (image path,label) pairs in the dataset.

        Args:
            class_labels: the class labels dictionary, with keys being the classes in this dataset
        Returns:
            list[(filepath, int)]: a list of filepaths and their class indices
        """

        img_paths = []  # a list of (filename, class index)

        for class_name, class_idx in class_labels.items():
            img_dir = os.path.join(self.curr_folder, class_name, "*.jpeg")

            files = glob.glob(img_dir)

            img_paths += [(f, class_idx) for f in files]

        return img_paths

    def get_classes(self) -> Dict[str, int]:
        """Get the classes (which are folder names in self.curr_folder)

        NOTE: Please make sure that your classes are sorted in alphabetical order
        i.e. if your classes are ['apple', 'giraffe', 'elephant', 'cat'], the
        class labels dictionary should be:
        {"apple": 0, "cat": 1, "elephant": 2, "giraffe":3}

        If you fail to do so, you will most likely fail the accuracy
        tests on Gradescope

        Returns:
            Dict of class names (string) to integer labels
        """

        classes = dict()

        classes_list = sorted([d.name for d in os.scandir(self.curr_folder) if d.is_dir()])

        classes = {classes_list[i]: i for i in range(len(classes_list))}
        return classes

    def load_img_from_path(self, path: str) -> Image:
        """Loads an image as grayscale (using Pillow).

        Note: do not normalize the image to [0,1]

        Args:
            path: the file path to where the image is located on disk
        Returns:
            image: grayscale image with values in [0,255] loaded using pillow
                Note: Use 'L' flag while converting using Pillow's function.
        """

        img = None

        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")

        return img

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Fetches the item (image, label) at a given index.

        Note: Do not forget to apply the transforms, if they exist

        Hint:
        1) get info from self.dataset
        2) use load_img_from_path
        3) apply transforms if valid

        Args:
            index: Index
        Returns:
            img: image of shape (H,W)
            class_idx: index of the ground truth class for this image
        """
        img = None
        class_idx = None

        filename, class_idx = self.dataset[index]
        # load the image and apply the transforms
        img = self.load_img_from_path(filename)

        if self.transform is not None:
            img = self.transform(img)

        return img, filename, class_idx

    def __len__(self) -> int:
        """Returns the number of items in the dataset.

        Returns:
            l: length of the dataset
        """

        l = len(self.dataset)

        return l


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

val_data_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

dataloader_args = {"num_workers": 4} if torch.cuda.is_available() else {}

val_dataset = ImageLoader("./data", split="test", transform=val_data_transforms)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, **dataloader_args)

model = models.__dict__["densenet201"](num_classes=80)
model = add_flops_counting_methods(model)
model.eval()

model = torch.nn.DataParallel(model).cuda()
checkpoint = torch.load("/srv/share/akrishna/bias-discovery/experiments/coco/coco_densenet201.pth.tar")
model.load_state_dict(checkpoint["state_dict"], strict=False)

for (x, fname, _) in val_loader:
    x = x.cuda()

    n = x.shape[0]
    with torch.no_grad():
        output = model(x)
        pred = output.data.gt(0.0).long()

        p = print_annots(pred)

        print(fname, p)
