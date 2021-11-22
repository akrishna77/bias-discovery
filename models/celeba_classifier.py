import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CelebAClassifier(nn.Module):
    def __init__(self):
        """
        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        """
        super(CelebAClassifier, self).__init__()

        self.model_ft = models.resnet50(pretrained=True)
        # for param in self.model_ft.parameters():
        #     param.requires_grad = False
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, 2)

        # self.model_ft = nn.Sequential(*list(self.model_ft.children())[:-1])

    def forward(self, images):
        """
        Take a batch of images and run them through the model to
        produce a score for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        """
        scores = None
        scores = self.model_ft(images)
        return scores
