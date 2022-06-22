from __future__ import print_function, division, absolute_import

from models.correspondences.bninception import bninception
import torch.nn as nn


class Siamese(nn.Module):  # one CNN but run on two input images and return two middle activations
    def __init__(self):

        super(Siamese, self).__init__()
        self.model = bninception()

    def forward(self, x1, x2):

        fA = self.model(x1)
        fB = self.model(x2)

        return fA, fB
