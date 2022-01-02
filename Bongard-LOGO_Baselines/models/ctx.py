# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils
from .models import register

from cross_transformers_pytorch import CrossTransformer


@register('ctx')
class CTX(nn.Module):
    def __init__(self, encoder, encoder_args={}):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        self.cross_transformer = CrossTransformer(dim = 512, dim_key = 128, dim_value = 128)

    def forward(self, x_shot, x_query, **kwargs):
        img_shape = x_query.shape[-3:]
        x_query = x_query.view(-1, *img_shape)
        
        logits = self.cross_transformer(self.encoder, x_query, x_shot)

        return logits