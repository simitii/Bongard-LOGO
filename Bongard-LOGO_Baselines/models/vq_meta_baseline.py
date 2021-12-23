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

from .nearest_embed import NearestEmbed, NearestEmbedEMA


@register('vq-meta-baseline')
class VQMetaBaseline(nn.Module):

    def __init__(self, encoder, encoder_args={}, method='cos',
                 temp=10., temp_learnable=True, k=512):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        self.embed = NearestEmbed(k, encoder_args.out_dim)
        self.method = method

        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

    def forward(self, x_shot, x_query, **kwargs):
        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]

        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))

        z_q, _ = self.embed(z_e, weight_sg=True)
        emb, _ = self.embed(z_e.detach())

        x_shot, x_query = z_q[:len(x_shot)], z_q[-len(x_query):]
        x_shot = x_shot.view(*shot_shape, -1)
        x_query = x_query.view(*query_shape, -1)

        if self.method == 'cos':
            x_shot = x_shot.mean(dim=-2)
            x_shot = F.normalize(x_shot, dim=-1)  # [ep_per_batch, way, feature_len]
            x_query = F.normalize(x_query, dim=-1)  # [ep_per_batch, way * query, feature_len]
            metric = 'dot'
        elif self.method == 'sqr':
            x_shot = x_shot.mean(dim=-2)
            metric = 'sqr'

        logits = utils.compute_logits(
                x_query, x_shot, metric=metric, temp=self.temp)  # [ep_per_batch, way * query, way]

        if self.training:
            return (z_q, emb), logits
        else:
            return logits

