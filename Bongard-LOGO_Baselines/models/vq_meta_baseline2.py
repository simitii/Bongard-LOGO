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

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=1, stride=1, padding=0)
        ]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)

class VQ_VAE(nn.Module):
    def __init__(self, d, k=10, bn=True, vq_coef=1, commit_coef=0.5, num_channels=3, **kwargs):
        super(VQ_VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=bn),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=bn),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=bn),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=bn),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=bn),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=bn),
            nn.BatchNorm2d(d),
        )
        self.decoder = nn.Sequential(
            ResBlock(d, d),
            nn.BatchNorm2d(d),
            ResBlock(d, d),
            nn.BatchNorm2d(d),
            ResBlock(d, d),
            nn.BatchNorm2d(d),
            ResBlock(d, d),
            nn.BatchNorm2d(d),
            ResBlock(d, d),
            nn.BatchNorm2d(d),
            ResBlock(d, d),
            nn.BatchNorm2d(d),
            nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                d, num_channels, kernel_size=4, stride=2, padding=1),
        )
        self.d = d
        self.emb = NearestEmbed(k, d)
        self.vq_coef = vq_coef
        self.commit_coef = commit_coef
        self.mse = 0
        self.vq_loss = torch.zeros(1)
        self.commit_loss = 0

        for l in self.modules():
            if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
                l.weight.detach().normal_(0, 0.02)
                torch.fmod(l.weight, 0.04)
                nn.init.constant_(l.bias, 0)

        self.encoder[-1].weight.detach().fill_(1 / 40)

        self.emb.weight.detach().normal_(0, 0.02)
        torch.fmod(self.emb.weight, 0.04)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return torch.tanh(self.decoder(x))

    def forward(self, x):
        z_e = self.encode(x)
        self.f = z_e.shape[-1]
        z_q, argmin = self.emb(z_e, weight_sg=True)
        emb, _ = self.emb(z_e.detach())
        return self.decode(z_q), z_e, emb

    def sample(self, size):
        sample = torch.randn(size, self.d, self.f,
                             self.f, requires_grad=False),
        if self.cuda():
            sample = sample.cuda()
        emb, _ = self.emb(sample)
        return self.decode(emb.view(size, self.d, self.f, self.f)).cpu()

    def loss_function(self, x, recon_x, z_e, emb):
        self.mse = F.mse_loss(recon_x, x)

        self.vq_loss = torch.mean(torch.norm((emb - z_e.detach())**2, 2, 1))
        self.commit_loss = torch.mean(torch.norm((emb.detach() - z_e)**2, 2, 1))

        return self.mse + self.vq_coef*self.vq_loss + self.commit_coef*self.commit_loss

    def latest_losses(self):
        return {'mse': self.mse, 'vq': self.vq_loss, 'commitment': self.commit_loss}


@register('vq-meta-baseline2')
class VQMetaBaseline(nn.Module):

    def __init__(self, encoder, encoder_args={}, method='cos',
                 temp=10., temp_learnable=True, hidden=128, k=1024, vq_coef=1, commit_coef=0.5):
        super().__init__()
        self.vq_vae = VQ_VAE(hidden, k=k, bn=True, vq_coef=vq_coef, commit_coef=commit_coef, num_channels=1)
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
        
        x = torch.cat([x_shot, x_query], dim=0)
        
        recon_x, z_e, emb = self.vq_vae(x)

        x_shot, x_query = emb[:len(x_shot)], emb[-len(x_query):]
        #print("x_shot.shape", x_shot.shape, "x_query.shape", x_query.shape)
        
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
        
        #print("logits.shape", logits.shape)
        
        return (x, recon_x, z_e, emb), logits

