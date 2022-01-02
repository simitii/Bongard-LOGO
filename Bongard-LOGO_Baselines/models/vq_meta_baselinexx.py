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


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False)


def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 1, bias=False)


def norm_layer(planes):
    return nn.BatchNorm2d(planes)


class Block(nn.Module):

    def __init__(self, inplanes, planes, downsample):
        super().__init__()

        self.relu = nn.LeakyReLU(0.1)

        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = norm_layer(planes)

        self.downsample = downsample

        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        out = self.maxpool(out)

        return out


class ResNet12(nn.Module):

    def __init__(self, channels, out_dim=512, inplanes=1):
        super().__init__()

        self.inplanes = inplanes  # 1 with 'L' or 3 with 'RGB' or 7 with non-meta-learning

        self.layer1 = self._make_layer(channels[0])
        self.layer2 = self._make_layer(channels[1])
        self.layer3 = self._make_layer(channels[2])
        self.layer4 = self._make_layer(channels[3])
        self.layer5 = self._make_layer(channels[4])
        if out_dim != 512:
            self.fc = nn.Linear(channels[-1], out_dim)

        self.out_dim = out_dim

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes):
        downsample = nn.Sequential(
            conv1x1(self.inplanes, planes),
            norm_layer(planes),
        )
        block = Block(self.inplanes, planes, downsample)
        self.inplanes = planes
        return block

    def forward(self, x):
#        print(x.shape)
        x = self.layer1(x)
#        print(x.shape)
        x = self.layer2(x)
#        print(x.shape)
        x = self.layer3(x)
#        print(x.shape)
        x = self.layer4(x)
#        print(x.shape)
        x = self.layer5(x)
#        print(x.shape)
        x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
#        print(x.shape)
        if self.out_dim != 512:
            x = self.fc(x)
#            print(x.shape)
        return x

class Decoder(nn.Module):
	def __init__(self, zsize):
		super(Decoder,self).__init__()
		self.relu = nn.LeakyReLU(0.1)
		self.dfc = nn.Linear(zsize, 512)
		self.bn = nn.BatchNorm1d(512)
		self.upsample=nn.Upsample(scale_factor=2)
		self.dconv7 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
		self.dconv6 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
		self.bn6 = norm_layer(512)
		self.dconv5 = conv3x3(512, 256)
		self.bn5 = norm_layer(256)
		self.dconv4 = conv3x3(256, 128)
		self.bn4 = norm_layer(128)
		self.dconv3 = conv3x3(128, 64)
		self.bn3 = norm_layer(64)
		self.dconv2 = conv3x3(64, 32)
		self.bn2 = norm_layer(32)
		self.dconv1 = conv3x3(32, 1)

	def forward(self,x):#,i1,i2,i3):        
		x = self.dfc(x)
		x = F.relu(self.bn(x))
#		print(x.shape)
        
		x = x.view(-1,512,1,1)
 
		x = self.upsample(x)
		x = self.relu(self.dconv7(x))
#		print(x.shape)

		x = self.upsample(x)
		x = self.relu(self.dconv6(x))
		x = self.bn6(x)
#		print(x.shape)
        
		x = self.upsample(x)
		x = self.relu(self.dconv5(x))
		x = self.bn5(x)
#		print(x.shape)
        
		x = self.upsample(x)        
		x = self.relu(self.dconv4(x))
		x = self.bn4(x)
#		print(x.shape)
        
		x = self.upsample(x)
		x = self.relu(self.dconv3(x))
		x = self.bn3(x)
#		print(x.shape)
        
		x = self.upsample(x)
		x = self.relu(self.dconv2(x))
		x = self.bn2(x)
#		print(x.shape)
        
		x = self.upsample(x)
		x = self.dconv1(x)
#		print(x.shape)

		return x

def resnet12(out_dim=512, inplanes=1, reduce_factor=2):
    assert 1 <= reduce_factor <= 32
    featmaps = [32 * 2 // reduce_factor, 64 * 2 // reduce_factor, 128 * 2 // reduce_factor,
                min(256 * 2 // reduce_factor, 256), min(512 * 2 // reduce_factor, 512)]
    out_dim = min(out_dim * 2 // reduce_factor, 128)
    print('resnet12 featmaps: {}, and out_dim: {}'.format(featmaps, out_dim))
    return ResNet12(featmaps, out_dim, inplanes)  # when image_res=512
    # return ResNet12([64, 128, 256, 512], out_dim)  # when image_res=256


class VQ_VAE(nn.Module):
    def __init__(self, d, k=4096, vq_coef=1, commit_coef=0.5):
        super(VQ_VAE,self).__init__()
        self.encoder = resnet12()
        self.decoder = Decoder(d)
        self.emb = NearestEmbedEMA(k, d)
        self.vq_coef = vq_coef
        self.commit_coef = commit_coef
        self.d = d
        self.k = k
        self.mse = 0
        self.vq_loss = torch.zeros(1)
        self.commit_loss = 0
    
    def forward(self, x):
        z_e = self.encoder(x)
        d_e = self.decoder(z_e)
        self.f = z_e.shape[-1]
        z_q, argmin = self.emb(z_e)
        emb, _ = self.emb(z_e.detach())
        return self.decoder(z_q), z_e, emb

    def sample(self, size):
        sample = torch.randn(size, self.d, self.f,
                             self.f, requires_grad=False),
        if self.cuda():
            sample = sample.cuda()
        emb, _ = self.emb(sample)
        return self.decoder(emb.view(size, self.d, self.f, self.f)).cpu()
    
    def loss_function(self, x, recon_x, z_e, emb):
        self.mse = F.mse_loss(recon_x, x)

        self.vq_loss = torch.mean(torch.norm((emb - z_e.detach())**2, 2, 1))
        self.commit_loss = torch.mean(torch.norm((emb.detach() - z_e)**2, 2, 1))

        return self.mse + self.vq_coef*self.vq_loss + self.commit_coef*self.commit_loss

    def latest_losses(self):
        return {'mse': self.mse, 'vq': self.vq_loss, 'commitment': self.commit_loss}
    
class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        #inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs#.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized, perplexity, encodings

    
@register('vq-meta-baseline3')
class VQMetaBaseline(nn.Module):

    def __init__(self, encoder, encoder_args={}, method='cos',
                 temp=10., temp_learnable=True, hidden=128, k=4096, vq_coef=1, commit_coef=0.25, decay=0.99):
        super().__init__()
        self.encoder = resnet12()
        self.decoder = Decoder(hidden)
        self.vq_vae = VectorQuantizerEMA(k, hidden, commit_coef, decay)
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
        
        z = self.encoder(x)
        loss, quantized, perplexity, _ = self.vq_vae(z)
        x_recon = self.decoder(quantized)

        x_shot, x_query = quantized[:len(x_shot)], quantized[-len(x_query):]
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
        
        return (x, x_recon, z, quantized, loss, perplexity), logits