import torch
from torch import nn as nn
import torch.nn.functional as F
from .unet_5_contextual_bneck3 import DoubleConv,DoubleDeconv


class Unet3ContextualBneck(torch.nn.Module):
    def __init__(self, in_channels, out_channels, embedding_size, hc1=32, hb1=16, hc2=256, stride=2, norm="instance_norm"):
        super(Unet3ContextualBneck, self).__init__()

        self.hc1 = hc1
        self.hb1 = hb1
        self.hc2 = hc2

        self.embedding_size = embedding_size

        # inchannels, outchannels, kernel size
        self.conv1 = DoubleConv(in_channels, hc1, 3, stride=stride, padding=1)
        self.conv2 = DoubleConv(hc1, hc1, 3, stride=stride, padding=1)
        self.conv3 = DoubleConv(hc1, hc1, 3, stride=stride, padding=1)

        self.deconv1 = DoubleDeconv(hc1, hc1, 3, stride=stride, padding=1)
        self.deconv2 = DoubleDeconv(hc1 + hb1, hc2, 3, stride=stride, padding=1)
        self.deconv3 = nn.ConvTranspose2d(hb1 + hc2, out_channels, 3, stride=stride, padding=1)

        self.act = nn.LeakyReLU()

        if norm == "instance_norm":
            self.norm1 = nn.InstanceNorm2d(hc1)
            self.norm2 = nn.InstanceNorm2d(hc1)
            self.dnorm1 = nn.InstanceNorm2d(hc1)
            self.dnorm2 = nn.InstanceNorm2d(hc1)
            self.fnorm1 = nn.InstanceNorm2d(hb1)
            self.fnorm2 = nn.InstanceNorm2d(hb1)
        elif norm == "batch_norm":
            self.norm1 = nn.BatchNorm2d(hc1)
            self.norm2 = nn.BatchNorm2d(hc1)
            self.dnorm1 = nn.BatchNorm2d(hc1)
            self.dnorm2 = nn.BatchNorm2d(hc1)
            self.fnorm1 = nn.BatchNorm2d(hb1)
            self.fnorm2 = nn.BatchNorm2d(hb1)

        self.lang15 = nn.Linear(self.embedding_size, hc1 * hb1)
        self.lang24 = nn.Linear(self.embedding_size, hc1 * hb1)
        self.lang3 = nn.Linear(self.embedding_size, hc1 * hc1)

    def init_weights(self):
        self.conv1.init_weights()
        self.conv2.init_weights()
        self.conv3.init_weights()
        self.deconv1.init_weights()
        self.deconv2.init_weights()

    def forward(self, input, embedding):
        x1 = self.norm1(self.act(self.conv1(input)))
        x2 = self.norm2(self.act(self.conv2(x1)))
        x3 = self.act(self.conv3(x2))

        if embedding is not None:
            embedding = F.normalize(embedding, p=2, dim=1)

            # These conv filters are different for each element in the batch, but the functional convolution
            # operator assumes the same filters across the batch.
            # TODO: Verify if slicing like this is a terrible idea for performance
            x1f = torch.zeros_like(x1[:,0:self.hb1,:,:].data)
            x2f = torch.zeros_like(x2[:,0:self.hb1,:,:].data)
            x3f = torch.zeros_like(x3.data)

            batch_size = embedding.size(0)
            for i in range(batch_size):
                lf1 = F.normalize(self.lang15(embedding[i:i+1])).view([self.hb1, self.hc1, 1, 1])
                lf2 = F.normalize(self.lang24(embedding[i:i+1])).view([self.hb1, self.hc1, 1, 1])
                lf3 = F.normalize(self.lang3(embedding[i:i+1])).view([self.hc1, self.hc1, 1, 1])

                x1f[i:i+1] = F.conv2d(x1[i:i+1], lf1)
                x2f[i:i+1] = F.conv2d(x2[i:i+1], lf2)
                x3f[i:i+1] = F.conv2d(x3[i:i+1], lf3)

            x1 = self.fnorm1(x1f)
            x2 = self.fnorm2(x2f)
            x3 = x3f

        x4 = self.dnorm1(self.act(self.deconv1(x3, output_size=x2.size())))
        x24 = torch.cat([x2, x4], 1)
        x5 = self.dnorm2(self.act(self.deconv2(x24, output_size=x1.size())))
        x15 = torch.cat([x1, x5], 1)
        out = self.deconv3(x15, output_size=input.size())

        return out
