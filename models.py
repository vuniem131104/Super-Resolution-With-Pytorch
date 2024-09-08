import torch
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
import math
from utils import *

class ResidualBlock(nn.Module):
    def __init__(self, kernel_size=3, in_channels=64, out_channels=64):
        super(ResidualBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                      padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                      padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return x + self.residual(x)


class SubPixelConv(nn.Module):
    def __init__(self, in_channels=64, scaling_factor=2, kernel_size=3):
        super(SubPixelConv, self).__init__()
        self.subpixel = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels * (scaling_factor ** 2),
                      kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.PixelShuffle(scaling_factor),
            nn.PReLU()
        )

    def forward(self, x):
        return self.subpixel(x)


class SRResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, small_kernel_size=3, large_kernel_size=9, n_residual_blocks=16,
                 scaling_factor=4):
        super(SRResNet, self).__init__()
        self.small_kernel_size = small_kernel_size
        self.large_kernel_size = large_kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_block = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.large_kernel_size, 1, self.large_kernel_size // 2),
            nn.PReLU()
        )
        self.all_residual_blocks = nn.Sequential(*[ResidualBlock() for _ in range(n_residual_blocks)])
        self.third_block = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, self.large_kernel_size, 1, self.large_kernel_size // 2),
            nn.BatchNorm2d(self.out_channels),
        )
        self.all_subpixel_blocks = nn.Sequential(*[SubPixelConv() for _ in range(int(math.log2(scaling_factor)))])
        self.last_block = nn.Sequential(
            nn.Conv2d(self.out_channels, self.in_channels, self.large_kernel_size, 1, self.large_kernel_size // 2),
            nn.Tanh()
        )

    def forward(self, x):
        output = self.first_block(x)
        residual_output = self.all_residual_blocks(output)
        output = output + self.third_block(residual_output)
        return self.last_block(self.all_subpixel_blocks(output))


class Generator(nn.Module):
    def __init__(self, checkpoint_path):
        super(Generator, self).__init__()
        self.generator = SRResNet()
        self.generator.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])

    def forward(self, x):
        return self.generator(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, is_norm=True):
        super(ConvBlock, self).__init__()
        if is_norm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2),
                nn.LeakyReLU(0.2)
            )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, small_kernel_size=3, large_kernel_size=9, n_second_subblock=7,
                 fc_size=1024):
        super(Discriminator, self).__init__()
        self.first_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=large_kernel_size, stride=1,
                      padding=large_kernel_size // 2),
            nn.LeakyReLU(0.2)
        )
        conv_list = []
        for i in range(n_second_subblock):
            if i % 2 == 1:
                conv_list.append(ConvBlock(out_channels, out_channels * 2))
                out_channels *= 2
            else:
                conv_list.append(ConvBlock(out_channels, out_channels, stride=2))
        self.second_block = nn.Sequential(*conv_list)
        self.third_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_channels * 6 * 6, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(fc_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.third_block(self.second_block(self.first_block(x)))




