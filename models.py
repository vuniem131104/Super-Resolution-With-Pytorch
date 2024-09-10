import torch
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
import math
from utils import *
import torchvision

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
    def __init__(self, in_channels=3, out_channels=64, small_kernel_size=3, large_kernel_size=9, n_residual_blocks=16,
                 scaling_factor=4):
        super(Generator, self).__init__()
        self.generator = SRResNet(in_channels=in_channels, out_channels=out_channels, small_kernel_size=small_kernel_size, large_kernel_size=large_kernel_size, n_residual_blocks=n_residual_blocks,
                 scaling_factor=scaling_factor)

    def init_with_checkpoint(self, srresnet_ckpt):
        srresnet = torch.load(srresnet_ckpt, map_location='cpu')
        self.generator.load_state_dict(srresnet)

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
            nn.Linear(fc_size, 1)
        )

    def forward(self, x):
        return self.third_block(self.second_block(self.first_block(x)))
    
class TruncatedVGG19(nn.Module):
    """
    A truncated VGG19 network, such that its output is the 'feature map obtained by the j-th convolution (after activation)
    before the i-th maxpooling layer within the VGG19 network', as defined in the paper.

    Used to calculate the MSE loss in this VGG feature-space, i.e. the VGG loss.
    """

    def __init__(self, i, j):
        """
        :param i: the index i in the definition above
        :param j: the index j in the definition above
        """
        super(TruncatedVGG19, self).__init__()

        # Load the pre-trained VGG19 available in torchvision
        vgg19 = torchvision.models.vgg19(weights='DEFAULT')

        maxpool_counter = 0
        conv_counter = 0
        truncate_at = 0
        # Iterate through the convolutional section ("features") of the VGG19
        for layer in vgg19.features.children():
            truncate_at += 1

            # Count the number of maxpool layers and the convolutional layers after each maxpool
            if isinstance(layer, nn.Conv2d):
                conv_counter += 1
            if isinstance(layer, nn.MaxPool2d):
                maxpool_counter += 1
                conv_counter = 0

            # Break if we reach the jth convolution after the (i - 1)th maxpool
            if maxpool_counter == i - 1 and conv_counter == j:
                break

        # Check if conditions were satisfied
        assert maxpool_counter == i - 1 and conv_counter == j, "One or both of i=%d and j=%d are not valid choices for the VGG19!" % (
            i, j)

        # Truncate to the jth convolution (+ activation) before the ith maxpool layer
        self.truncated_vgg19 = nn.Sequential(*list(vgg19.features.children())[:truncate_at + 1])

    def forward(self, input):
        output = self.truncated_vgg19(input)  # (N, feature_map_channels, feature_map_w, feature_map_h)

        return output