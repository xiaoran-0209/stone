import torch
from torch import nn
from einops.layers.torch import Rearrange

# Custom Convolution Layers
class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_cd = torch.zeros(conv_shape[0], conv_shape[1], 3 * 3, device=conv_weight.device)
        conv_weight_cd[:, :, :] = conv_weight[:, :, :]
        conv_weight_cd[:, :, 4] = conv_weight[:, :, 4] - conv_weight[:, :, :].sum(2)
        conv_weight_cd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(
            conv_weight_cd)
        return conv_weight_cd, self.conv.bias

class Conv2d_ad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_ad, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_ad = conv_weight - self.theta * conv_weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
        conv_weight_ad = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(
            conv_weight_ad)
        return conv_weight_ad, self.conv.bias

class Conv2d_hd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_hd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight  # Shape: [out_channels, in_channels, kernel_size, kernel_size]
        conv_shape = conv_weight.shape

        # Initialize conv_weight_hd with the same shape as conv_weight
        conv_weight_hd = torch.zeros_like(conv_weight)

        # Assign values to specific positions in conv_weight_hd
        conv_weight_hd[:, :, 0, 0] = conv_weight[:, :, 0, 0]  # Top-left
        conv_weight_hd[:, :, 1, 0] = conv_weight[:, :, 1, 0]  # Middle-left
        conv_weight_hd[:, :, 2, 0] = conv_weight[:, :, 2, 0]  # Bottom-left

        conv_weight_hd[:, :, 0, 2] = -conv_weight[:, :, 0, 0]  # Top-right (negative)
        conv_weight_hd[:, :, 1, 2] = -conv_weight[:, :, 1, 0]  # Middle-right (negative)
        conv_weight_hd[:, :, 2, 2] = -conv_weight[:, :, 2, 0]  # Bottom-right (negative)

        return conv_weight_hd, self.conv.bias

class Conv2d_vd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_vd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight  # Shape: [out_channels, in_channels, kernel_size, kernel_size]
        conv_shape = conv_weight.shape

        # Initialize conv_weight_vd with the same shape as conv_weight
        conv_weight_vd = torch.zeros_like(conv_weight)

        # Assign values to specific positions in conv_weight_vd
        conv_weight_vd[:, :, 0, 0] = conv_weight[:, :, 0, 0]  # Top-left
        conv_weight_vd[:, :, 0, 1] = conv_weight[:, :, 0, 1]  # Top-middle
        conv_weight_vd[:, :, 0, 2] = conv_weight[:, :, 0, 2]  # Top-right

        conv_weight_vd[:, :, 2, 0] = -conv_weight[:, :, 0, 0]  # Bottom-left (negative)
        conv_weight_vd[:, :, 2, 1] = -conv_weight[:, :, 0, 1]  # Bottom-middle (negative)
        conv_weight_vd[:, :, 2, 2] = -conv_weight[:, :, 0, 2]  # Bottom-right (negative)

        return conv_weight_vd, self.conv.bias

# DEConv Module
class DEConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias):
        super(DEConv, self).__init__()
        self.conv1_1 = Conv2d_cd(in_channels, out_channels, kernel_size, bias=bias)
        self.conv1_2 = Conv2d_hd(in_channels, out_channels, kernel_size, bias=bias)
        self.conv1_3 = Conv2d_vd(in_channels, out_channels, kernel_size, bias=bias)
        self.conv1_4 = Conv2d_ad(in_channels, out_channels, kernel_size, bias=bias)
        self.conv1_5 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)

    def forward(self, x):
        w1, b1 = self.conv1_1.get_weight()
        w2, b2 = self.conv1_2.get_weight()
        w3, b3 = self.conv1_3.get_weight()
        w4, b4 = self.conv1_4.get_weight()
        w5, b5 = self.conv1_5.weight, self.conv1_5.bias

        w = w1 + w2 + w3 + w4 + w5
        b = b1 + b2 + b3 + b4 + b5
        res = nn.functional.conv2d(input=x, weight=w, bias=b, stride=1, padding=1, groups=1)
        return res

# DEBlock Module
class DEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DEBlock, self).__init__()
        self.conv1 = DEConv(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2, bias=True)

    def forward(self, x):
        res = self.conv1(x)
        res = self.act1(res)
        res = res + x
        res = self.conv2(res)
        res = res + x
        return res

# Test DEBlock
if __name__ == '__main__':
    block = DEBlock(in_channels=16, out_channels=16, kernel_size=3).cuda()
    input = torch.rand(4, 16, 64, 64).cuda()
    output = block(input)
    print("输入尺寸:", input.size())
    print("输出尺寸:", output.size())
