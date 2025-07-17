from nets.resnet import resnet50
from nets.vgg import VGG16
from nets.MSCB import DEBlock
import torch
import torch.nn as nn
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        # 确保reduction后的通道数至少为1
        reduced_planes = max(1, in_planes // reduction_ratio)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, reduced_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(reduced_planes, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, skip_channels, dropout_rate=0.2):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.deab = DEBlock(out_size, out_size, kernel_size=3)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

        # 添加Dropout层
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else None

        # 使用传入的skip_channels作为LPA的输入通道数
        self.lpa = LPA(in_channel=skip_channels)

    def forward(self, inputs1, inputs2):
        inputs1 = self.lpa(inputs1)
        outputs = self.up(inputs2)
        outputs = torch.cat([inputs1, outputs], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        outputs = self.deab(outputs)

        # 应用Dropout
        if self.dropout is not None:
            outputs = self.dropout(outputs)
        return outputs


class LPA(nn.Module):
    def __init__(self, in_channel):
        super(LPA, self).__init__()
        self.ca = ChannelAttention(in_channel)
        self.sa = SpatialAttention()

    def forward(self, x):
        # 简化处理，不再分割图像
        x_attn = self.ca(x) * x
        x_attn = self.sa(x_attn) * x_attn
        return x_attn


class Unet(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, backbone='vgg', dropout_rate=0.2, reduce_layers=1):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained)
            # 调整通道数以匹配减少的层数
            in_filters = [192, 384, 768][:3]  # 固定3层
            skip_channels = [64, 128, 256][:3]  # 固定3层
            feat_channels = 256  # 第三层特征通道数
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)
            in_filters = [192, 512, 1024][:3]
            skip_channels = [256, 512, 1024][:3]
            feat_channels = 1024
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))

        out_filters = [64, 128, 256][:3]

        # 在瓶颈层添加LPA
        self.bottleneck_lpa = LPA(in_channel=feat_channels)

        # upsampling - 根据reduce_layers减少上采样层数
        if reduce_layers == 1:
            # 减少1层，保留3个上采样层
            self.up_concat3 = unetUp(in_filters[2], out_filters[2], skip_channels[2], dropout_rate)
            self.up_concat2 = unetUp(in_filters[1], out_filters[1], skip_channels[1], dropout_rate)
            self.up_concat1 = unetUp(in_filters[0], out_filters[0], skip_channels[0], dropout_rate)
        elif reduce_layers == 2:
            # 减少2层，保留2个上采样层
            self.up_concat2 = unetUp(in_filters[1], out_filters[1], skip_channels[1], dropout_rate)
            self.up_concat1 = unetUp(in_filters[0], out_filters[0], skip_channels[0], dropout_rate)
        else:
            # 完整结构
            in_filters = [192, 384, 768, 1024]
            skip_channels = [64, 128, 256, 512]
            out_filters = [64, 128, 256, 512]
            feat_channels = 512 if backbone == 'vgg' else 2048

            self.up_concat4 = unetUp(in_filters[3], out_filters[3], skip_channels[3], dropout_rate)
            self.up_concat3 = unetUp(in_filters[2], out_filters[2], skip_channels[2], dropout_rate)
            self.up_concat2 = unetUp(in_filters[1], out_filters[1], skip_channels[1], dropout_rate)
            self.up_concat1 = unetUp(in_filters[0], out_filters[0], skip_channels[0], dropout_rate)

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)
        self.backbone = backbone
        self.reduce_layers = reduce_layers

    def forward(self, inputs):
        if self.backbone == "vgg":
            features = self.vgg.forward(inputs)
            # 确保获取足够多的特征图
            if len(features) < 4:
                raise ValueError("VGG backbone should return at least 4 feature maps")
        elif self.backbone == "resnet50":
            features = self.resnet.forward(inputs)
            # 确保获取足够多的特征图
            if len(features) < 4:
                raise ValueError("ResNet50 backbone should return at least 4 feature maps")

        # # 打印特征图信息用于调试
        # for i, feat in enumerate(features):
        #     print(f"Feature {i} shape: {feat.shape if feat is not None else 'None'}")

        # 根据减少的层数选择特征图
        if self.reduce_layers == 1:
            # 使用前3个特征图(跳过最深的一个)
            feat1, feat2, feat3 = features[:3]
            # 确保特征图不为None
            if feat3 is None:
                raise ValueError("feat3 is None, check backbone implementation")
            feat3 = self.bottleneck_lpa(feat3)
            up2 = self.up_concat2(feat2, feat3)
            up1 = self.up_concat1(feat1, up2)
        elif self.reduce_layers == 2:
            # 使用前2个特征图(跳过最深的两个)
            feat1, feat2 = features[:2]
            # 确保特征图不为None
            if feat2 is None:
                raise ValueError("feat2 is None, check backbone implementation")
            feat2 = self.bottleneck_lpa(feat2)
            up1 = self.up_concat1(feat1, feat2)
        else:
            # 完整结构
            feat1, feat2, feat3, feat4, feat5 = features[:5]
            # 确保特征图不为None
            if feat5 is None:
                raise ValueError("feat5 is None, check backbone implementation")
            feat5 = self.bottleneck_lpa(feat5)
            up4 = self.up_concat4(feat4, feat5)
            up3 = self.up_concat3(feat3, up4)
            up2 = self.up_concat2(feat2, up3)
            up1 = self.up_concat1(feat1, up2)

        if self.up_conv is not None:
            up1 = self.up_conv(up1)

        final = self.final(up1)
        return final

    # 保持原有的freeze_backbone和unfreeze_backbone方法不变
    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True