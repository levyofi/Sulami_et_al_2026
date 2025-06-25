import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F


class PixelCNN(nn.Module):

    def __init__(self, conv_channels=16, task='regression'):
        super().__init__()
        self.output_dim = 1 if task == 'regression' else 210
        self.conv_layers = nn.Sequential(
            nn.Conv2d(5, conv_channels, 3),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(conv_channels, 2 * conv_channels, 3),
            nn.BatchNorm2d(2 * conv_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(2 * conv_channels, 4 * conv_channels, 3),
            nn.BatchNorm2d(4 * conv_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(4 * conv_channels * 2 * 2 + 8, 1),
            nn.ReLU(inplace=True),
            #             nn.Linear(4*conv_channels, conv_channels),
            #             nn.ReLU(inplace=True),
            #             nn.Linear(conv_channels, 1),
            #             nn.ReLU(inplace=True)
        )
        self.model_name = 'pixel_cnn'

    def forward(self, x, met_data):
        x_1 = self.conv_layers(x)
        x_2 = x_1.view(x_1.size(0), -1)
        x_3 = torch.cat([x_2, met_data], dim=1)
        x_4 = self.fc_layers(x_3)
        x_5 = x_4.view(x_4.size(0))
        return x_5

class ResNet(PixelCNN):

    def __init__(self, spatial_size, pretrained=False, freeze_grads=False, method='modified'):
        super().__init__()
        self.conv_layers = self.modified_resnet_conv(pretrained, freeze_grads, method)
        if spatial_size<33:
            linear_size = 512
        elif spatial_size<65:
            linear_size = 512*2*2
        else:
            linear_size = 512*3*3
        self.fc_layers = nn.Sequential(
            nn.Linear(linear_size + 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.output_dim)
        )
        self.model_name = self.get_model_name(pretrained, freeze_grads, method)

    def modified_resnet_conv(self, pretrained, freeze_grads, method):
        weights = 'ResNet18_Weights.DEFAULT' if pretrained else None
        resnet = models.resnet18(weights=weights)
        if freeze_grads:
            for param in resnet.parameters():
                param.requires_grad = False
        if method == 'modified':
            resnet.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif method == 'adapter':
            resnet = nn.Sequential(nn.Conv2d(5, 3, 1), *list(resnet.children()))
        resnet_conv = nn.Sequential(*list(resnet.children())[:-2])#,
                                 #   torch.nn.AdaptiveAvgPool2d((1, 1)))
        return resnet_conv

    def forward(self, x, met_data):
        x_1 = self.conv_layers(x)
        x_2 = x_1.view(x_1.size(0), -1)
        x_3 = torch.cat([x_2, met_data], dim=1)
        x_4 = self.fc_layers(x_3)
        x_5 = x_4.view(x_4.size(0))
        return x_5

    def get_model_name(self, pretrained, freeze_grads, method):
        structure = {'pretrained': '_pretrained' if pretrained else '',
                     'first_layer': 'modified_conv1' if method == 'modified' else 'conv_adapter',
                     'grads': 'frozen' if freeze_grads else 'unfrozen'}
        return f'pixel_resnet18{structure["pretrained"]}_{structure["first_layer"]}_{structure["grads"]}_grads'

