import torch
import torch.nn as nn
from torchvision import models

from abc import abstractmethod,ABCMeta


class BaseCNN(nn.Module):
    """ Pretrained PyTorch CNNs expecting unnormalized RGB images """
    def __init__(self):
        super(BaseCNN, self).__init__()

        self.mean = (
            torch.FloatTensor([0.485, 0.456, 0.406])
            .unsqueeze(0)
            .unsqueeze(2)
            .unsqueeze(3)
        )
        self.std = (
            torch.FloatTensor([0.229, 0.224, 0.225])
            .unsqueeze(0)
            .unsqueeze(2)
            .unsqueeze(3)
        )

    def normalize(self, image):
        if self.mean.device != image.device:
            self.mean = self.mean.to(image.device)
            self.std = self.std.to(image.device)
        output = ((image / 255.0) - self.mean) / self.std
        return output


class ResNet(BaseCNN):
    """
    ImageNet pretrained ResNet to generate feature map, with optional finetuning of upper layers
    """
    __metaclass__ = ABCMeta

    def __init__(self, finetune=True):
        """
            finetune (bool, optional): Defaults to True. Set to True for finetuning of upper
                                       layers. If finetune=False, requires_grad is set to False.
        """

        super(ResNet, self).__init__()

        mv = self.model_version()(pretrained=True)
        self.conv1 = mv.conv1
        self.bn1 = mv.bn1
        self.relu = mv.relu
        self.maxpool = mv.maxpool
        self.layer1 = mv.layer1
        self.layer2 = mv.layer2
        self.layer3 = mv.layer3
        self.layer4 = mv.layer4

        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.bn1.parameters():
            param.requires_grad = False

        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False

        if not finetune:
            for param in self.layer3.parameters():
                param.requires_grad = False
            for param in self.layer4.parameters():
                param.requires_grad = False

    @abstractmethod
    def feature_size(self):
        pass

    @abstractmethod
    def model_version(self):
        pass

    def forward(self, image):
        """Forward function

        Args:
            image (torch.FloatTensor): Mini-batches of 3-channel RGB images of shape
                (N x 3 x H x W), with H and W >= 224 and values in range [0,255]

        Returns:
            (torch.FloatTensor): Image features

        Shape:
            Input:
                image: (batch_size, 3, image_height, image_width)
            Output:
                output: (batch_size, 512, features_height, features_width)
        """
        x = self.conv1(self.normalize(image))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class DenserResNet(ResNet):
    """
    ResNet class with denser spatial output by combining features from second last and last layer of ResNet 34.

    Logic:
        ResNet34: Conv --> layer1 --> layer2 --> layer3 --> layer4 --> AvgPool --> FC

        features = concatenate(layer3_features, upscale(layer4_features), dim=1)

        layer3_features: X, Y, 256
        layer4_features: X/2, Y/2, 512
        upscale(layer4_features): X, Y, 512
        features: X, Y, (256 + 512)
    """

    def forward(self, image):
        """Forward function

        Args:
            image (torch.FloatTensor): Mini-batches of 3-channel RGB images of shape
                (N x 3 x H x W), with H and W >= 224 and values in range [0,255]

        Returns:
            (torch.FloatTensor): Image features

        Shape:
            Input:
                image: (batch_size, 3, image_height, image_width)
            Output:
                features: (batch_size, 768, features_height, features_width)
        """

        x = self.conv1(self.normalize(image))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        features_256 = self.layer3(x)
        features_512 = self.layer4(features_256)

        # Nearest Neighbour Upscaling
        features_512 = nn.functional.interpolate(features_512, scale_factor=2, mode='bilinear', align_corners=False)
        features = torch.cat((features_256, features_512), dim=1)
        return features


class DenserResNet34(DenserResNet):

    def model_version(self):
        return models.resnet34

    def feature_size(self):
        return 768
