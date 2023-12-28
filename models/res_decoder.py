import torch
import torch.nn as nn
import torchvision

print("PyTorch Version: ", torch.__version__)
# print("Torchvision Version: ",torchvision.__version__)

__all__ = ['ResNet50', 'ResNet101', 'ResNet152']


def DeConv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_planes, out_channels=places, kernel_size=8, stride=stride, padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(in_channels=places, out_channels=places, kernel_size=4, stride=stride, padding=1, bias=False),
    )


class Bottleneck(nn.Module):
    def __init__(self, in_places, places, stride=1, upsampling=False, expansion=2):
        super(Bottleneck, self).__init__()
        self.expansion = expansion   #用于扩展通道数
        self.upsampling = upsampling  #是否进行上采样

        if self.upsampling and stride==2:
            self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=places, out_channels=places, kernel_size=4, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places // self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places // self.expansion),
            )
        else:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(places),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1,
                                   bias=False),
                nn.BatchNorm2d(places),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=places, out_channels=places // self.expansion, kernel_size=1, stride=1,
                          bias=False),
                nn.BatchNorm2d(places // self.expansion),
            )

        if self.upsampling and stride==2:
            self.upsampling = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_places, out_channels=places // self.expansion,  kernel_size=4, stride=stride, padding=1,
                          bias=False),
                nn.BatchNorm2d(places // self.expansion)
            )
        else:
            self.upsampling = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places // self.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(places // self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        if self.upsampling:
            residual = self.upsampling(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, blocks, num_classes=1000, expansion=2):
        super(ResNet, self).__init__()
        self.expansion = expansion

        self.deconv1 = DeConv1(in_planes=16,places=3)
        self.layer1 = self.make_layer(in_places=2048, places=1024, block=blocks[0], stride=2)
        self.layer2 = self.make_layer(in_places=512, places=256, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=128, places=64, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=32, places=32, block=blocks[3], stride=1)
        # print(self.layer4)


        # self.avgpool = nn.AvgPool2d(7, stride=1)

        # self.fc = nn.Linear(4096, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, stride, upsampling=True))
        for i in range(1, block):
            layers.append(Bottleneck(places // self.expansion, places))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.deconv1(x)
        x = self.layer1(x)

        x = self.layer2(x)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        x=self.deconv1(x)

        return x


def ResNet50(num_classes=4):
    return ResNet([3, 4, 6, 3], num_classes=num_classes)


def ResNet101():
    return ResNet([3, 4, 23, 3])


def ResNet152():
    return ResNet([3, 8, 36, 3])


if __name__ == '__main__':
    # model = torchvision.models.resnet50()
    model = ResNet101()
    print(model)

    input = torch.randn(1, 2048, 8, 8)
    out = model(input)
    print(out.shape)
