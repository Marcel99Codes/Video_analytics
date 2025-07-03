import torch
import torch.nn as nn

class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes=25, return_features=False):
        super().__init__()
        self.in_planes = 64
        self.return_features = return_features

        self.stem = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        )

        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes)
            )

        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.return_features:
            return x
        return self.fc(x)

def inflate_weights_2d_to_3d(model3d, model2d):
    for (name3d, module3d), (_, module2d) in zip(model3d.named_modules(), model2d.named_modules()):
        if isinstance(module3d, nn.Conv3d) and isinstance(module2d, nn.Conv2d):
            with torch.no_grad():
                w2d = module2d.weight.data
                w3d = w2d.unsqueeze(2).repeat(1, 1, module3d.weight.shape[2], 1, 1) / module3d.weight.shape[2]
                module3d.weight.copy_(w3d)