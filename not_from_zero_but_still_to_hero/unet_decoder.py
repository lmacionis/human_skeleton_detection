import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels//2, out_channels)

    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)


class UNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.up_convolution_1 = UpSample(in_channels, 1024)
        self.up_convolution_2 = UpSample(1024, 512)
        self.up_convolution_3 = UpSample(512, 256)
        self.up_convolution_4 = UpSample(256, 128)

        self.out = nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1)
        # self.pool = nn.AdaptiveAvgPool2d(1)  # Global pooling
        # self.fc = nn.Linear(128, 2)  # Predict [x, y]

    def forward(self, x):
        # Dont use all up layers, cause of a huge feature amount from resnet50 to reduce training time, some spatial dimensions can be lost, cause image is not getting fully decoded (resized to bigger dimensions)
        x = self.up_convolution_1(x)
        # x = self.up_convolution_2(x)
        # x = self.up_convolution_3(x)
        # x = self.up_convolution_4(x)

        # x = self.pool(up_4)  # (N, 128, 1, 1)
        # x = x.view(x.size(0), -1)  # (N, 128)
        # x = self.fc(x)  # (N, 2)
        x = self.out(x)
        # Dont need sigmoid cause BCEWithLogitsLoss has it.
        # x = self.out(x).sigmoid()
        return x