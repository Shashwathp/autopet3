import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, strides=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=strides, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels) 
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.skip = nn.Sequential()
        if in_channels != out_channels or strides != 1:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=strides),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class UpSampleConcatBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampleConcatBlock, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x, skip):
        x = self.up(x)
        
        # Crop or pad the skip connection tensor to match the size of x
        diff_depth = skip.size(2) - x.size(2)
        diff_height = skip.size(3) - x.size(3)
        diff_width = skip.size(4) - x.size(4)

        x = nn.functional.pad(x, [diff_width // 2, diff_width - diff_width // 2,
                                  diff_height // 2, diff_height - diff_height // 2,
                                  diff_depth // 2, diff_depth - diff_depth // 2])

        x = torch.cat((x, skip), dim=1)
        return x

class ResUNet3D(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(ResUNet3D, self).__init__()
        f = [16, 32, 64, 128, 256]
        
        # Encoder
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, f[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(f[0]),
            nn.ReLU(inplace=True)
        )
        self.encoder1 = ResidualBlock(f[0], f[1], strides=2)
        self.encoder2 = ResidualBlock(f[1], f[2], strides=2)
        self.encoder3 = ResidualBlock(f[2], f[3], strides=2)
        self.encoder4 = ResidualBlock(f[3], f[4], strides=2)

        # Bridge
        self.bridge1 = ResidualBlock(f[4], f[4])
        self.bridge2 = ResidualBlock(f[4], f[4])
        
        # Decoder
        self.up4 = UpSampleConcatBlock(f[4], f[3])
        self.decoder4 = ResidualBlock(f[4] + f[3], f[3])
        
        self.up3 = UpSampleConcatBlock(f[3], f[2])
        self.decoder3 = ResidualBlock(f[3] + f[2], f[2])
        
        self.up2 = UpSampleConcatBlock(f[2], f[1])
        self.decoder2 = ResidualBlock(f[2] + f[1], f[1])
        
        self.up1 = UpSampleConcatBlock(f[1], f[0])
        self.decoder1 = ResidualBlock(f[1] + f[0], f[0])
        
        # Final output layer
        self.final_conv = nn.Conv3d(f[0], out_channels, kernel_size=1)
        self.final_upsample = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)  # Added to adjust size

    def forward(self, x):
        # Encoder
        e0 = self.stem(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Bridge
        b = self.bridge1(e4)
        b = self.bridge2(b)

        # Decoder
        u4 = self.up4(b, e4)
        d4 = self.decoder4(u4)

        u3 = self.up3(d4, e3)
        d3 = self.decoder3(u3)

        u2 = self.up2(d3, e2)
        d2 = self.decoder2(u2)

        u1 = self.up1(d2, e1)
        d1 = self.decoder1(u1)

        outputs = self.final_conv(d1)
        outputs = self.final_upsample(outputs)  # Adjust output size to match target
        return outputs