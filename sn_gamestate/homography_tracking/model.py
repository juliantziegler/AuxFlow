import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

# Two consecutive convolutional layers with BatchNorm and ReLU
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# Down-sampling block: max pooling followed by a double convolution
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# Up-sampling block: upsample (or transposed convolution), concatenate with skip connection, and double conv.
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            # When not using bilinear, use transposed convolution.
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # In case the upsampled tensor doesn't match the size of the skip connection exactly.
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # Concatenate along the channel dimension.
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# Final 1x1 convolution to map features to the desired number of classes.
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv(x))


# Dynamic U-Net with an initial pooling layer
class DynamicUNet(nn.Module):
    def __init__(self, n_channels, n_classes, depth=4, base_channels=64, bilinear=True):
        """
        n_channels: Number of input channels (e.g., 3 for RGB)
        n_classes: Number of output classes (use 1 for binary segmentation)
        depth: Number of down/up-sampling steps.
        base_channels: Number of feature channels in the first level.
        bilinear: If True, use bilinear upsampling; otherwise, use transposed convolutions.
        """
        super(DynamicUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.depth = depth
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        # Initial 2x2 pooling to reduce spatial dimensions and save memory.
        self.input_pool = nn.MaxPool2d(2)

        # Encoder: initial block + a series of down blocks.
        self.inc = DoubleConv(n_channels, base_channels)
        self.downs = nn.ModuleList()
        enc_channels = [base_channels]  # to record channels at each encoder level
        current_channels = base_channels
        for _ in range(depth):
            next_channels = current_channels * 2
            self.downs.append(Down(current_channels, next_channels))
            current_channels = next_channels
            enc_channels.append(current_channels)

        # Bottleneck: processes the deepest feature maps.
        self.bottleneck = DoubleConv(current_channels, current_channels * 2 // factor)
        dec_channels = current_channels * 2 // factor

        # In the decoder, we use only the encoder outputs except the last one (used in bottleneck).
        skip_channels = enc_channels[:-1]  # this list has `depth` elements.
        self.ups = nn.ModuleList()
        # Build the up blocks by iterating over the skip channels in reverse order.
        for skip in reversed(skip_channels):
            self.ups.append(Up(dec_channels + skip, skip // factor if bilinear else skip, bilinear))
            dec_channels = skip // factor if bilinear else skip

        self.outc = OutConv(dec_channels, n_classes)

    def forward(self, x):
        # Apply the initial pooling layer.
        x = self.input_pool(x)

        # Encoder pathway.
        x_inc = self.inc(x)
        skips = [x_inc]
        x_enc = x_inc
        for down in self.downs:
            x_enc = down(x_enc)
            skips.append(x_enc)
        # Bottleneck.
        x_bottleneck = self.bottleneck(x_enc)
        # Use only the skip connections except the last encoder output.
        skips = skips[:-1]
        # Decoder pathway: apply up blocks using skip connections in reverse order.
        x_dec = x_bottleneck
        for up, skip in zip(self.ups, reversed(skips)):
            x_dec = up(x_dec, skip)
        logits = self.outc(x_dec)
        # Upsample the final output to match the original input resolution.
        logits = F.interpolate(logits, scale_factor=2, mode='bilinear', align_corners=True)
        return logits


# Example usage:
if __name__ == "__main__":
    # Create a random input tensor (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 1080, 1920, dtype=torch.float32)
    # Create the model instance with the desired depth and base channels.
    model = DynamicUNet(n_channels=3, n_classes=1, depth=4, base_channels=32, bilinear=True)
    output = model(input_tensor)
    summary(model, input_size=(1, 3, 1080, 1920))
    print("Output shape:", output.shape)
