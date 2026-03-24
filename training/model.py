import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False) if down 
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if not down else nn.LeakyReLU(0.2, inplace=True)
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.down1 = nn.Sequential(nn.Conv2d(in_channels, 64, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True))
        self.down2 = UNetBlock(64, 128)
        self.down3 = UNetBlock(128, 256)
        self.down4 = UNetBlock(256, 512)
        self.down5 = UNetBlock(512, 512)
        self.down6 = UNetBlock(512, 512)
        self.down7 = UNetBlock(512, 512)
        self.down8 = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1, bias=False), nn.ReLU(inplace=True))

        self.up1 = UNetBlock(512, 512, down=False, use_dropout=True)
        self.up2 = UNetBlock(1024, 512, down=False, use_dropout=True)
        self.up3 = UNetBlock(1024, 512, down=False, use_dropout=True)
        self.up4 = UNetBlock(1024, 512, down=False)
        self.up5 = UNetBlock(1024, 256, down=False)
        self.up6 = UNetBlock(512, 128, down=False)
        self.up7 = UNetBlock(256, 64, down=False)
        
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        
        return self.final_up(torch.cat([u7, d1], 1))

def weights_init_normal(m):
    """
    Standalone function to initialize the discriminator weights 
    using a normal distribution (mean=0.0, std=0.02).
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight') and m.weight is not None:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('InstanceNorm2d') != -1:
        # InstanceNorm2d defaults to affine=False, so weight/bias may be None.
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=6, ndf=64):
        """
        Input constraint: Concatenation of the conditioning image (3 channels) 
        and the target/generated image (3 channels) = 6 channels total.
        Output: N x N patch map indicating Real/Fake (No Sigmoid).
        """
        super().__init__()

        def discriminator_block(in_filters, out_filters, normalization=True, stride=2):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=stride, padding=1, bias=False)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # Input shape: (6, H, W)
            *discriminator_block(in_channels, ndf, normalization=False), # No norm on first layer
            *discriminator_block(ndf, ndf * 2),
            *discriminator_block(ndf * 2, ndf * 4),
            # Downsample stops here mathematically; stride=1 preserves N x N map size
            *discriminator_block(ndf * 4, ndf * 8, stride=1),
            # Final output layer; padding=1 preserves features, outputs 1 channel patch map
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)
            # NO Sigmoid layer applied. Handled by BCEWithLogitsLoss.
        )

    def forward(self, img_condition, img_target_or_fake):
        # Concatenate condition image and target/generated image along channel dimension
        img_input = torch.cat((img_condition, img_target_or_fake), dim=1)
        return self.model(img_input)