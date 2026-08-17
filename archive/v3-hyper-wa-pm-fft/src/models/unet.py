import torch
import torch.nn as nn

class Double_conv(nn.Module):
    def __init__(self, in_ch, hidden, out_ch, kernel_size=3):
        super(Double_conv, self).__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size, padding=padding),
            nn.Tanhshrink(),
            nn.Conv2d(hidden, out_ch, kernel_size, padding=padding),
            nn.Tanhshrink(),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch=2, hidden=32, out_ch=12, kernel_size=5):
        super().__init__()

        # Down (encoder) - reduced by one level
        self.dconv_down1 = Double_conv(in_ch, hidden, hidden, kernel_size)
        self.dconv_down2 = Double_conv(hidden, hidden*2, hidden*2, kernel_size)
        self.dconv_down3 = Double_conv(hidden*2, hidden*4, hidden*4, kernel_size)

        # Up (decoder) - reduced by one level
        self.dconv_up2 = Double_conv(hidden*4 + hidden*2, hidden*2, hidden*2, kernel_size)
        self.dconv_up1 = Double_conv(hidden*2 + hidden, hidden, hidden, kernel_size)

        # Pooling and upsampling
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Final convolution
        self.conv_last = nn.Conv2d(hidden, out_ch, 1)

    def forward(self, x):
        # Encoder
        conv1 = self.dconv_down1(x)      # -> hidden channels
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)      # -> hidden*2 channels
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)      # -> hidden*4 channels

        # Decoder
        x = self.upsample(conv3)                         # up to match conv2 spatial
        x = torch.cat([x, conv2], dim=1)                 # channels: hidden*4 + hidden*2
        x = self.dconv_up2(x)                            # -> hidden*2 channels

        x = self.upsample(x)                             # up to match conv1 spatial
        x = torch.cat([x, conv1], dim=1)                 # channels: hidden*2 + hidden
        x = self.dconv_up1(x)                            # -> hidden channels

        out = self.conv_last(x)                          # -> out_ch
        return out

    def load_model(self, filepath, mode='eval'):
        self.load_state_dict(torch.load(filepath))
        if mode == 'eval':
            self.eval()
        elif mode == 'train':
            self.train()
        else:
            raise ValueError("Mode must be 'eval' or 'train'.")
        return self

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)
    