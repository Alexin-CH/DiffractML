import torch
import torch.nn as nn

class PermittivityCNN(nn.Module):
    def __init__(self):
        super(PermittivityCNN, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv_4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.activation = nn.RReLU()
        
    def forward(self, x):
        in_1 = x
        out_1 = self.activation(self.conv_1(in_1))

        in_2 = self.pool(out_1 + in_1.repeat(1, 8, 1, 1))
        out_2 = self.activation(self.conv_2(in_2))

        in_3 = self.pool(out_2 + in_2.repeat(1, 2, 1, 1))
        out_3 = self.activation(self.conv_3(in_3))

        in_4 = self.pool(out_3 + in_3.repeat(1, 2, 1, 1))
        out_4 = self.activation(self.conv_4(in_4))

        return out_4


class AngleWavelengthMLP(nn.Module):
    def __init__(self):
        super(AngleWavelengthMLP, self).__init__()
        self.activation = nn.Tanh()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            self.activation,
            nn.Linear(32, 64),
            self.activation,
            nn.Linear(64, 128)
        )
        
    def forward(self, context):
        return self.net(context.float())

class ReconstructionCNN(nn.Module):
    def __init__(self):
        super(ReconstructionCNN, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.conv_4 = nn.Conv2d(in_channels=16, out_channels=12, kernel_size=3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.activation = nn.RReLU()

    def forward(self, permittivity_features, mlp_output):
        mlp_out = mlp_output.unsqueeze(-1).unsqueeze(-1)
        mlp_out = mlp_out.repeat(1, 1, permittivity_features.shape[2], permittivity_features.shape[3])

        # print("Shapes")
        # print(permittivity_features.shape)
        # print(mlp_out.shape)

        combined_factors = 0.6
        combined = combined_factors * permittivity_features + (1 - combined_factors) * mlp_out

        out_1 = self.activation(self.conv_1(combined))

        in_2 = self.upsample(out_1)
        out_2 = self.activation(self.conv_2(in_2))

        in_3 = self.upsample(out_2)
        out_3 = self.activation(self.conv_3(in_3))

        in_4 = self.upsample(out_3)
        out_4 = self.conv_4(in_4)

        return out_4

class EMFieldModel(nn.Module):
    def __init__(self):
        super(EMFieldModel, self).__init__()
        self.permittivity_cnn = PermittivityCNN()
        self.angle_wavelength_mlp = AngleWavelengthMLP()
        self.reconstruction_cnn = ReconstructionCNN()

    def forward(self, context, permittivity_map):
        features = self.permittivity_cnn(permittivity_map)
        mlp_output = self.angle_wavelength_mlp(context)
        reconstructed_field = self.reconstruction_cnn(features, mlp_output)
        return reconstructed_field
