import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            UNetBlock(in_channels, 64),
            nn.MaxPool2d(2),
            UNetBlock(64, 128),
            nn.MaxPool2d(2),
            UNetBlock(128, 256),
        )

        self.middle = UNetBlock(256, 512)

        self.decoder = nn.Sequential(
            UNetBlock(512, 256),
            UNetBlock(256, 128),
            UNetBlock(128, 64),
        )

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = nn.functional.interpolate(x, scale_factor=4, mode="nearest")
        x = self.decoder(x)
        x = self.final_conv(x)
        return x

class DDPM(nn.Module):
    def __init__(self, in_channels, out_channels, num_timesteps):
        super(DDPM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_timesteps = num_timesteps
        self.unet = UNet(in_channels, out_channels)

    def forward(self, x, t):
        noise_std = self.noise_schedule(t)
        noise = torch.randn_like(x) * noise_std
        x = x + noise
        x = self.unet(x)
        return x

    def noise_schedule(self, t):
        # 这里实现了一个简单的线性噪声调度方案，您可以根据需要修改它
        return torch.linspace(1.0, 0.0, self.num_timesteps)[t]

def main():
    # Example usage
    in_channels = 1
    out_channels = 1
    num_timesteps = 100
    model = DDPM(in_channels, out_channels, num_timesteps)
    input_tensor = torch.randn(1, 1, 64, 128)
    print(input_tensor)
    timestep = 50
    # 输出
    output = model(input_tensor, timestep)
    print(output)
    print(output.shape)


if __name__ == "__main__":
    main()

