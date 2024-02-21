import torch
import torch.nn as nn
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
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


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_x = x2.size()[2] - x1.size()[2]
        diff_y = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2))
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, 1, kernel_size=1)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

class DDPM(nn.Module):
    def __init__(self,  num_timesteps,segments):
        super(DDPM, self).__init__()
        self.num_timesteps = num_timesteps
        self.segments = segments
        self.unet = UNet()

    def forward(self, x, t):
        noise_std = self.piecewise_linear_noise_schedule(t)
        noise = torch.randn_like(x) * noise_std
        x = x + noise
        x = self.unet(x)
        return x

    def piecewise_linear_noise_schedule(self, t):
        assert self.segments[0][0] == 0, "First segment should start at t=0"

        noise_schedule = torch.zeros(self.num_timesteps)
        prev_end_t = 0
        prev_noise_level = 1.0

        for end_t, slope in self.segments:
            duration = end_t - prev_end_t
            if duration > 0:
                noise_levels = torch.linspace(prev_noise_level, prev_noise_level - duration * slope, duration)
                noise_schedule[prev_end_t:end_t] = noise_levels
                prev_noise_level = noise_levels[-1]
            prev_end_t = end_t

        return noise_schedule[t]


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(64 * 8 * 16, 128)
        self.fc2 = torch.nn.Linear(128, 32 * 64)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))  #16,32,64,128
        x = self.pool(x)                              #16,32,32,64
        x = torch.nn.functional.relu(self.conv2(x))   #16,64,32,64
        x = self.pool(x)                               #16,64,16,32
        x = x.view(-1, 64 * 8 * 16)                     #64,8192
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.reshape(-1, 1, 64, 128)
        return x

if __name__ == "__main__":

    num_timesteps = 101
    segments = [(0, 0.02), (30, 0.01), (70, 0.005), (100, 0)]
    model1 = DDPM(num_timesteps,segments)
    model2 = UNet()
    input_tensor = torch.randn(128, 1, 64, 128)
    timestep = 100
    # 输出
    output1 = model1(input_tensor, timestep)
    output2 = model2(input_tensor)
    print(output1)
    print(output1.shape)
    print(output2)
    print(output2.shape)