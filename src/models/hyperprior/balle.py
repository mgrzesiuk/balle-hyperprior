import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.utils import Conv2dWithSampling

class HyperpriorEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_layer = nn.Conv2d(3*192, 3*128, 3, padding='same')

       # self.hidden_layer = Conv2dWithSampling(in_channels=3*128, out_channels=3*128, kernel_size=5, padding='same', scale_factor=0.5, mode="nearest")
        #self.out_layer = Conv2dWithSampling(in_channels=3*128, out_channels=3*128, kernel_size=5, padding='same', scale_factor=0.5, mode="nearest")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # create a copy so that we can 
        x = F.relu(self.in_layer(torch.abs(x)))
        #x = F.relu(self.hidden_layer(x))
        #z = self.out_layer(x)

        return x#z

class HyperpriorDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layers_mean = nn.ModuleList()
        #self.hidden_layers_mean.append(Conv2dWithSampling(in_channels=3*128, out_channels=3*128, kernel_size=5, padding='same', scale_factor=2, mode="nearest"))
        #self.hidden_layers_mean.append(Conv2dWithSampling(in_channels=3*128, out_channels=3*128, kernel_size=5, padding='same', scale_factor=2, mode="nearest"))
        self.hidden_layers_mean.append(nn.Conv2d(in_channels=3*128, out_channels=3*192, kernel_size=3, padding='same'))

        self.hidden_layers_std_deviation = nn.ModuleList()
        #self.hidden_layers_std_deviation.append(Conv2dWithSampling(in_channels=3*128, out_channels=3*128, kernel_size=5, padding='same', scale_factor=2, mode="nearest"))
        #self.hidden_layers_std_deviation.append(Conv2dWithSampling(in_channels=3*128, out_channels=3*128, kernel_size=5, padding='same', scale_factor=2, mode="nearest"))
        self.hidden_layers_std_deviation.append(nn.Conv2d(in_channels=3*128, out_channels=3*192, kernel_size=3, padding='same'))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std_deviation = x, x
        for layer in self.hidden_layers_mean:
            mean = F.relu(layer(mean))
        
        for layer in self.hidden_layers_std_deviation:
            std_deviation = F.relu(layer(std_deviation))
        # add small non zero val to ensure this is > 0
        std_deviation = std_deviation + 1e-10
        
        return mean, std_deviation
