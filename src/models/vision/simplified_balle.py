import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.utils import Conv2dWithSampling

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(Conv2dWithSampling(in_channels=3, out_channels=128*3, kernel_size=5, padding='same', scale_factor=0.5, mode="nearest"))
        #self.hidden_layers.append(Conv2dWithSampling(in_channels=128*3, out_channels=128*3, kernel_size=5, padding='same', scale_factor=0.5, mode="nearest"))
        #self.hidden_layers.append(Conv2dWithSampling(in_channels=128*3, out_channels=128*3, kernel_size=5, padding='same', scale_factor=0.5, mode="nearest"))
        self.final_layer = Conv2dWithSampling(in_channels=128*3, out_channels=192*3, kernel_size=5, padding='same', scale_factor=0.5, mode="nearest")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.final_layer(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(Conv2dWithSampling(in_channels=192*3, out_channels=128*3, kernel_size=5, padding='same', scale_factor=2, mode="nearest"))
        #self.hidden_layers.append(Conv2dWithSampling(in_channels=128*3, out_channels=128*3, kernel_size=5, padding='same', scale_factor=2, mode="nearest"))
        #self.hidden_layers.append(Conv2dWithSampling(in_channels=128*3, out_channels=128*3, kernel_size=5, padding='same', scale_factor=2, mode="nearest"))
        self.final_layer = Conv2dWithSampling(in_channels=128*3, out_channels=3, kernel_size=5, padding='same', scale_factor=2, mode="nearest")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.final_layer(x)
