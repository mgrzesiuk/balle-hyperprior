import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Uniform

from src.models.loss.loss_helper import LossDistortionHelper


class Conv2dWithSampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, scale_factor, mode):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.scale = scale_factor
        self.mode = mode
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.conv2d(x)
        b = F.interpolate(a, scale_factor=self.scale, mode=self.mode)
        return b

class TrainingModule(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, hyperprior_encoder: nn.Module, hyperprior_decoder: nn.Module, distortion_loss_helper: LossDistortionHelper):
        super().__init__()
        self.noise = Uniform(torch.tensor(-1/2), torch.tensor(1/2))
        self.distortion_helper = distortion_loss_helper

        self.hyperprior_mean = nn.Parameter(torch.empty((1, 384, 64, 64), requires_grad=True))
        nn.init.xavier_uniform_(self.hyperprior_mean)

        self.hyperprior_std_deviation = nn.Parameter(torch.empty((1, 384, 64, 64), requires_grad=True))
        nn.init.xavier_uniform_(self.hyperprior_std_deviation)

        self.encoder = encoder
        self.decoder = decoder
        self.hyperprior_encoder = hyperprior_encoder
        self.hyperprior_decoder = hyperprior_decoder
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        y: torch.Tensor = self.encoder(x)
        y_tilde = self.add_uniform_noise(y)
        z = self.hyperprior_encoder(y)
        z_tilde = self.add_uniform_noise(z)
        hyperprior_mean, hyperprior_std_deviation = self.hyperprior_decoder(z_tilde)
        x_tilde = self.decoder(y_tilde)
        
        return x_tilde, y_tilde, z_tilde, hyperprior_mean, hyperprior_std_deviation
    
    def add_uniform_noise(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.noise.sample(x.size()).to(x.device)
    
    def side_info_rate(self, z_tilde) -> torch.Tensor:
        # I bet simple normal works just as well as that non parametric distribution defined in the paper
        # to power of 2 since need to guarantee this is positive
        normal = Normal(self.hyperprior_mean, self.hyperprior_std_deviation.pow(2) + 1e-10)
        return -torch.log((normal.cdf(z_tilde + 1/2) - normal.cdf(z_tilde - 1/2)).mean())
    
    def rate(self, y_tilde: torch.Tensor, mean: torch.Tensor,  std_deviation: torch.Tensor) -> torch.Tensor:
        normal = Normal(mean, std_deviation)
        return -torch.log((normal.cdf(y_tilde + 1/2) - normal.cdf(y_tilde - 1/2)).mean())
    
    def distortion(self, x: torch.Tensor, x_tilde: torch.Tensor) -> torch.Tensor:
        return self.distortion_helper.distortion(x_tilde, x)

class CompressionModule(nn.Module):
    def from_training_module(training_module: TrainingModule):
        raise NotImplementedError()
