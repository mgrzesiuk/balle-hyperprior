import torch
import torch.nn as nn

from src.models.loss.loss_helper import LossDistortionHelper


class MSEDistortionHelper(LossDistortionHelper):
    def __init__(self):
        self.mse_loss = nn.MSELoss()

    def distortion(self, x: torch.Tensor, x_tilde: torch.Tensor):
        return self.mse_loss(x_tilde, x)
