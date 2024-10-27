from abc import ABC, abstractmethod

import torch

class LossDistortionHelper(ABC):
    @abstractmethod
    def distortion() -> torch.Tensor:
        pass
