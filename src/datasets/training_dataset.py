import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, RandomCrop

class Caltech101Dataset(datasets.Caltech101):
    def __init__(self, root: str):
        super().__init__(root, download=True)
        self.tensor_transform = ToTensor()
        self.random_crop = RandomCrop(256, pad_if_needed=True)
        
    def __getitem__(self, index: int) -> torch.Tuple[torch.Tensor]:
        image, _ = super().__getitem__(index)
        tensor = self.tensor_transform(self.random_crop(image))
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        return tensor
