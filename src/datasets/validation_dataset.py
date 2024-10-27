import os
from pathlib import Path
import requests
from functools import cache

import torch
from torchvision.transforms import ToTensor, RandomCrop
from PIL import Image

class KodakDataset(torch.utils.data.Dataset):
    SIZE = 24

    def __init__(self, root: str, download=False):
        self.root = root
        self.tensor_transform = ToTensor()
        self.random_crop = RandomCrop(256, pad_if_needed=True)

        if download:
            self._download()
    
    @cache
    def __getitem__(self, index: int) -> torch.Tuple[torch.Tensor]:
        """
        Args:
            index (int): Index

        Returns:
            (Any): Sample and meta data, optionally transformed by the respective transforms.
        """
        if index < 0 or index >= self.SIZE:
            raise IndexError(f"index {index} is out of bounds for Kodak dataset")

        image = Image.open(os.path.join(self.root, f"{index+1}.png"))
        tensor = self.tensor_transform(self.random_crop(image))
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        return tensor

    def __len__(self) -> int:
        return self.SIZE

    def _download(self) -> None:
        if self._is_already_downloaded():
            print("Data already downloaded")
            return
    
        Path(self.root).mkdir(parents=True, exist_ok=True)

        print("Starting the download")
        for i in range(1, self.SIZE+1):
            response = requests.get(f"https://r0k.us/graphics/kodak/kodak/kodim{i:02}.png")
            with open(os.path.join(self.root, f"{i}.png"), "wb") as handle:
                handle.write(response.content)
        print("Download finished")
    
    def _is_already_downloaded(self) -> bool:
        if not os.path.exists(self.root):
            return False

        dir_content = os.listdir(self.root)
        if len(dir_content) == 0:
            return False
        
        # lets validate that content thats there is what we expect
        for i in range(1, self.SIZE+1):
            # for now lets just check file names
            if f"{i}.png" not in dir_content:
                # we could technically fill this in but for now lets just throw
                # since the validation is very shallow, no expectations can be made about
                # where does this dataset come from then or whats in it
                raise ValueError("Corrupted dataset")
        
        return True