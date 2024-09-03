import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, RandomCrop

class Encode(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(270000, 1)
    
    def forward(self, x):
        return self.layer(x)

class Decode(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 270000)
    
    def forward(self, x):
        return self.layer(x)

class INaturalistDataset(datasets.INaturalist):
    def __init__(self):
        super().__init__("data", version="2021_train_mini", download=False)
        self.tensor_transform = ToTensor()
        self.random_crop = RandomCrop(300, pad_if_needed=True)
    
    def __getitem__(self, index: int) -> torch.Tuple[torch.Tensor]:
        image, target = super().__getitem__(index)
        image.show()
        return self.tensor_transform(self.random_crop(image)), target
        

class Trainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encode()
        self.decoder = Decode()
    
    def forward(self, x):
        return self.decoder(self.encoder(x))

def main():
    batch_size = 64

    data = INaturalistDataset()
    dataloader = DataLoader(data, batch_size=batch_size)
    
    trainer = Trainer()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(trainer.parameters(), lr=1e-3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device")
    size = len(dataloader.dataset)

    trainer.to(device=device)
    for batch, (x, _) in enumerate(dataloader):
        x: torch.Tensor = x.to(device=device)
        x = x.reshape((batch_size, 270000))
        pred = trainer(x)
        loss = loss_fn(pred, x)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

if __name__ == "__main__":
    main()