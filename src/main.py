from random import randint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Uniform
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, RandomCrop

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

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(Conv2dWithSampling(in_channels=3, out_channels=128*3, kernel_size=5, padding='same', scale_factor=0.5, mode="bilinear"))
        self.hidden_layers.append(Conv2dWithSampling(in_channels=128*3, out_channels=128*3, kernel_size=5, padding='same', scale_factor=0.5, mode="bilinear"))
        self.hidden_layers.append(Conv2dWithSampling(in_channels=128*3, out_channels=128*3, kernel_size=5, padding='same', scale_factor=0.5, mode="bilinear"))
        self.final_layer = Conv2dWithSampling(in_channels=128*3, out_channels=192*3, kernel_size=5, padding='same', scale_factor=0.5, mode="bilinear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.final_layer(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(Conv2dWithSampling(in_channels=192*3, out_channels=128*3, kernel_size=5, padding='same', scale_factor=2, mode="bilinear"))
        self.hidden_layers.append(Conv2dWithSampling(in_channels=128*3, out_channels=128*3, kernel_size=5, padding='same', scale_factor=2, mode="bilinear"))
        self.hidden_layers.append(Conv2dWithSampling(in_channels=128*3, out_channels=128*3, kernel_size=5, padding='same', scale_factor=2, mode="bilinear"))
        self.final_layer = Conv2dWithSampling(in_channels=128*3, out_channels=3, kernel_size=5, padding='same', scale_factor=2, mode="bilinear")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.final_layer(x)

class HyperpriorEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_layer = nn.Conv2d(3*192, 3*128, 3, padding='same')

        self.hidden_layer = Conv2dWithSampling(in_channels=3*128, out_channels=3*128, kernel_size=5, padding='same', scale_factor=0.5, mode="bilinear")
        self.out_layer = Conv2dWithSampling(in_channels=3*128, out_channels=3*128, kernel_size=5, padding='same', scale_factor=0.5, mode="bilinear")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.in_layer(torch.abs(x)))
        x = F.relu(self.hidden_layer(x))
        z = self.out_layer(x)

        return z

class HyperpriorDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(Conv2dWithSampling(in_channels=3*128, out_channels=3*128, kernel_size=5, padding='same', scale_factor=2, mode="bilinear"))
        self.hidden_layers.append(Conv2dWithSampling(in_channels=3*128, out_channels=3*128, kernel_size=5, padding='same', scale_factor=2, mode="bilinear"))
        self.hidden_layers.append(nn.Conv2d(in_channels=3*128, out_channels=3*192, kernel_size=3, padding='same'))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        # add small non zero val to ensure this is > 0
        return x + 1e-10*torch.ones_like(x)

class HyperpriorEncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = HyperpriorEncoder()
        self.decoder = HyperpriorDecoder()
    
    def forward(self, x) -> torch.Tensor:
        return self.decoder(self.encoder(x))    
    

class TrainingModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.noise = Uniform(torch.tensor(-1/2), torch.tensor(1/2))
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        self.hyperprior_encoder = HyperpriorEncoder()
        self.hyperprior_decoder = HyperpriorDecoder()
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        y_tilde = self.add_uniform_noise(self.encoder(x))
        z_tilde = self.add_uniform_noise(self.hyperprior_encoder(y_tilde))
        hyperprior_std_deviation = self.hyperprior_decoder(z_tilde)
        x_tilde = self.decoder(y_tilde)
        
        return x_tilde, y_tilde, z_tilde, hyperprior_std_deviation
    
    def add_uniform_noise(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.noise.sample(x.size()).to(x.device)

class CompressionModule(nn.Module):
    def from_training_module(training_module: TrainingModule):
        pass

class INaturalistDataset(datasets.INaturalist):
    def __init__(self):
        super().__init__("data", version="2021_train_mini", download=False)
        self.tensor_transform = ToTensor()
        self.random_crop = RandomCrop(256, pad_if_needed=True)
    
    def __len__(safe) -> int:
        # this dataset is pretty large, lets see if we can get away with using only 10th of it
        return super().__len__()//10
    
    def __getitem__(self, index: int) -> torch.Tuple[torch.Tensor]:
        # this will kind of work like stochastic gradient descent maybe?
        index = randint(0, 9) * 10 + index
        image, target = super().__getitem__(index)
        return self.tensor_transform(self.random_crop(image)), target

class HyperpriorLoss:
    def __init__(self, distortion_weight: float, prior_mean: torch.Tensor, prior_std_dev: torch.Tensor):
        self.distortion_weight = distortion_weight
        self.prior_mean = prior_mean
        self.prior_std_dev = prior_std_dev
        self.mse_loss = nn.MSELoss()

    def __call__(self, x: torch.Tensor, x_tilde: torch.Tensor, y_tilde: torch.Tensor, z_tilde: torch.Tensor, std_deviation: torch.Tensor) -> torch.Tensor:
        return self.distortion_weight * self.distortion(x, x_tilde) + self.rate(y_tilde, std_deviation) + self.side_info_rate(z_tilde)
    
    def distortion(self, x: torch.Tensor, x_tilde: torch.Tensor) -> torch.Tensor:
        val = self.mse_loss(x_tilde, x)
        return val

    def rate(self, y_tilde: torch.Tensor, std_deviation: torch.Tensor) -> torch.Tensor:
        normal = Normal(torch.zeros_like(std_deviation), std_deviation)
        ones = torch.ones_like(y_tilde)
        return (normal.cdf(y_tilde + ones * 1/2) - normal.cdf(y_tilde - ones * 1/2)).mean()

    def side_info_rate(self, z_tilde) -> torch.Tensor:
        # I bet simple normal works just as well as that non parametric distribution defined in the paper
        normal = Normal(self.prior_mean, self.prior_std_dev)
        ones = torch.ones_like(z_tilde)
        
        return (normal.cdf(z_tilde + ones * 1/2) - normal.cdf(z_tilde - ones * 1/2)).mean()

def train(model: nn.Module, dataset: Dataset, loss_fn: HyperpriorLoss, optimizer: torch.optim.Optimizer, batch_size: int, device: torch.device) -> None:
    dataloader = DataLoader(dataset, batch_size=batch_size)
    size = len(dataloader.dataset)
    model.to(device)
    model.train()
    for batch_idx, (x, _) in enumerate(dataloader):
        # overtrain on one batch for now to see if this even works
        for epoch in range(1000):
            x: torch.Tensor = x.to(device=device)
            x_tilde, y_tilde, z_tilde, hyperprior_std_deviation = model(x)
            loss = loss_fn(x, x_tilde, y_tilde, z_tilde, hyperprior_std_deviation)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if epoch % 10 == 0:
                #loss, current = loss.item(), (epoch + 1) * len(x)
                #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                print(f"loss: {loss:>7f}  [{epoch}]")



def main():
    batch_size = 8

    data = INaturalistDataset()
    
    model = TrainingModule()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss = HyperpriorLoss(0.8, torch.tensor(0), torch.tensor(1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device")
    train(model, data, loss, optimizer, batch_size, device)

if __name__ == "__main__":
    main()