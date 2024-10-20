from random import randint
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Uniform
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, ToPILImage, RandomCrop

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
        self.mse_loss = nn.MSELoss()

        self.hyperprior_mean = nn.Parameter(torch.empty((1, 384, 64, 64), requires_grad=True))
        nn.init.xavier_uniform_(self.hyperprior_mean)

        self.hyperprior_std_deviation = nn.Parameter(torch.empty((1, 384, 64, 64), requires_grad=True))
        nn.init.xavier_uniform_(self.hyperprior_std_deviation)

        self.encoder = Encoder()
        self.decoder = Decoder()
        
        self.hyperprior_encoder = HyperpriorEncoder()
        self.hyperprior_decoder = HyperpriorDecoder()
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        y: torch.Tensor = self.encoder(x)
        #print("y", y.max(), y.median())
        y_tilde = self.add_uniform_noise(y)
        z = self.hyperprior_encoder(y)
        #print("z", z.max(), z.median())
        z_tilde = self.add_uniform_noise(z)
        hyperprior_mean, hyperprior_std_deviation = self.hyperprior_decoder(z_tilde)
        x_tilde = self.decoder(y_tilde)
        
        return x_tilde, y_tilde, z_tilde, hyperprior_mean, hyperprior_std_deviation
    
    def add_uniform_noise(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.noise.sample(x.size()).to(x.device)
    
    def side_info_rate(self, z_tilde) -> torch.Tensor:
        # I bet simple normal works just as well as that non parametric distribution defined in the paper
        # to power of 2 since need to guarantee this is positive
        normal = Normal(self.hyperprior_mean, self.hyperprior_std_deviation.pow(2)+ + 1e-10)
        return -torch.log((normal.cdf(z_tilde + 1/2) - normal.cdf(z_tilde - 1/2)).mean())
    
    def rate(self, y_tilde: torch.Tensor, mean: torch.Tensor,  std_deviation: torch.Tensor) -> torch.Tensor:
        normal = Normal(mean, std_deviation)
        return -torch.log((normal.cdf(y_tilde + 1/2) - normal.cdf(y_tilde - 1/2)).mean())
    
    def distortion(self, x: torch.Tensor, x_tilde: torch.Tensor) -> torch.Tensor:
        return self.mse_loss(x_tilde, x)

class CompressionModule(nn.Module):
    def from_training_module(training_module: TrainingModule):
        pass

class INaturalistDataset(datasets.Caltech101):
    def __init__(self):
        super().__init__("data", download=True)
        self.tensor_transform = ToTensor()
        self.random_crop = RandomCrop(256, pad_if_needed=True)
        
    def __getitem__(self, index: int) -> torch.Tuple[torch.Tensor]:
        image, target = super().__getitem__(index)
        tensor = self.tensor_transform(self.random_crop(image))
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        return tensor, target

def train(model: TrainingModule, dataset: Dataset, optimizer: torch.optim.Optimizer, batch_size: int, device: torch.device, distortion_weight: float, epochs: int, num_samples: int) -> None:
    model.to(device)
    model.train()
    for epoch in range(epochs):
        train_single_epoch(model, dataset, optimizer, batch_size, device, distortion_weight, epoch)
    model.eval()
    visual_loss_eval(model, dataset, num_samples, device)

def train_single_epoch(model: TrainingModule, dataset: Dataset, optimizer: torch.optim.Optimizer, batch_size: int, device: torch.device, distortion_weight: float, epoch: int) -> None:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    size = len(dataloader.dataset)
    loss_distortion_list = []
    for batch_idx, (x, _) in enumerate(dataloader):
        # overtrain on one batch for now to see if this even works
        x: torch.Tensor = x.to(device=device)
        x_tilde, y_tilde, z_tilde, hyperprior_mean, hyperprior_std_deviation = model(x)
        loss_distortion = model.distortion(x, x_tilde)
        loss_distortion_list.append(loss_distortion)
        loss_rate = model.rate(y_tilde, hyperprior_mean, hyperprior_std_deviation)
        loss_side_info = model.side_info_rate(z_tilde)
        loss = distortion_weight * loss_distortion + loss_rate + loss_side_info
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
                    
        if batch_idx % 10 == 0:
            loss, current = loss.item(), (batch_idx + 1) * len(x)
            print(f"loss: {loss:>7f} (distortion: {loss_distortion:>7f}, rate: {loss_rate:>7f}, side information: {loss_side_info:>7f}) [{current:>5d}/{size:>5d}, epoch: {epoch}]")
            wandb.log({"loss": loss, "distortion": loss_distortion, "rate": loss_rate, "side info": loss_side_info})
    wandb.log({"avg_distortion_over_epoch": sum(loss_distortion_list)/len(loss_distortion_list)})

def visual_loss_eval(model: TrainingModule, dataset: Dataset, num_samples: int, device: torch.device):
    table = wandb.Table(columns=["original", "decompressed"])
    for _ in range(num_samples):
        x, _ = dataset[randint(0, len(dataset) - 1)]
        x = x.to(device)
        y_test = model.encoder(x)
        x_test_decompressed = model.decoder(y_test)
        image = ToPILImage()(x)
        image_tilde = ToPILImage()(x_test_decompressed)
        table.add_data(wandb.Image(image), wandb.Image(image_tilde))
    wandb.log({"visual_loss": table})

def init_wandb(learning_rate: float, arch_name: str, dataset: str, epochs: int):
    wandb.init(
        # set the wandb project where this run will be logged
        project="compression",

        # track hyperparameters and run metadata
        config={
        "learning_rate": learning_rate,
        "architecture": arch_name,
        "dataset": dataset,
        "epochs": epochs,
        }
    )

def main():
    learning_rate = 1e-3
    arch_name = "BALLE-Standard"
    dataset = "INaturalist"
    epochs = 10
    init_wandb(learning_rate, arch_name, dataset, epochs)
    
    batch_size = 32
    torch.autograd.set_detect_anomaly(True)
    data = INaturalistDataset()
    
    model = TrainingModule()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device")
    train(model, data, optimizer, batch_size, device, 255**2, epochs, 15)

if __name__ == "__main__":
    main()