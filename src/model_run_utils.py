from random import randint
import wandb
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToPILImage

from src.models.utils import TrainingModule

def train(model: TrainingModule, dataset: Dataset, optimizer: torch.optim.Optimizer, batch_size: int, device: torch.device, distortion_weight: float, epochs: int) -> None:
    model.to(device)
    model.train()
    for epoch in range(epochs):
        train_single_epoch(model, dataset, optimizer, batch_size, device, distortion_weight, epoch)

def train_single_epoch(model: TrainingModule, dataset: Dataset, optimizer: torch.optim.Optimizer, batch_size: int, device: torch.device, distortion_weight: float, epoch: int) -> None:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    size = len(dataloader.dataset)
    loss_distortion_list = []
    for batch_idx, x in enumerate(dataloader):
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

def validation(model: TrainingModule, dataset: Dataset, batch_size: int, device: torch.device):
    model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    distortion = []
    rate = []
    side_info =[]

    for x in dataloader:
        x: torch.Tensor = x.to(device=device)
        x_tilde, y_tilde, z_tilde, hyperprior_mean, hyperprior_std_deviation = model(x)
        distortion.append(model.distortion(x, x_tilde))
        rate.append(model.rate(y_tilde, hyperprior_mean, hyperprior_std_deviation))
        side_info.append(model.side_info_rate(z_tilde))

    avg_val_distortion = sum(distortion)/len(distortion)
    avg_val_rate = sum(rate)/len(rate)
    avg_val_side_info = sum(side_info)/len(side_info)

    print(f"avg_val_distortion: {avg_val_distortion:>7f}, avg_val_rate: {avg_val_rate:>7f}, avg_val_side_info: {avg_val_side_info:>7f}")
    wandb.log({"avg_val_distortion": avg_val_distortion, "avg_val_rate": avg_val_rate, "avg_val_side_info": avg_val_side_info})