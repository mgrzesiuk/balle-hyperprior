import argparse
from random import randint
import wandb
import torch

from src.datasets.training_dataset import Caltech101Dataset
from src.datasets.validation_dataset import KodakDataset
from src.model_run_utils import train, validation
from src.models.hyperprior.balle import HyperpriorDecoder, HyperpriorEncoder
from src.models.loss.mse_distortion import MSEDistortionHelper
from src.models.utils import TrainingModule
from src.models.vision.simplified_balle import Decoder, Encoder


def get_architecture(architecture_name) -> TrainingModule:
    if architecture_name == "SimplifiedBalle-mse":
        return TrainingModule(Encoder(), Decoder(), HyperpriorEncoder(), HyperpriorDecoder(), MSEDistortionHelper())

    raise ValueError("Unrechognised architecture")

def init_wandb(learning_rate: float, arch_name: str, epochs: int):
    wandb.init(
        # set the wandb project where this run will be logged
        project="compression",

        # track hyperparameters and run metadata
        config={
        "learning_rate": learning_rate,
        "architecture": arch_name,
        "epochs": epochs,
        }
    )

def main():
    parser = argparse.ArgumentParser(prog="MLCompression")

    parser.add_argument("--lr", type=float, help="learning rate for the training of the model", required=True)
    parser.add_argument("--arch", type=str, help="which architecture to use", required=True)
    parser.add_argument("--epochs", type=int, help="number of epochs for training of the model", required=True)
    parser.add_argument("--batch-size", type=int, help="batch size during the training of the model", required=True)
    parser.add_argument("--distortion-weight", type=float, help="weight of the distortion loss", required=True)

    args = parser.parse_args()

    learning_rate = args.lr
    arch_name = args.arch
    epochs = args.epochs
    batch_size = args.batch_size
    distortion_weight = args.distortion_weight
    
    model = get_architecture(arch_name)
    training_data = Caltech101Dataset("data/training")
    validation_data = KodakDataset("data/validation", download=True)

    init_wandb(learning_rate, arch_name, epochs)
    torch.autograd.set_detect_anomaly(True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device")
    train(model, training_data, optimizer, batch_size, device, distortion_weight, epochs)
    validation(model, validation_data, batch_size, device)

if __name__ == "__main__":
    main()