import argparse
import os

import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch import nn
from torch.utils.data import DataLoader

import wandb

from torchvision import transforms
from torchvision import datasets
from models.resnet import PreActResNet18
from catalyst import dl
from catalyst.utils import set_global_seed

from callbacks.wandb import WandbCallback


def main(args):
    wandb.init(project="teacher-pruning", config=vars(args))
    set_global_seed(42)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(
        root=os.getcwd(), train=True, transform=transform_train, download=True
    )
    valid_dataset = datasets.CIFAR10(
        root=os.getcwd(), train=False, transform=transform_test
    )
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=128, num_workers=2)
    loaders = {
        "train": train_dataloader,
        "valid": valid_dataloader,
    }

    model = PreActResNet18()
    model.fc = nn.Linear(512, 10)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
    )
    scheduler = MultiStepLR(
        optimizer, milestones=[80, 120], gamma=args.gamma
    )
    runner = dl.SupervisedRunner(device=args.device)
    logdir=f"logs/{wandb.run.name}"

    runner.train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=nn.CrossEntropyLoss(),
        loaders=loaders,
        callbacks=[
            dl.AccuracyCallback(num_classes=10),
            WandbCallback()
        ],
        num_epochs=args.epoch,
        logdir=logdir,
        verbose=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("-d", "--device", default="cuda:0", type=str)
    parser.add_argument("--epoch", default=200, type=int)
    parser.add_argument("--gamma", default=0.1, type=float)
    args = parser.parse_args()
    main(args)
