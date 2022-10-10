# source: https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb

import os
import random
import wandb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from pathlib import Path
import pandas as pd
import copy

from utils.data import BatchReader
from utils.metric import my_log_loss

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # remove slow mirror from list of MNIST mirrors
# torchvision.datasets.MNIST.mirrors = [
#     mirror for mirror in torchvision.datasets.MNIST.mirrors if not mirror.startswith("http://yann.lecun.com")
# ]

# Define the experiment
# config = dict(
#     epochs=5, classes=10, kernels=[16, 32], batch_size=128, learning_rate=0.005, dataset="RocketLeague", architecture="XGBoost"
# )

config = dict(
    features=6,
    epochs=5,
    batch_size=2**10,
    colsample_bytree=0.3,
    learning_rate=0.1,
    max_depth=5,
    alpha=10,
    n_estimators=10,
    eval_metric=my_log_loss,
    dataset="RocketLeague",
    architecture="Net",
)


def model_pipeline(hyperparameters):

    # tell wandb to get started
    with wandb.init(project="xgb-test", config=hyperparameters):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # make the model, data, and optimization problem
        model, train_loader, test_loader, criterion, optimizer = make(config)
        print(model)

        # and use them to train the model
        train(model, train_loader, criterion, optimizer, config)

        # and test its final performance
        test(model, test_loader)

    return model


def make(config):
    # Make the data
    train = [
        "/Users/franwe/repos/RocketLeague/data/raw/train_0.csv",
        "/Users/franwe/repos/RocketLeague/data/raw/train_1.csv",
    ]
    test = ["/Users/franwe/repos/RocketLeague/data/processed/train_4.csv"]
    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    # Make the model
    model = Net(config.features).to(device)

    # Make the loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    return model, train_loader, test_loader, criterion, optimizer


def make_loader(dataset, batch_size):
    loader = BatchReader(dataset, N=batch_size)
    return loader.read()


# neural network


class Net(nn.Module):
    def __init__(self, features):
        super(Net, self).__init__()
        self.hid1 = nn.Linear(features, 10)  # N-(10-10)-1
        self.hid2 = nn.Linear(10, 10)
        self.oupt = nn.Linear(10, 1)

        nn.init.xavier_uniform_(self.hid1.weight)
        nn.init.zeros_(self.hid1.bias)
        nn.init.xavier_uniform_(self.hid2.weight)
        nn.init.zeros_(self.hid2.bias)
        nn.init.xavier_uniform_(self.oupt.weight)
        nn.init.zeros_(self.oupt.bias)

    def forward(self, x):
        z = F.relu(self.hid1(x))
        z = F.relu(self.hid2(z))
        z = self.oupt(z)  # no activation
        return z


def train(model, loader, criterion, optimizer, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    total_batches = None
    example_ct = 0  # number of examples seen
    batch_ct = 0
    for epoch in range(config.epochs):
        for _, df in enumerate(loader):
            values = df[FEATURES].values
            labels = df[TARGET].values
            loss = train_batch(values, labels, model, optimizer, criterion)
            example_ct += len(values)
            batch_ct += 1

            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)


def train_batch(images, labels, model, optimizer, criterion):
    images, labels = torch.from_numpy(images), torch.from_numpy(labels)
    images, labels = images.float(), labels.float()
    # TODO: move to loader (above)

    images, labels = images.to(device), labels.to(device)

    # Forward pass ➡
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss


def train_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")


def test(model, test_loader):
    model.eval()

    # Run the model on some test examples
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy of the model on the {total} " + f"test images: {correct / total:%}")

        wandb.log({"test_accuracy": correct / total})

    # Save the model in the exchangeable ONNX format
    torch.onnx.export(model, images, "model.onnx")
    wandb.save("model.onnx")


if __name__ == "__main__":
    FEATURES = ["ball_pos_x", "ball_pos_y", "ball_pos_z", "ball_vel_x", "ball_vel_y", "ball_vel_z"]
    TARGET = "team_A_scoring_within_10sec"
    # Build, train and analyze the model with the pipeline
    model = model_pipeline(config)
