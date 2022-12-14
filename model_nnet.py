import random
from re import L
import wandb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.data import BatchReader, DataPreparerTorch
from utils.metric import my_log_loss

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    architecture="ClassifierNet",
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
    train_loader = make_loader(TRAIN_FILES, batch_size=config.batch_size)
    test_loader = make_loader(TEST_FILES, batch_size=config.batch_size)

    # Make the model
    model = ClassifierNet(config.features).to(device)
    criterion = model.get_criterion()
    optimizer = model.get_optimizer(config.learning_rate)

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
        z = self.oupt(z)  # TODO: everything is zero here after a while, sometimes also nan
        return z

    @staticmethod
    def get_criterion():
        return torch.nn.MSELoss()

    def get_optimizer(self, learning_rate):
        return torch.optim.SGD(self.parameters(), lr=learning_rate)


class ClassifierNet(nn.Module):
    def __init__(self, features):
        super(ClassifierNet, self).__init__()
        self.hid1 = nn.Linear(features, 10)  # N-(10-10)-1
        self.hid2 = nn.Linear(10, 10)
        self.oupt = nn.Linear(10, 3)

        nn.init.xavier_uniform_(self.hid1.weight)
        nn.init.zeros_(self.hid1.bias)
        nn.init.xavier_uniform_(self.hid2.weight)
        nn.init.zeros_(self.hid2.bias)
        nn.init.xavier_uniform_(self.oupt.weight)
        nn.init.zeros_(self.oupt.bias)

    def forward(self, x):
        z = F.relu(self.hid1(x))
        z = F.relu(self.hid2(z))
        z = F.softmax(self.oupt(z), dim=1)
        return z

    @staticmethod
    def get_criterion():
        return nn.CrossEntropyLoss()

    def get_optimizer(self, learning_rate, momentum=0.9):
        return optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)


def train(model, loader, criterion, optimizer, config, log_freq=10):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=log_freq)

    # Run training and track with wandb
    total_batches = None
    example_ct = 0  # number of examples seen
    batch_ct = 0
    for epoch in range(config.epochs):
        for _, df in enumerate(loader):
            values, labels = DPT.prepare_data(df)
            loss, accuracy = train_batch(values, labels, model, optimizer, criterion)
            example_ct += len(values)
            batch_ct += 1

            # Report metrics every 25th batch
            if ((batch_ct + 1) % log_freq) == 0:
                train_log(loss, example_ct, epoch, accuracy)


def get_accuracy(labels, outputs, correct_pred, total_pred):
    _, predicted = torch.max(outputs.data, 1)
    for label, prediction in zip(labels, predicted):
        if label == prediction:
            correct_pred[label.item()] += 1
        total_pred[label.item()] += 1
    accuracy = {f"accuracy_{k}": correct_pred[k] / n for k, n in total_pred.items()}
    return accuracy, correct_pred, total_pred


def train_batch(values, labels, model, optimizer, criterion):
    # zero the parameter gradients
    optimizer.zero_grad()

    # Forward pass ???
    outputs = model(values)
    loss = criterion(outputs, labels)

    # Backward pass ???
    loss.backward()

    # Step with optimizer
    optimizer.step()

    # Get Accuracy by Class
    labels_list = [0, 1, 2]
    total_pred = {i: 1 for i in labels_list}
    correct_pred = {i: 0 for i in labels_list}
    accuracy, correct_pred, total_pred = get_accuracy(labels, outputs, correct_pred, total_pred)

    return loss, accuracy


def train_log(loss, example_ct, epoch, accuracy):
    entry = {"epoch": epoch, "loss": loss}
    entry.update(accuracy)

    # Where the magic happens
    wandb.log(entry, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")
    if torch.isnan(loss):
        raise ValueError("Something is wrong with NN architecture! Received NANs")


def test(model, test_loader):
    labels_list = [0, 1, 2]
    model.eval()
    total_pred = {i: 1 for i in labels_list}
    correct_pred = {i: 0 for i in labels_list}
    # TODO: add my loss function

    # Run the model on some test examples
    with torch.no_grad():
        for _, df in enumerate(test_loader):
            values, labels = DPT.prepare_data(df)
            outputs = model(values)
            accuracy, correct_pred, total_pred = get_accuracy(labels, outputs, correct_pred, total_pred)

        print(accuracy, total_pred)
        wandb.log(accuracy)

    # Save the model in the exchangeable ONNX format
    torch.onnx.export(model, values, "model.onnx")
    wandb.save("model.onnx")


if __name__ == "__main__":
    TRAIN_FILES = [
        "/Users/franwe/repos/RocketLeague/data/processed/train_0_train.csv",
        "/Users/franwe/repos/RocketLeague/data/processed/train_1_train.csv",
    ]
    TEST_FILES = ["/Users/franwe/repos/RocketLeague/data/processed/all_test.csv"]
    VALUE_RANGES_FILE = "/Users/franwe/repos/RocketLeague/data/processed/all_ranges.csv"

    FEATURES = ["ball_pos_x", "ball_pos_y", "ball_pos_z", "ball_vel_x", "ball_vel_y", "ball_vel_z"]
    TARGET = ["team_scoring_within_10sec"]
    DPT = DataPreparerTorch(features=FEATURES, target=TARGET, range_filename=VALUE_RANGES_FILE, device=device)

    # Build, train and analyze the model with the pipeline
    try:
        model = model_pipeline(config)
    except Exception as e:
        print(e)
    finally:
        wandb.finish()
