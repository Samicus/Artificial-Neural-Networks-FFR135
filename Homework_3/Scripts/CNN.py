import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

REBUILD_DATA = True


class CNN_1(nn.Module):
    def __init__(self):
        super(CNN_1, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=20,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(3380, 100)
        self.fc2 = nn.Linear(100, 10)

        nn.init.normal_(self.fc1.weight, mean=0, std=0.01)
        nn.init.normal_(self.fc1.bias, mean=0, std=0.01)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.01)
        nn.init.normal_(self.conv1.weight, mean=0, std=0.01)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN_2(nn.Module):
    def __init__(self):
        super(CNN_2, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=20,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.batch1 = nn.BatchNorm2d(num_features = 20, eps= 1e-05, momentum = 0.9, affine = True, track_running_stats=True)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=20,
            out_channels=30,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.batch2 = nn.BatchNorm2d(num_features = 30, eps= 1e-05, momentum = 0.9, affine = True, track_running_stats=True)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv3 = nn.Conv2d(
            in_channels=30,
            out_channels=50,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.batch3 = nn.BatchNorm2d(num_features = 50, eps= 1e-05, momentum = 0.9, affine = True, track_running_stats=True)


        self.fc = nn.Linear(2450, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.batch1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.batch2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.batch3(x))
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


def check_accuaracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuaracy {float(num_correct)/float(num_samples)*100:.2f}"
        )
    model.train()


if __name__ == "__main__":

    # Initialize tensors

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Hyperparameters
    in_channels = 1
    num_classes = 10
    learning_rate = 0.001
    num_epochs = 1
    batch_size = 64

    # Load Data
    train_dataset = datasets.MNIST(
        root="../dataset/", train=True, transform=transforms.ToTensor(), download=True
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataset = datasets.MNIST(
        root="../dataset/", train=False, transform=transforms.ToTensor(), download=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # Initialize network
    model = CNN().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train Network

    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            # get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

    # check the accuaracy on training & test
    print("result on training set:")
    check_accuaracy(train_loader, model)
    print("result on test set:")
    check_accuaracy(test_loader, model)
