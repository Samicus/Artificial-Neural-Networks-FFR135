import CNN
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Hyperparameters
momentum = 0.9
n_epochs = 60
batch_size = 8192
learning_rate = 0.01
validation_patience = 5
validation_frequency = 5


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


# Initialize tensors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load Data
train_dataset = datasets.MNIST(
    root="../dataset/", train=True, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(
    root="../dataset/", train=False, transform=transforms.ToTensor(), download=True
)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = CNN.CNN_1().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# Train Network

for epoch in range(n_epochs):
    print("EPOCH", epoch)
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

        print("step", batch_idx, ", loss:", loss.item())

    # check the accuaracy on training & test
    if (batch_idx + (epoch * 8)) % 30 == 0:
        check_accuaracy(train_loader, model)


print("result on test set:")
check_accuaracy(test_loader, model)
