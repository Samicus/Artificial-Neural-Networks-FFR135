import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
from numpy import genfromtxt


class Network(torch.nn.Module):
    def __init__(self, M1, M2):
        super().__init__()

        self.M1 = M1
        self.M2 = M2

        self.fc1 = nn.Linear(2, M1)
        self.fc2 = nn.Linear(M1, M2)
        self.fc_out = nn.Linear(M2, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc_out(x)
        out = self.sigmoid(x)

        return out


M1 = 3
M2 = 3
model = Network(M1, M2)
x_train = genfromtxt("Homework_2/training_set.csv", delimiter=",")
y_train = genfromtxt("Homework_2/validation_set.csv", delimiter=",")
dataset = TensorDataset(
    torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
)

data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

loss_fn = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(25):  # Doing 20 epochs
    for (
        b_x,
        b_y,
    ) in t_train_data_loader:  # Looping through all the batches in every epoch
        pred = model(b_x)

        loss = loss_fn(pred, b_y)
        print(loss)
        # back prop
        loss.backward()
        # step and update parameters
        optimizer.step()
        # reset the gradient
        optimizer.zero_grad()
