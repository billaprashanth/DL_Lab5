import torch
import torch.nn as nn
import torch.optim as otim
import torch.nn.functional as F
# import matplotlib.pylot as plt
# from keras.callbacks import ModelCheckpoint

#sample data
x=torch.rand([100, 10])
y=torch.rand([100, 1])
print(x)
print(y)

# build the architecture
class simplenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 1)
    # processing
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

# object creation

model = simplenet()
# define loss and optimizer
criterian = nn.MSELoss()
optimizer = otim.SGD(model.parameters(), lr = 0.001)

# prediction

test = model.forward(x)
# print(test[:5])

# train the model
for epoch in range(10):
    result = model.forward(x)
    # find the loss
    loss = criterian(result, y)
    # backpropaation
    loss.backward()
    # optimization
    optimizer.step()
    optimizer.zero_grad()
    print(f"eopch - {epoch+1} : loss = { loss.item()} ")


