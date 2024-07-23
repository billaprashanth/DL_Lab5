import torch
import torch.nn as nn
import torch.optim as otim
import torch.nn.functional as F

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
optimizer = otim.Adam(model.parameters(), lr = 0.001)

# prediction

test = model.forward(x)
# print(test[:5])

# train the model
for epoch in range(10):
    model.forward(x)
print(x)