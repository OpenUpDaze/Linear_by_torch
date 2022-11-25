import torch
import numpy as np

# x_data = [1.0, 2.0, 3.0]
# y_data = [2.0, 4.0, 6.0]

data_path = r'F:\22postgraduate_class\machine_learning\Linear_by_torch\regress_data111.csv'
xy = np.loadtxt(data_path, delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:,:-1])
y_data = torch.from_numpy(xy[:,[-1]])

w = torch.Tensor([1.0])
w.requires_grad = True

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print("predict (befor training)", 4, forward(4).item())

for epoch in range(1000):
    for x,y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        # print('\tgrad:', x, y, w.grad.item())
        w.data = w.data - 0.01 * w.grad.data

        w.grad.data.zero_()
    print("progress:", epoch, l.item())

# print("predict (after training)", 4, forward(4).item())


