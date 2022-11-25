import torch 
import numpy as np
import matplotlib.pyplot as plt

# 存在问题，更换损失函数后，基本能算，损失为2点多
data_path = r'F:\22postgraduate_class\machine_learning\Linear_by_torch\regress_data111.csv'
xy = np.loadtxt(data_path, delimiter=',', dtype=np.float32)
print(xy.dtype)

x_data = torch.from_numpy(xy[:,:-1])
y_data = torch.from_numpy(xy[:,[-1]])

# plt.scatter(x_data.data, y_data.data)
# print('x_data = ',x_data.data)
# print('y_data = ',y_data.data)

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel,self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred
model = LinearModel()

# criterion = torch.nn.MSELoss(size_average=False)
criterion = torch.nn.L1Loss(size_average=None,reduce=None)
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    if loss.data < 2:
        break
    print(epoch,loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


#训练完成后的测试
# x_test = 
# y_pred = model(x_test)
# print('y_pred = ',y_pred.data)
# 输出权重和偏置
print('w = ',model.linear.weight.item())
print('b = ',model.linear.bias.item())
# 画图
w = model.linear.weight.item()
b = model.linear.bias.item()
x = np.linspace(0, 25, 100)
f_loss = loss.item()
f = x * w + b

fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle(f"test3 loss = {f_loss}")

ax.plot(x, f, 'r', label='预测值')
ax.scatter(x_data.data, y_data.data, label='训练数据')
plt.show()
