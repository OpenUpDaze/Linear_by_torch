import torch 
import numpy as np

# 存在问题，迭代几次后就容易出现损失位inf 或Nan的情况
data_path = 'regress_data1.csv'
xy = np.loadtxt(data_path, delimiter=',', dtype=np.float32)
print(xy.dtype)

x_data = torch.from_numpy(xy[:,:-1])
y_data = torch.from_numpy(xy[:,[-1]])

print('x_data = ',x_data.data)
print('y_data = ',y_data.data)

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel,self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred
model = LinearModel()

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(),lr=0.001)

for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch,loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#训练完成后的测试
# x_test = 
# y_pred = model(x_test)
# print('y_pred = ',y_pred.data)

