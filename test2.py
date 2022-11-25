import torch 
import numpy as np
from torch.utils.data import Dataset,DataLoader

data_path = r'F:\22postgraduate_class\machine_learning\Linear_by_torch\regress_data111.csv'
filepath = data_path
class Dataset1(Dataset):
    def __init__(self,filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:,:-1])
        self.y_data = torch.from_numpy(xy[:,[-1]])

    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len

dataset = Dataset1(filepath)
train_loader = DataLoader(dataset=dataset,
                            batch_size=32,
                            shuffle=True,
                            num_workers=0)


 # 构造模型  
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel,self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred
model = LinearModel()

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

for epoch in range(100):
    for i, data in enumerate(train_loader,0):
        # 1.Prepare data
        inputs, labels = data

        #2.Forward
        y_pred = model(inputs)
        loss = criterion(y_pred,labels)
        print(epoch,i,loss.item())

        #3.Backward
        optimizer.zero_grad()
        loss.backward()

        #4.Update
        optimizer.step()


#训练完成后的测试
# x_test = 
# y_pred = model(x_test)
# print('y_pred = ',y_pred.data)

