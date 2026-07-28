import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
#读取数据集
import sys,os

import torch.optim

sys.path.append(os.pardir)
from mnist import load_mnist
(x_train,t_train),(x_test,t_test)=load_mnist(flatten=True,normalize=False)
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

#定义网络
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(784,100),
            nn.ReLU(),
            nn.Linear(100,10)
        )
    def forward(self,x):
        return self.net(x)

model=MLP()
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

#训练
epochs=10
batch_size=100

x_train_t = torch.tensor(x_train, dtype=torch.float32)
t_train_t = torch.tensor(t_train, dtype=torch.long)
x_test_t = torch.tensor(x_test, dtype=torch.float32)
t_test_t = torch.tensor(t_test, dtype=torch.long)

for epoch in range(10):
    total_loss=0
    indices=torch.randperm(x_train_t.size(0))
    x_train_shuffled=x_train_t[indices]
    t_train_shuffled=t_train_t[indices]

    for i in range(0,len(x_train_t),batch_size):
        x_batch=x_train_shuffled[i:i+batch_size]
        t_batch=t_train_shuffled[i:i+batch_size]

        y_batch=model(x_batch)
        loss=criterion(y_batch,t_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()

    avg_loss=total_loss/(len(x_train_t)//batch_size)
    print(f"Epoch{epoch+1}/{epochs},loss:{avg_loss:.4f}")

#评估
with torch.no_grad():
    y_test_pred=model(x_test_t)
    test_accuracy=(y_test_pred.argmax(1)==t_test_t).float().mean().item()
    print(f"Test Accuracy:{test_accuracy:.4f}")

#可视化
classes=('0','1','2','3','4','5','6','7','8','9')
def visualize(model,x_data,t_data,num_img=10):
         model.eval()
         with torch.no_grad():
             total=len(x_data)
             indices=torch.randperm(total)[:num_img]
             x_sample=x_data[indices]
             t_sample=t_data[indices]
             y_pred=model(x_sample)
             _,predicted=torch.max(y_pred,1)
             plt.figure(figsize=(12,6))

             for i in range(num_img):
                 plt.subplot(2,5,i+1)
                 img=x_sample[i].reshape(28,28)
                 plt.imshow(img,cmap='gray')
                 plt.title(f"True:{t_sample[i]},predicted:{predicted[i].item()}")
                 plt.axis('off')

             plt.show()

visualize(model,x_test_t,t_test_t,num_img=10)
