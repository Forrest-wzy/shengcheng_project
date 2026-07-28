import torch
import torch.nn as nn
import matplotlib.pyplot as plt
#读取数据
import sys,os
sys.path.append(os.pardir)
from cifar10 import load_cifar10
(x_train,t_train),(x_test,t_test)=load_cifar10(flatten=False,normalize=False)
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

#定义网络
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,6,kernel_size=5)
        self.pool=nn.AvgPool2d(kernel_size=2,stride=2)
        self.conv2=nn.Conv2d(6,16,kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)

        self.fc1=nn.Linear(120,84)
        self.fc2=nn.Linear(84,10)

    def forward(self,x):
        x=self.pool(torch.tanh_(self.conv1(x)))
        x=self.pool(torch.tanh_(self.conv2(x)))
        x=self.conv3(x)
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        x=torch.tanh_(x)
        x=self.fc2(x)
        return x

model=LeNet()
criterions=nn.CrossEntropyLoss()
optimizers=torch.optim.Adam(model.parameters(),lr=0.001)

#训练
epochs=10
batch_sizes=100
x_train_c=torch.from_numpy(x_train).float().permute(0,3,1,2)
t_train_c=torch.from_numpy(t_train).long()
x_test_c=torch.from_numpy(x_test).float().permute(0, 3, 1, 2)
t_test_c=torch.from_numpy(t_test).long()
for epoch in range(10):
    indices=torch.randperm(x_train_c.size(0))
    x_train_s=x_train_c[indices]
    t_train_s=t_train_c[indices]
    total_losses=0
    for i in range(0,len(x_train_c),batch_sizes):
        x_bat=x_train_s[i:i+batch_sizes]
        t_bat=t_train_s[i:i+batch_sizes]

        y_bat=model(x_bat)
        loss=criterions(y_bat,t_bat)
        optimizers.zero_grad()
        loss.backward()
        optimizers.step()

        total_losses+=loss.item()

    avg_losses=total_losses/(len(x_train_c)//batch_sizes)
    print(f"Epoch{epoch+1}/{epochs},loss:{avg_losses:.4f}")

#评估
with torch.no_grad():
    y_test_predicted=model(x_test_c)
    test_acc=(y_test_predicted.argmax(1)==t_test_c).float().mean().item()
    print(f"Test Accuracy:{test_acc:.4f}")

classes=('airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck')
def visualize(model, x_data, t_data, num_img=10):
    model.eval()
    with torch.no_grad():
        total = len(x_data)
        indices = torch.randperm(total)[:num_img]
        x_sample = x_data[indices]
        t_sample = t_data[indices]
        y_pred = model(x_sample)
        _, predicted = torch.max(y_pred, 1)
        plt.figure(figsize=(12, 6))

        for i in range(num_img):
            plt.subplot(2, 5, i + 1)
            img = x_sample[i].permute(1,2,0)
            img=img/255.0
            plt.imshow(img)
            plt.title(f"True:{classes[t_sample[i].item()]},predicted:{classes[predicted[i].item()]}")
            plt.axis('off')

        plt.show()
visualize(model,x_test_c,t_test_c,num_img=10)