import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
import numpy
#读取数据
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = CIFAR10(root='D:\\PythonProject2', train=True, download=False, transform=transform_train)
test_dataset = CIFAR10(root='D:\\PythonProject2', train=False, download=False, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

#定义网络
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, mid_channels,out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)  # 旁路
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y += identity  # 残差连接
        y = self.relu(y)
        return y

class plusCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,32,kernel_size=3,padding=1)
        self.bn1=ResidualBlock(32,64,64,stride=1)
        self.bn2=ResidualBlock(64,128,128,stride=1)
        self.bn3=ResidualBlock(128,256,256,stride=2)
        self.pool=nn.MaxPool2d(2)
        self.dropout1=nn.Dropout(0.5)
        self.fc1=nn.Linear(256*4*4,256)
        self.fc2=nn.Linear(256,10)

    def forward(self,x):
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(self.bn1(x))
        x=self.bn2(x)
        x=self.bn3(x)
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        x=self.dropout1(x)
        x=self.fc2(x)
        return x

model=plusCNN()
criterions=nn.CrossEntropyLoss()
optimizers=torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers, T_max=20)

#训练
train_losses=[]
train_accs=[]
test_accs=[]
epochs=10
for epoch in range(10):
    total_losses=0
    for images,labels in train_loader:
        y=model(images)
        loss=criterions(y,labels)

        optimizers.zero_grad()
        loss.backward()
        optimizers.step()

        total_losses+=loss.item()

    avg_losses=total_losses/len(train_loader)
    train_losses.append(avg_losses)
    print(f"Epoch{epoch+1}/{epochs}\nloss:{avg_losses:.4f}")

#评估
    model.eval()
    total=0
    acc=0
    with torch.no_grad():
        for images, labels in test_loader:
            y_test = model(images)
            _,pred=torch.max(y_test,1)
            total+=labels.size(0)
            acc+=(pred==labels).sum().item()
        test_acc=acc/total
        scheduler.step()
        print(f"Test Accuracy:{test_acc:.4f}")
        test_accs.append(test_acc)

        train_images, train_labels = next(iter(train_loader))
        train_pred = model(train_images)
        train_acc = (train_pred.argmax(1) == train_labels).float().mean().item()
        print(f"Train Accuracy:{train_acc:.4f}")
        train_accs.append(train_acc)
    model.train()

plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss')
plt.show()
plt.plot(test_accs,label='Test Accuracy')
plt.plot(train_accs,label='Train Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

classes=('airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck')
def visualize(model, test_loader, num_img=10):
    model.eval()
    with torch.no_grad():
        images, labels = next(iter(test_loader))  # 取第一批
        images = images[:num_img]
        labels = labels[:num_img]
        y_pred = model(images)
        _, predicted = torch.max(y_pred, 1)
        plt.figure(figsize=(12, 6))

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        for i in range(num_img):
            plt.subplot(2, 5, i + 1)

            img = images[i].cpu()  # 移到 CPU（如果数据在 GPU）
            img = img * std + mean  # 反归一化
            img = img.clamp(0, 1)
            img = img.permute(1,2,0).numpy()
            plt.imshow(img)
            plt.title(f"True:{classes[labels[i].item()]}\npredicted:{classes[predicted[i].item()]}")
            plt.axis('off')

        plt.show()
visualize(model,test_loader,num_img=10)