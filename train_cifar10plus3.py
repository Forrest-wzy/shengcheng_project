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
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
        self.bn3=ResidualBlock(128,128,128,stride=2)
        self.pool=nn.MaxPool2d(2)
        self.dropout1=nn.Dropout(0.5)
        self.fc1=nn.Linear(128*4*4,128)
        self.fc2=nn.Linear(128,10)

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

device=torch.device('cuda'if torch.cuda.is_available()else'cpu')
print(f"Using device:{device}")

model=plusCNN().to(device)
criterions=nn.CrossEntropyLoss()
optimizers=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers, T_max=20)

#训练
best_acc=0
train_losses=[]
train_accs=[]
test_accs=[]
epochs=20
for epoch in range(20):
    total_losses=0
    for images,labels in train_loader:
        images, labels = images.to(device), labels.to(device)
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
    total_train=0
    acc=0
    acc_train=0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            y_test = model(images)
            _,pred=torch.max(y_test,1)
            total+=labels.size(0)
            acc+=(pred==labels).sum().item()
        test_acc=acc/total
        scheduler.step()
        print(f"Test Accuracy:{test_acc:.4f}")
        test_accs.append(test_acc)
    """"
        train_images, train_labels = next(iter(train_loader))
        train_images = train_images.to(device)
        train_labels = train_labels.to(device)
        train_pred = model(train_images)
        total_train+=train_labels.size(0)
        acc_train = (train_pred.argmax(1) == train_labels).sum().item()
        train_acc=acc_train/total_train
        print(f"Train Accuracy:{train_acc:.4f}")
        train_accs.append(train_acc)
    torch.cuda.empty_cache()
    model.train()
    """
    if test_acc>best_acc:
        best_acc=test_acc
        torch.save(model.state_dict(), 'best_model_cifar10.pth')
        print(f"Epoch {epoch+1}: 新最佳模型已保存 (Acc: {best_acc:.4f})")

plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss')
plt.show()
plt.plot(test_accs,label='Test Accuracy')
#plt.plot(train_accs,label='Train Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"参数量: {count_parameters(model):,}")

classes=('airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck')
def visualize(model, test_loader, num_img=10):
    model.eval()
    with torch.no_grad():
        images, labels = next(iter(test_loader))  # 取第一批
        images,labels=images.to(device),labels.to(device)
        indices = torch.randperm(images.size(0))[:num_img]
        images = images[indices]
        labels = labels[indices]
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