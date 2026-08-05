import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch.nn.init as init
import torch.nn.functional as F
import numpy
import torchvision.models as models

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
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze操作：全局平均池化
        y = self.avg_pool(x).view(b, c)
        # Excitation操作：学习通道权重
        y = self.fc(y).view(b, c, 1, 1)
        # Scale操作：将权重应用到特征图
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    expansion=1
    def __init__(self, in_channels, out_channels, stride=1,reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SEBlock(out_channels, reduction)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride,bias=False),
                nn.BatchNorm2d(self.expansion*out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)  # 旁路
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y=self.se(y)
        y += identity  # 残差连接
        y = self.relu(y)
        return y

class ResNet(nn.Module):
    def __init__(self,block,num_block,num_classes=10):
        super(ResNet,self).__init__()
        self.in_channels=16
        self.conv1=nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(16)
        self.layer1=self._make_layer(block,16,num_block[0],stride=1)
        self.layer2=self._make_layer(block,32,num_block[1],stride=2)
        self.layer3= self._make_layer(block,64,num_block[2],stride=2)
        #self.layer4 = self._make_layer(block, 128, num_block[3], stride=2)

        self.pool=nn.MaxPool2d(2)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.dropout1=nn.Dropout(0.5)
        self.fc=nn.Linear(64,10)

    def _make_layer(self,block,out_channels,num_block,stride):
        strides=[stride]+[1]*(num_block-1)
        layer=[]
        for stride in strides:
            layer.append(block(self.in_channels,out_channels,stride))
            self.in_channels=out_channels*block.expansion
        return nn.Sequential(*layer)

    def forward(self,x):
        x=torch.relu(self.bn1(self.conv1(x)))
        #x=self.pool(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=F.avg_pool2d(x,x.size()[3])
        x=x.view(x.size(0),-1)
        #x=torch.relu(self.fc1(x))
        x=self.dropout1(x)

        x=self.fc(x)
        return x

def ResNet34():
    return ResNet(ResidualBlock,[9,9,9])

def mixup_data(x, y, alpha=1.0):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}



device=torch.device('cuda'if torch.cuda.is_available()else'cpu')
print(f"Using device:{device}")

model=ResNet34().to(device)
model.load_state_dict(torch.load('best_model_cifar10.pth'))
criterions=nn.CrossEntropyLoss(label_smoothing=0.1)
optimizers=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=1e-4)
#optimizers=torch.optim.Adam(model.parameters(),lr=0.001)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers, T_max=60,eta_min=0)
# 替换你的 scheduler
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizers, milestones=[30,50, 70], gamma=0.3)

ema = EMA(model, decay=0.999)
ema.register()



#训练
best_acc=0
train_losses=[]
train_accs=[]
test_accs=[]
epochs=100
for epoch in range(100):
    total_losses=0
    for images,labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.1)
        y=model(images)
        loss=mixup_criterion(criterions,y,labels_a,labels_b,lam)

        optimizers.zero_grad()
        loss.backward()
        optimizers.step()
        ema.update()
        total_losses+=loss.item()

    avg_losses=total_losses/len(train_loader)
    train_losses.append(avg_losses)
    print(f"Epoch{epoch+1}/{epochs}\nloss:{avg_losses:.4f}")

#评估
    ema.apply_shadow()
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
        ema.restore()
        scheduler.step()
        print(f"Test Accuracy:{test_acc:.4f}")
        test_accs.append(test_acc)

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