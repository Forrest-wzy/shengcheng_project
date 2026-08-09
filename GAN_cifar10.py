import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch.nn.init as init
import numpy
import torchvision
import os


#读取数据
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    #transforms.RandomCrop(32, padding=4),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],
                         std=[0.5,0.5,0.5])
])

train_dataset = CIFAR10(root='D:\\PythonProject2', train=False, download=False, transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)

os.makedirs('real_images', exist_ok=True)
count=0
for i, (images, _) in enumerate(train_loader):
    for j in range(images.size(0)):
        img = images[j] * 0.5 + 0.5  # 反归一化到 [0,1]
        img = torchvision.transforms.ToPILImage()(img)
        img.save(f'real_images/{count:05d}.png')
        count+=1

device=torch.device('cuda'if torch.cuda.is_available()else'cpu')
print(f"Using device:{device}")

#搭建GAN生成对抗网络
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

#生成器
class Generator(nn.Module):
    def __init__(self,nz=100,ngf=64,nc=3):
        super().__init__()
        self.fc=nn.Linear(nz,256*4*4)
        self.res_blocks=nn.Sequential(
            ResidualBlock(ngf*4,ngf*4),
            ResidualBlock(ngf*4,ngf*4),
            ResidualBlock(ngf*4,ngf*4),
        )

        self.deconv=nn.Sequential(
            nn.ConvTranspose2d(ngf*4,ngf*2,4,2,1),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(),
            nn.ConvTranspose2d(ngf*2,ngf,4,2,1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
            nn.ConvTranspose2d(ngf,nc,4,2,1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1,256, 4, 4)
        x = self.res_blocks(x)
        x = self.deconv(x)
        return x
#判别器
class Discriminator(nn.Module):
    def __init__(self,nc=3,ndf=64):
        super().__init__()
        self.main=nn.Sequential(
            nn.Conv2d(nc,ndf,4,2,1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf,ndf*2,4,2,1),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf*2,ndf*4,4,2,1),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf*4,1,4,1,0),
            nn.Sigmoid()
        )

    def forward(self,x):
        x=self.main(x)
        return x.view(-1,1)

generator=Generator().to(device)
discriminator=Discriminator().to(device)
optimizer_G=torch.optim.Adam(generator.parameters(),lr=0.0002,betas=(0.5,0.999))
optimizer_D=torch.optim.Adam(discriminator.parameters(),lr=0.0002,betas=(0.5,0.999))
criterion=nn.BCELoss()

#训练
epochs=100
d_losses=[]
g_losses=[]
for epoch in range(100):
    for images, _ in train_loader:
        bs=images.size(0)
        real_images=images.to(device)

        #训练判别器

        #判别器看真实图
        real_output=discriminator(real_images)
        real_labels = torch.ones_like(real_output).to(device)
        loss_real=criterion(real_output,real_labels)

        #判别器看假图
        noise=torch.randn(bs, 100).to(device)
        fake_images = generator(noise)
        fake_output=discriminator(fake_images.detach())
        fake_labels = torch.zeros_like(fake_output).to(device)
        loss_fake=criterion(fake_output,fake_labels)

        d_loss=loss_real+loss_fake
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        #训练生成器
        noise=torch.randn(bs,100).to(device)
        fake_images=generator(noise)
        fake_output=discriminator(fake_images)
        g_loss=criterion(fake_output,real_labels)


        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1:4d} | D_loss: {d_loss:.4f} | G_loss: {g_loss:.4f}")

#可视化
def visualize_results(generator,num_samples=16,nz=100,device='cuda',save_dir='fake_images'):
    generator.eval()
    os.makedirs(save_dir,exist_ok=True)
    with torch.no_grad():
        noise=torch.randn(num_samples,nz).to(device)
        fake_images=generator(noise)
        fake_images=fake_images*0.5+0.5
        for j in range(num_samples):
            img = torchvision.transforms.ToPILImage()(fake_images[j].cpu())
            img.save(f'fake_images/{j+1}.png')
        plt.figure(figsize=(8,8))
        for i in range(num_samples):
            plt.subplot(4,4,i+1)
            img=fake_images[i].cpu().permute(1,2,0).numpy()
            plt.imshow(img)
            plt.axis('off')
        plt.show()
    generator.train()

# 训练过程中或训练结束后调用
visualize_results(generator, num_samples=16, nz=100, device=device)