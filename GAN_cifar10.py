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
import torch.optim as optim

# 读取数据
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    # transforms.RandomCrop(32, padding=4),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

train_dataset = CIFAR10(root='D:\\PythonProject2', train=False, download=False, transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)

os.makedirs('real_images', exist_ok=True)
count = 0
for i, (images, _) in enumerate(train_loader):
    for j in range(images.size(0)):
        img = images[j] * 0.5 + 0.5  # 反归一化到 [0,1]
        img = torchvision.transforms.ToPILImage()(img)
        img.save(f'real_images/{count:05d}.png')
        count += 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device:{device}")


# 搭建GAN生成对抗网络
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


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
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SEBlock(out_channels, reduction)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)  # 旁路
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.se(y)
        y += identity  # 残差连接
        y = self.relu(y)
        return y


# 可视化
def visualize_results(generator, num_samples=16, nz=100, device='cuda', save_dir='fake_images'):#####!!!!!
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, nz).to(device)
        fake_images = generator(noise)
        fake_images = fake_images * 0.5 + 0.5
        plt.figure(figsize=(8, 8))
        for i in range(num_samples):
            plt.subplot(4, 4, i + 1)
            img = fake_images[i].cpu().permute(1, 2, 0).numpy()
            plt.imshow(img)
            plt.axis('off')
        plt.show()
    generator.train()


"""""
#生成器
class Generator(nn.Module):
    def __init__(self,nz=128,ngf=64,nc=3):
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
            nn.ConvTranspose2d(ngf,ngf*0.5, 4, 2, 1),
            nn.BatchNorm2d(ngf*0.5),
            nn.ReLU(),
            nn.ConvTranspose2d(ngf*0.5,nc,4,2,1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1,256, 4, 4)
        x = self.res_blocks(x)
        x = self.deconv(x)
        return x

"""


class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features=100, kernel_dim=50):
        super().__init__()
        self.T = nn.Parameter(torch.randn(in_features, out_features, kernel_dim))

    def forward(self, x):
        # x: [batch, in_features]
        # M = x @ T: [batch, out_features, kernel_dim]
        M = torch.matmul(x, self.T.view(x.size(1), -1))  # [batch, out_features * kernel_dim]
        M = M.view(-1, self.T.size(1), self.T.size(2))  # [batch, out_features, kernel_dim]

        # 计算 batch 内两两之间的 L1 距离
        M_expanded = M.unsqueeze(0)  # [1, batch, out_features, kernel_dim]
        M_transposed = M.unsqueeze(1)  # [batch, 1, out_features, kernel_dim]
        diff = torch.abs(M_expanded - M_transposed)
        similarity = torch.exp(-torch.sum(diff, dim=3))  # [batch, batch, out_features]
        similarity = torch.sum(similarity, dim=1)  # [batch, out_features]
        return torch.cat([x, similarity], dim=1)

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):####!!!!
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 输入: 100 维噪声
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 4×4 → 8×8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 8×8 → 16×16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 16×16 → 32×32
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)  # 把噪声变成 4D 张量
        out = self.main(z)
        # print(f"deconv 输出形状: {out.shape}")
        return out


# 判别器
class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        """""
        self.classifier=nn.Sequential(
            nn.Conv2d(ndf * 4, 1, 4, 1, 0),
            nn.Dropout(0.3),
            nn.Sigmoid()
        )
        """
        self.fc = nn.Linear(ndf * 4 * 4 * 4, 128)  # 根据你的特征图尺寸调整
        self.minibatch = MinibatchDiscrimination(128, out_features=100, kernel_dim=50)
        self.final = nn.Linear(128 + 100, 1)  # 128 原始特征 + 100 小批量特征
        self.sigmoid = nn.Sigmoid()

    def forward(self, x,return_features=False):
        features= self.features(x)
        features=features.view(features.size(0),-1)
        if return_features:
            return features
        fc_out=self.fc(features)
        minibatch_out=self.minibatch(fc_out)
        out=self.final(minibatch_out)
        # x = nn.AdaptiveAvgPool2d(1)(x)
        return self.sigmoid(out).view(-1,1)


nz = 100######!!!!!
generator = Generator().to(device)
discriminator = Discriminator().to(device)
generator.apply(weights_init)
discriminator.apply(weights_init)

print("Generator parameters:", sum(p.numel() for p in generator.parameters()))
print("Discriminator parameters:", sum(p.numel() for p in discriminator.parameters()))
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
# WGAN 建议使用 RMSprop 或 SGD，不要用 Adam（Adam 在 WGAN 中可能不稳定）
# optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=0.0001)
# optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=0.0001)
criterion = nn.BCELoss()

# 训练
epochs = 200
d_losses = []
g_losses = []
for epoch in range(200):
    for images, _ in train_loader:
        bs = images.size(0)
        real_images = images.to(device)

        # 训练判别器

        # 判别器看真实图
        real_output = discriminator(real_images)
        real_labels = torch.ones_like(real_output).to(device) * 0.9
        loss_real = criterion(real_output, real_labels)

        # 判别器看假图
        noise = torch.randn(bs, nz).to(device)
        fake_images = generator(noise)
        fake_output = discriminator(fake_images.detach())
        fake_labels = torch.zeros_like(fake_output).to(device) + 0.1
        loss_fake = criterion(fake_output, fake_labels)

        d_loss = loss_real + loss_fake
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # 训练生成器
        real_features = discriminator(real_images, return_features=True)
        noise = torch.randn(bs, nz).to(device)
        fake_images = generator(noise)
        fake_output = discriminator(fake_images)
        fake_features = discriminator(fake_images, return_features=True)
        #g_loss = criterion(fake_output, real_labels)
        g_loss_adv = criterion(fake_output, real_labels)
        g_loss_fm=torch.mean(torch.abs(real_features.mean(0)-fake_features.mean(0)))
        lambda_fm=10
        g_loss=g_loss_adv+lambda_fm*g_loss_fm


        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1:4d} | D_loss: {d_loss:.4f} | G_loss: {g_loss:.4f}")
        if (epoch + 1) % 10== 0:
            visualize_results(generator, num_samples=16, nz=nz, device=device)


def generate_fake_images(generator, num_images=10000, batch_size=64, nz=100, device='cuda', save_dir='fake_images'):######!!!!!
    os.makedirs(save_dir, exist_ok=True)
    generator.eval()
    count = 0
    with torch.no_grad():
        for i in range(0, num_images, batch_size):
            bs = min(batch_size, num_images - i)
            noise = torch.randn(bs, nz).to(device)
            fake = generator(noise)
            fake = fake * 0.5 + 0.5
            for j in range(bs):
                img = torchvision.transforms.ToPILImage()(fake[j].cpu())
                img.save(f'{save_dir}/{count:05d}.png')
                count += 1
    print(f"已生成 {count} 张假图片到 {save_dir}/")
    generator.train()


generate_fake_images(generator, num_images=10000, batch_size=64, device=device)

# 训练结束后
torch.save(generator.state_dict(), 'generator_54.84.pth')
print("模型已保存为 generator_54.84.pth")