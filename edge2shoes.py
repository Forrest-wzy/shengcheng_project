from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.onnx.symbolic_opset9 import detach
from torchvision import transforms
import torchvision
import os
import torch.optim as optim
import torch.nn.functional as F

from pairdataset import PairedDataset

transforms_train=transforms.Compose([
    transforms.Resize((256,256)),
    #transforms.RandomRotation(5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

transforms_test=transforms.Compose([
transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

train_dataset=PairedDataset('D:\\PythonProject2\\edges2shoes\\train',transform=transforms_train,split_radio=0.5)
train_loader=DataLoader(train_dataset,batch_size=4,shuffle=True,num_workers=0)

test_dataset=PairedDataset('D:\\PythonProject2\\edges2shoes\\val',transform=transforms_test,split_radio=0.5)
test_loader=DataLoader(test_dataset,batch_size=4,shuffle=False,num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device:{device}")
print(torch.cuda.get_device_name(0))

#搭建网络
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
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
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.InstanceNorm2d(out_channels)

        self.se = SEBlock(out_channels, reduction)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride,bias=False),
                nn.InstanceNorm2d(self.expansion*out_channels)
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

class lightGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 32
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.bn1 = nn.InstanceNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(ResidualBlock, 32, 3, stride=1)  # 256×256
        self.layer2 = self._make_layer(ResidualBlock, 64, 3, stride=2)  # 128×128
        self.layer3 = self._make_layer(ResidualBlock, 128, 3, stride=2)  # 64×64
        self.layer4 = self._make_layer(ResidualBlock, 256, 3, stride=2)  # 32x32
        self.bottleneck = self._make_layer(ResidualBlock, 256, 3, stride=1)
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, 2, 1),  # 输入 64 通道（拼接后）
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 4, 2, 1),  # 输入 32 通道（拼接后）
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.decoder4 = nn.Sequential(
            nn.Conv2d(64, 3, 3, 1, 1),  # (32+32拼接)
            nn.Tanh()
        )

    def _make_layer(self,block,out_channels,num_block,stride):
        strides=[stride]+[1]*(num_block-1)
        layer=[]
        for s in strides:
            layer.append(block(self.in_channels,out_channels,s))
            self.in_channels=out_channels*block.expansion
        return nn.Sequential(*layer)

    def forward(self,x):
        e1 = self.relu(self.bn1(self.conv1(x)))  # 256×256
        e2 = self.layer1(e1)  # 256×256
        e3 = self.layer2(e2)  # 128×128
        e4 = self.layer3(e3)  # 64×64
        e5 = self.layer4(e4)

        d1 = self.bottleneck(e5)
        d1 = self.decoder1(d1)
        d1 = torch.cat([d1, e4], dim=1)
        d2 = self.decoder2(d1)
        d2 = torch.cat([d2, e3], dim=1)
        d3 = self.decoder3(d2)  # 256×256 → 输出
        d3 = torch.cat([d3, e2], dim=1)  # 拼接 e2 (256x256)

        out = self.decoder4(d3)  # 输出
        return out


class Patchblock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1)]
        if use_bn:
            layers.append(nn.InstanceNorm2d(out_ch))  # 推荐用 InstanceNorm，对 batch_size=1 更友好
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class PatchGan(nn.Module):
    def __init__(self, in_ch=3, ndf=64, out_ch=1):
        super().__init__()
        # 第1层：不加 BN，保留原始图像信息
        self.inc = nn.Sequential(
            nn.Conv2d(in_ch * 2, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 第2层：ndf -> ndf*2
        self.con1 = Patchblock(ndf, ndf * 2, stride=2, use_bn=True)

        # 第3层：ndf*2 -> ndf*4，stride=1 减缓下采样，扩大感受野
        self.con2 = Patchblock(ndf * 2, ndf * 4, stride=1, use_bn=True)

        # 第4层（输出层）：ndf*4 -> 1，stride=1，输出判别图
        # 绝对不加 BN 和 Dropout
        self.con3 = nn.Sequential(
            nn.Conv2d(ndf * 4, out_ch, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.inc(x)
        x = self.con1(x)
        x = self.con2(x)
        return self.con3(x)

class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer_ids=[2, 7, 12, 21,26]):
        super().__init__()
        import torchvision.models as models
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.layers = nn.ModuleList([vgg[i] for i in range(max(layer_ids) + 1)])
        self.layer_ids = set(layer_ids)
        for p in self.layers.parameters():
            p.requires_grad = False

            # ImageNet 归一化参数
        self.register_buffer('imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        # 1. [-1, 1] -> [0, 1]
        x = (x + 1.0) / 2.0
        y = (y + 1.0) / 2.0

        # 2. 【核心修正】强制 Resize 到 224x224！
        # 必须使用双线性插值，确保梯度能正常回传
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        y = torch.nn.functional.interpolate(y, size=(224, 224), mode='bilinear', align_corners=False)

        # 3. ImageNet 标准化
        x = (x - self.imagenet_mean) / self.imagenet_std
        y = (y - self.imagenet_mean) / self.imagenet_std

        loss = 0.0
        for i, layer in enumerate(self.layers):
            x = layer(x)
            y = layer(y)
            if i in self.layer_ids:
                #norm = 1.0 / (x.shape[1] * x.shape[2] * x.shape[3])
                loss +=nn.functional.l1_loss(x, y)
        #print(f"VGG Loss: {loss.item():.6f}")  # 加这行
        return loss

vgg_loss = VGGPerceptualLoss().to(device)
lambda_vgg =0


class SobelEdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Sobel 核（1通道输入，1通道输出）
        self.register_buffer('sobel_x',
            torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],dtype=torch.float32))
        self.register_buffer('sobel_y',
            torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]],dtype=torch.float32))

    def forward(self, pred, target):
        # 手动转灰度：加权平均，避免 conv2d 通道不匹配
        pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]

        # 创建 Mask：非白色区域为 1，白色背景为 0
        mask = (target_gray < 0.85).float()

        # 计算梯度（此时输入是 1 通道，和 sobel 核匹配）
        grad_pred_x = F.conv2d(pred_gray, self.sobel_x, padding=1)
        grad_pred_y = F.conv2d(pred_gray, self.sobel_y, padding=1)
        edge_pred = torch.sqrt(grad_pred_x**2 + grad_pred_y**2 + 1e-8)

        grad_target_x = F.conv2d(target_gray, self.sobel_x, padding=1)
        grad_target_y = F.conv2d(target_gray, self.sobel_y, padding=1)
        edge_target = torch.sqrt(grad_target_x**2 + grad_target_y**2 + 1e-8)

        # 只计算线条区域的 L1 Loss
        loss = F.l1_loss(edge_pred * mask, edge_target * mask)
        return loss
edge_detector = SobelEdgeLoss().to(device)
lambda_edge = 8

def tv_loss(img):
    """
    计算图像的总变分损失
    img: 形状为 [batch_size, channels, height, width] 的张量
    """
    # 水平方向：每个像素和它右边像素的差
    h_tv = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    # 垂直方向：每个像素和它下面像素的差
    w_tv = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return h_tv + w_tv
lambda_tv = 0.00001

#准备
generator=lightGenerator().to(device)
discriminator=PatchGan(in_ch=3,ndf=64,out_ch=1).to(device)

total_params_G = sum(p.numel() for p in generator.parameters())
total_params_D = sum(p.numel() for p in discriminator.parameters())
print(f'G 参数量: {total_params_G}')
print(f'D 参数量: {total_params_D}')
print(f'总计: {total_params_G + total_params_D}')


lambda_gan=1
criterion_gan=nn.BCEWithLogitsLoss()
lambda_L1=150
criterion_L1=nn.L1Loss()
optimizer_G=optim.Adam(generator.parameters(),lr=0.0002,betas=(0.5,0.999))
optimizer_D=optim.Adam(discriminator.parameters(),lr=0.00005,betas=(0.5,0.999))

def lambda_rule(epoch):
    # 前 50 轮保持 100% 学习率（让模型充分探索）
    # 50 轮后在接下来的 50 轮内线性衰减到 0
    if epoch < 50:
        return 1.0
    else:
        return max(0.1, 1.0 - (epoch - 50) / 50.0)
scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda_rule)

start_epoch =10
generator.load_state_dict(torch.load(f'edges2shoes_generator1_epoch_{start_epoch}.pth', map_location=device))
discriminator.load_state_dict(torch.load(f'edges2shoes_discriminator1_epoch_{start_epoch}.pth', map_location=device))
#训练
import torch
torch.cuda.empty_cache()
epochs=100

for epoch in range(start_epoch+1,epochs):
    for i,(line,photo,filename) in enumerate(train_loader):
        bs=line.size(0)
        line=line.to(device)
        photo=photo.to(device)

        if epoch < 20:
            w_l1, w_vgg, w_tv = lambda_L1, lambda_vgg, 1e-5
            w_gan, w_edge = 1.0, 8.0
        elif epoch < 35:
            progress = (epoch - 20) / 15.0
            w_l1, w_vgg, w_tv = lambda_L1, lambda_vgg+1, 1e-5
            w_gan =1.0
            w_edge = 5.0
        elif epoch < 50:
            progress = (epoch - 35) / 15.0
            w_l1, w_vgg, w_tv = lambda_L1*0.5, lambda_vgg+1.5, 1e-6
            w_gan = 1.0
            w_edge = progress * 5.0
        else:
            w_l1, w_vgg, w_tv = lambda_L1 * 0.1, lambda_vgg+2, 1e-6
            w_gan, w_edge = 1.0, 2.0

        optimizer_D.zero_grad()
        fake_photo = generator(line).detach()

        real_pairs = torch.cat([line, photo], dim=1)
        fake_pairs = torch.cat([line, fake_photo], dim=1)

        pred_real = discriminator(real_pairs)
        pred_fake = discriminator(fake_pairs)

        loss_D_real = criterion_gan(pred_real, torch.ones_like(pred_real) * 0.9)
        loss_D_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))
        loss_D = (loss_D_real + loss_D_fake) * 0.5

        loss_D.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
        optimizer_D.step()

        # ========== 训练 G（每个 iteration 都执行）==========
        optimizer_G.zero_grad()
        fake_photo = generator(line)

        # GAN Loss（前20轮 w_gan=0，乘了也是0）
        fake_pairs = torch.cat([line, fake_photo], dim=1)
        pred_fake_for_G = discriminator(fake_pairs)
        loss_G_gan = criterion_gan(pred_fake_for_G, torch.ones_like(pred_fake_for_G)) * w_gan

        # L1 Loss
        loss_G_l1 = criterion_L1(fake_photo, photo) * w_l1

        # VGG Loss
        loss_G_vgg = vgg_loss(fake_photo, photo) * w_vgg

        # TV Loss
        loss_G_tv = tv_loss(fake_photo) * w_tv

        # Edge Loss（前35轮 w_edge=0，乘了也是0）
        loss_G_edge = edge_detector(fake_photo, photo) * w_edge

        # 总 Loss
        loss_G = loss_G_gan + loss_G_l1 + loss_G_vgg + loss_G_tv + loss_G_edge

        loss_G.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        optimizer_G.step()

    scheduler_G.step()
    scheduler_D.step()

    if epoch % 5 == 0:
        # 如果是预热阶段，只打印 G_loss
            #print(f"Epoch {epoch}: D_loss: {loss_D.item():.4f}, G_loss: {loss_G.item():.4f}")
         print(f"Epoch [{epoch}/{epochs}] | "
              f"D_loss: {loss_D.item():.4f} | G_loss: {loss_G.item():.4f} | "
              f"G_L1: {loss_G_l1.item():.4f} | G_VGG: {loss_G_vgg.item():.8f} | "
              f"G_Edge: {loss_G_edge.item():.4f} | G_tv: {loss_G_tv.item():.8f}")
         torch.save(generator.state_dict(), f'edges2shoes_generator1_epoch_{epoch}.pth')
         torch.save(discriminator.state_dict(), f'edges2shoes_discriminator1_epoch_{epoch}.pth')
         torch.save(optimizer_G.state_dict(), f'edges2shoes_optimizer_G_epoch_{epoch}.pth')
         torch.save(optimizer_D.state_dict(), f'edges2shoes_optimizer_D_epoch_{epoch}.pth')

    generator.eval()
    os.makedirs('generated_shoes', exist_ok=True)
    with torch.no_grad():
        test_line, test_photo, test_filename = next(iter(test_loader))
        test_line = test_line.to(device)
        fake_photo = generator(test_line)
        fake_photo = fake_photo * 0.5 + 0.5
        img = torchvision.transforms.ToPILImage()(fake_photo[1].cpu())
        img.save(f'generated_shoes/epoch_{epoch}_fake.png')
    generator.train()

    # 测试
    generator.eval()
    os.makedirs('generated_shoes', exist_ok=True)
    count = 0
    with torch.no_grad():
        for line, photo,filename in test_loader:
            line = line.to(device)
            fake_photo = generator(line)
            fake_photo = fake_photo * 0.5 + 0.5
            for i in range(fake_photo.size(0)):
                img = torchvision.transforms.ToPILImage()(fake_photo[i].cpu())
                img.save(f'generated_shoes/{filename[i]}')
                count += 1
    print(f"已生成 {count} 张假图片到 generated_shoes/")

#可视化
def visualize_results(generator, test_loader, num_samples=5,epoch=0):
    generator.eval()

    # 获取一批数据
    line, photo,filename = next(iter(test_loader))
    actual_samples = min(num_samples, line.size(0))

        # 截取并移动到设备
    line = line[:actual_samples].to(device)
    photo = photo[:actual_samples].to(device)

        # 生成假图片
    with torch.no_grad():
        fake_photo = generator(line)

    fig, axes = plt.subplots(actual_samples, 3, figsize=(9, 3 * actual_samples), squeeze=False)

    for i in range(actual_samples):
        input_img = line[i].cpu().permute(1, 2, 0) * 0.5 + 0.5
        if input_img.shape[2] == 1:
            axes[i, 0].imshow(input_img.squeeze(), cmap='gray')
        else:
            axes[i, 0].imshow(input_img)
        axes[i, 0].set_title('Input (Line)')
        axes[i, 0].axis('off')


        fake_img = fake_photo[i].cpu().permute(1, 2, 0) * 0.5 + 0.5
            # 防止数值溢出导致颜色诡异，必须 clamp
        fake_img = torch.clamp(fake_img, 0, 1)
        axes[i, 1].imshow(fake_img)
        axes[i, 1].set_title('Generated')
        axes[i, 1].axis('off')

        real_img = photo[i].cpu().permute(1, 2, 0) * 0.5 + 0.5
        axes[i, 2].imshow(real_img)
        axes[i, 2].set_title('Real')
        axes[i, 2].axis('off')
    plt.tight_layout()
    plt.savefig(f'result_{epoch}.png')
    plt.show()

visualize_results(generator, test_loader, num_samples=5,epoch=epoch)