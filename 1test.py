import os
import random

ir_dir = r'D:\PythonProject2\M3FD_Fusion\ir'
vis_dir = r'D:\PythonProject2\M3FD_Fusion\M3FD_Detection\vi'

# 1. 列出所有红外文件，筛选出有配对的
ir_files = sorted([f for f in os.listdir(ir_dir)
                   if f.endswith(('.jpg', '.png', '.jpeg'))])
paired = [f for f in ir_files
          if os.path.exists(os.path.join(vis_dir, f))]

print(f"总配对: {len(paired)}")

# 2. 随机打乱
random.seed(42)
random.shuffle(paired)

# 3. 8:1:1 划分
n = len(paired)
n_train = int(n * 0.8)
n_val = int(n * 0.1)

train_files = paired[:n_train]
val_files = paired[n_train:n_train + n_val]
test_files = paired[n_train + n_val:]

print(f"train: {len(train_files)}, val: {len(val_files)}, test: {len(test_files)}")


with open('m3fd_train.txt', 'w') as f:
    f.write('\n'.join(train_files))
with open('m3fd_val.txt', 'w') as f:
    f.write('\n'.join(val_files))
with open('m3fd_test.txt', 'w') as f:
    f.write('\n'.join(test_files))

from torch.utils.data import DataLoader
from LLVIPdataset import M3FDDataset
import matplotlib.pyplot as plt
from torch.onnx.symbolic_opset9 import detach
from torchvision import transforms
import torchvision
import os
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torchvision.transforms.v2 import ToPILImage
import torchvision.utils as vutils
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.models as models


train_dataset = M3FDDataset(ir_dir, vis_dir, train_files,
                            crop_size=256, out_size=256, train=True)
val_dataset = M3FDDataset(ir_dir, vis_dir, val_files,
                          crop_size=256, out_size=256, train=False)
test_dataset = M3FDDataset(ir_dir, vis_dir, test_files,
                           crop_size=256, out_size=256, train=False)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device:{device}")

# ============ 基础模块 ============

import torch.nn.functional as F
import torchvision.models as models


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, padding_mode='reflect'),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, padding_mode='reflect'),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return F.leaky_relu(self.block(x) + x, 0.2)


class SlimGenerator(nn.Module):
    """1/32 基础通道，只下采样 2 次，瓶颈 1/4 分辨率"""
    def __init__(self, base=32):
        super().__init__()
        c = base
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, c, 7, 1, 3, padding_mode='reflect'),
            nn.InstanceNorm2d(c),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc1 = nn.Sequential(                      # 1/2
            nn.Conv2d(c, c * 2, 4, 2, 1),
            nn.InstanceNorm2d(c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(c * 2),
        )
        self.enc2 = nn.Sequential(                      # 1/4
            nn.Conv2d(c * 2, c * 4, 4, 2, 1),
            nn.InstanceNorm2d(c * 4),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(c * 4),
        )
        self.bottleneck = nn.Sequential(*[ResBlock(c * 4) for _ in range(4)])

        self.dec2 = nn.Sequential(                      # -> 1/2
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(c * 4, c * 2, 3, 1, 1, padding_mode='reflect'),
            nn.InstanceNorm2d(c * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(c * 2 + c * 2, c * 2, 3, 1, 1, padding_mode='reflect'),
            nn.InstanceNorm2d(c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(c * 2),
        )
        self.dec1 = nn.Sequential(                      # -> 1/1
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(c * 2, c, 3, 1, 1, padding_mode='reflect'),
            nn.InstanceNorm2d(c),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(c + c, c, 3, 1, 1, padding_mode='reflect'),
            nn.InstanceNorm2d(c),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(c),
        )
        self.out = nn.Sequential(
            nn.Conv2d(c, 3, 3, 1, 1, padding_mode='reflect'),
            nn.Tanh(),
        )

    def forward(self, x):
        e0 = self.init_conv(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        d2 = self.dec2(b);   d2 = self.fuse2(torch.cat([d2, e1], dim=1))
        d1 = self.dec1(d2);  d1 = self.fuse1(torch.cat([d1, e0], dim=1))
        return self.out(d1)


# ==================== 判别器 (标准 4 通道) ====================
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_ch=4, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.InstanceNorm2d(ndf * 2, affine=True), # ★ 换成 IN
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.InstanceNorm2d(ndf * 4, affine=True), # ★ 换成 IN
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 1)
        )
    def forward(self, x):
        return self.net(x)
# ============ 方案 B 的判别器输入 ============

def get_ir_detail(ir_img, kernel_size=15):
    pad = kernel_size // 2
    ir_low = F.avg_pool2d(ir_img, kernel_size=kernel_size, stride=1, padding=pad)
    return ir_img - ir_low


def get_ir_low(ir_img, kernel_size=15):
    pad = kernel_size // 2
    return F.avg_pool2d(ir_img, kernel_size=kernel_size, stride=1, padding=pad)


def build_d_input(ir_img, vis_img):
    ir_detail = get_ir_detail(ir_img)
    ir_low = get_ir_low(ir_img)
    return torch.cat([ir_detail, ir_low, vis_img], dim=1)   # 5 通道


# ============ 损失函数 ============

def make_low_freq(vis, kernel_size=15, sigma=5.0):
    coords = torch.arange(kernel_size, dtype=vis.dtype, device=vis.device) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = (g[:, None] * g[None, :]).expand(3, 1, kernel_size, kernel_size).contiguous()
    return F.conv2d(vis, kernel_2d, padding=kernel_size // 2, groups=3)


class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.vgg = nn.Sequential(*list(vgg[:16])).to(device)
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.criterion = nn.L1Loss()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, fake, real):
        fake = (fake + 1) / 2.0
        real = (real + 1) / 2.0
        fake = (fake - self.mean) / self.std
        real = (real - self.mean) / self.std

        return self.criterion(self.vgg(fake), self.vgg(real))

def color_stat_loss(fake, real):
    """★ 新增：颜色统计损失，解决颜色太杂"""
    fake_01 = (fake + 1) / 2.0
    real_01 = (real + 1) / 2.0
    loss = 0.0
    for c in range(3):
        loss += torch.abs(fake_01[:, c].mean() - real_01[:, c].mean())
        loss += torch.abs(fake_01[:, c].std() - real_01[:, c].std())
    return loss / 3.0


def color_distribution_loss(fake, real):
    """
    颜色分布损失：让生成图的 R/G/B 三通道全局分布与真实图一致。
    这能直接打破“红外亮度=伪彩色”的映射关系，强制天空变蓝、草地变绿。
    """
    fake_mean = fake.mean(dim=[2, 3])
    real_mean = real.mean(dim=[2, 3])
    fake_std = fake.std(dim=[2, 3])
    real_std = real.std(dim=[2, 3])

    return F.mse_loss(fake_mean, real_mean) + F.mse_loss(fake_std, real_std)

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.vgg_layers = torch.nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:16], # relu3_3
        ]).eval()
        for p in self.vgg_layers.parameters():
            p.requires_grad = False

        # 初始权重：浅层弱一点，relu3_3 主导
        self.layer_weights = torch.tensor([0.5, 1, 2])

    def forward(self, fake, real):
        # ★ 直接强制转换，不要判断 max
        fake = (fake + 1) / 2.0
        real = (real + 1) / 2.0

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(fake.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(fake.device)
        fake_n = (fake - mean) / std
        real_n = (real - mean) / std

        loss = 0.0
        f_f, f_r = fake_n, real_n
        for i, layer in enumerate(self.vgg_layers):
            f_f = layer(f_f)
            f_r = layer(f_r)
            w = self.layer_weights[i].to(fake.device)  # ★ 去掉除以方差
            loss += (w * F.l1_loss(f_f, f_r, reduction='none')).mean()
        return loss

class FFTLoss(nn.Module):
    """数值稳定的频域损失：幅值 + 复数域差异"""
    def forward(self, fake, real):
        fake = (fake + 1) / 2.0
        real = (real + 1) / 2.0
        fft_fake = torch.fft.rfft2(fake, norm='ortho')
        fft_real = torch.fft.rfft2(real, norm='ortho')

        # 1) 幅值损失（这个是稳定且最重要的）
        loss_amp = F.l1_loss(fft_fake.abs(), fft_real.abs())

        # 2) 复数域损失：用复数除法表达相位差，避免 atan2 的原点奇异和 ±π 卷绕
        #    fft_fake / fft_real 的角度 = 相位差，且连续无跳变
        ratio = fft_fake * torch.conj(fft_real)
        denom = fft_real.abs() + 1e-7          # 防除零
        phase_diff = ratio.real / denom        # ≈ cos(相位差)，值域 [-1,1]，处处可导
        loss_phase = F.l1_loss(phase_diff, torch.ones_like(phase_diff))

        return loss_amp + 0.5 * loss_phase


def color_distribution_loss(fake, real):
    """空间分布匹配：整图颜色分布对齐，不只是均值"""
    fake_01 = (fake + 1) / 2.0
    real_01 = (real + 1) / 2.0
    loss = 0.0
    for c in range(3):
        loss += F.mse_loss(fake_01[:, c].mean(dim=[1, 2]), real_01[:, c].mean(dim=[1, 2]))
        loss += F.mse_loss(fake_01[:, c].std(dim=[1, 2]),   real_01[:, c].std(dim=[1, 2]))
    return loss / 3.0

# 训练循环里加：

# 初始化（不要加载任何旧权重！）
generator = SlimGenerator().to(device)  # ← 换成 IN 版本
discriminator = PatchGANDiscriminator(in_ch=3, ndf=64).to(device)  # 只看生成图

optimizer_G = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

criterion_gan = nn.BCEWithLogitsLoss()   # D 输出不加 sigmoid，直接进这个 loss

criterion_l1 = nn.L1Loss()
#criterion_perceptual = PerceptualLoss(device) # 如果你有定义这个类
criterion_perceptual = VGGLoss().to(device)  # ← 换成新的
# ==================== 权重配置 ====================
lambda_l1 = 5.0
lambda_gan = 1
lambda_perceptual = 2.0
lambda_color = 1
criterion_fft = FFTLoss().to(device)
lambda_fft = 0.5


# 辅助头不要了，所以把 lambda_aux = 0
lambda_aux = 0.0

total_params_G = sum(p.numel() for p in generator.parameters())
total_params_D = sum(p.numel() for p in discriminator.parameters())
print(f'G 参数量: {total_params_G}')
print(f'D 参数量: {total_params_D}')
print(f'总计: {total_params_G + total_params_D}')

start_epoch=7
generator.load_state_dict(torch.load(f'm3fd1_generator256_epoch_{start_epoch}.pth', map_location=device))
discriminator.load_state_dict(torch.load(f'm3fd1_discriminator256_epoch_{start_epoch}.pth', map_location=device))
optimizer_G.load_state_dict(torch.load(f'm3fd1_optimizer_G256_epoch_{start_epoch}.pth',map_location=device))
optimizer_D.load_state_dict(torch.load(f'm3fd1_optimizer_D256_epoch_{start_epoch}.pth',map_location=device))
epochs=30
best_val_loss = float('inf')
patience = 10
epochs_no_improve = 0
best_val_lpips =float('inf')
from pytorch_msssim import ssim
import lpips
loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)

from pytorch_msssim import ssim
import torchvision.utils as vutils
import os

for epoch in range(start_epoch+1, epochs + 1):
    generator.train()
    discriminator.train()

    train_loss_G, train_loss_D = 0.0, 0.0
    train_loss_l1, train_loss_gan, train_loss_perc, train_loss_color = 0.0, 0.0, 0.0, 0.0

    for i, (ir_img, vis_img, filename) in enumerate(train_loader):
        ir_img = ir_img.to(device)
        vis_img = vis_img.to(device)

        # ==================== 1. 训练 D（只看 RGB）====================
        optimizer_D.zero_grad()
        with torch.no_grad():
            fake_vis = generator(ir_img)

        pred_real = discriminator(vis_img)
        pred_fake = discriminator(fake_vis)

        loss_D_real = criterion_gan(pred_real, torch.ones_like(pred_real))
        loss_D_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))
        loss_D = (loss_D_real + loss_D_fake) * 0.5

        loss_D.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=10)
        optimizer_D.step()

        # ==================== 2. 训练 G ====================
        optimizer_G.zero_grad()
        for p in discriminator.parameters():
            p.requires_grad = False

        fake_vis = generator(ir_img)
        pred_fake = discriminator(fake_vis)

        loss_G_gan = criterion_gan(pred_fake, torch.ones_like(pred_fake)) * lambda_gan
        loss_G_l1 = criterion_l1(fake_vis, vis_img) * lambda_l1
        loss_G_perc = criterion_perceptual(fake_vis, vis_img) * lambda_perceptual
        loss_G_color = color_distribution_loss(fake_vis, vis_img) * lambda_color  # 换成分布损失
        loss_G_fft = criterion_fft(fake_vis, vis_img) * lambda_fft

        loss_G = loss_G_l1 + loss_G_gan + loss_G_perc + loss_G_color + loss_G_fft

        loss_G.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=10)
        optimizer_G.step()

        for p in discriminator.parameters():
            p.requires_grad = True

        # 统计
        train_loss_D+=loss_D.item()
        train_loss_G += loss_G.item()
        train_loss_l1 += loss_G_l1.item()
        train_loss_gan += loss_G_gan.item()
        train_loss_perc += loss_G_perc.item()
        train_loss_color += loss_G_color.item()

    # ==================== 3. 打印 ====================
    n_train = len(train_loader)


    print(f"Epoch [{epoch}/{epochs}] | D_loss: {train_loss_D/n_train:.4f} | "
          f"G_loss: {train_loss_G / n_train:.4f} | G_L1: {train_loss_l1 / n_train:.4f} | "
          f"G_gan: {train_loss_gan / n_train:.4f} | G_perc: {train_loss_perc / n_train:.4f} | "
          f"G_color: {train_loss_color / n_train:.4f}")

    torch.save(generator.state_dict(), f'm3fd1_generator256_epoch_{epoch}.pth')
    torch.save(discriminator.state_dict(), f'm3fd1_discriminator256_epoch_{epoch}.pth')
    torch.save(optimizer_G.state_dict(), f'm3fd1_optimizer_G256_epoch_{epoch}.pth')
    torch.save(optimizer_D.state_dict(), f'm3fd1_optimizer_D256_epoch_{epoch}.pth')

    # ----- 验证 -----
    generator.eval()
    #generator.use_aux = False
    val_loss = 0.0
    val_ssim = 0.0  # ★ 加这个
    val_lpips = 0.0  # ★ 加这个
    saved = False
    import os

    os.makedirs('generated256_m3fd1_val', exist_ok=True)  # ★ 加上这行自动建文件夹

    with torch.no_grad():
        for ir_val, vis_val, filename in val_loader:
            ir_val = ir_val.to(device)
            vis_val = vis_val.to(device)
            fake_vis_val = generator(ir_val)
            loss_l1 = criterion_l1(fake_vis_val, vis_val)
            val_loss += loss_l1.item()
            # ★ 加 SSIM
            ssim_val = ssim(fake_vis_val, vis_val, data_range=2.0)
            val_ssim += ssim_val.item()
            lpips_val = loss_fn_lpips(fake_vis_val, vis_val).mean()
            val_lpips += lpips_val.item()
            for i in range(ir_val.size(0)):
                name = os.path.splitext(filename[i])[0]
                ir_rgb = ir_val[i].repeat(3, 1, 1)
                combined = torch.cat([ir_rgb, fake_vis_val[i], vis_val[i]], dim=2)
                vutils.save_image(combined * 0.5 + 0.5,
                                  f'generated256_m3fd1_val/{name}_compare.png')
                if not saved:
                    vutils.save_image(combined * 0.5 + 0.5,
                                      f'generated256_m3fd1_val/{epoch:03d}_{name}_compare.png')
                    saved = True


    avg_val_loss = val_loss / len(val_loader)
    avg_val_ssim = val_ssim / len(val_loader)  # ★ 加这个
    avg_val_lpips = val_lpips / len(val_loader)  # ★ 加这个
    print(f"Val L1: {avg_val_loss:.4f} | Val SSIM: {avg_val_ssim:.4f} | Val LPIPS: {avg_val_lpips:.4f}")
    print(f"Val loss: {avg_val_loss:.4f} | Val SSIM: {avg_val_ssim:.4f}")  # ★ 改打印
    print(f"生成图 max: {fake_vis_val.max():.3f}, 真实图 max: {vis_val.max():.3f}")
    print(f"生成图 min: {fake_vis_val.min():.3f}, 真实图 min: {vis_val.min():.3f}")

    if avg_val_lpips < best_val_lpips:  # LPIPS 越小越好
        best_val_lpips = avg_val_lpips
        torch.save(generator.state_dict(), 'm3fd1_slim_generator_best.pth')
        print(f"  -> 保存最佳模型 (LPIPS={avg_val_lpips:.4f})")

    generator.train()
    #generator.use_aux = True

from pytorch_msssim import ssim


def compute_psnr(fake, real):
    mse = torch.mean((fake - real) ** 2)
    if mse == 0:
        return float('inf')
    return (10 * torch.log10(1.0 / mse)).item()


best_path = 'm3fd1_slim_generator_best.pth'   # ← 改

if os.path.exists(best_path):
    generator.load_state_dict(torch.load(best_path, map_location=device))
    print(f"已加载最佳模型: {best_path}")
else:
    generator.load_state_dict(torch.load(f'm3fd1_generator256_epoch_{epochs}.pth', map_location=device))
    print(f"使用 epoch {epochs} 的模型")

generator.eval()
generator.use_aux = False
os.makedirs('fid_fakem3fd1', exist_ok=True)
os.makedirs('fid_realm3fd1', exist_ok=True)

test_loss = 0.0
psnr_total = 0.0
ssim_total = 0.0
count = 0

with torch.no_grad():
    for ir_test, vis_test, filename in test_loader:
        ir_test = ir_test.to(device)
        vis_test = vis_test.to(device)
        fake_vis_test= generator(ir_test)

        loss_l1 = criterion_l1(fake_vis_test, vis_test)
        test_loss += loss_l1.item()

        fake_01 = fake_vis_test * 0.5 + 0.5
        real_01 = vis_test * 0.5 + 0.5

        for i in range(fake_01.size(0)):
            psnr_total += compute_psnr(fake_01[i], real_01[i])
            ssim_total += ssim(fake_01[i:i + 1], real_01[i:i + 1], data_range=1.0).item()
            count += 1

        for i in range(ir_test.size(0)):
            name = os.path.splitext(filename[i])[0]
            vutils.save_image(fake_01[i], f'fid_fakem3fd1/{name}.png')
            vutils.save_image(real_01[i], f'fid_realm3fd1/{name}.png')

avg_test_loss = test_loss / len(test_loader)
avg_psnr = psnr_total / count
avg_ssim = ssim_total / count

print(f"\n{'=' * 50}")
print(f"测试集结果:")
print(f"  测试集 loss: {avg_test_loss:.4f}")
print(f"  PSNR: {avg_psnr:.2f} dB")
print(f"  SSIM: {avg_ssim:.4f}")
print(f"  共 {count} 张图")
print(f"{'=' * 50}")
print(f"\n生成图和真值图已保存到 fid_fakem3fd1/ 和 fid_realm3fd1/")
print(f"算 FID 命令:")
print(f"  python -m pytorch_fid fid_fakem3fd1 fid_realm3fd1 --device cuda:0 --dims 768/2048？？？")