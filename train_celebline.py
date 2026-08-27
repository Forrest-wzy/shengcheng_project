
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import os
import torch.optim as optim
from dataset_c import datasetself
def main():
    import torch
    #读取数据
    transforms_train=transforms.Compose([
        transforms.Resize((128,128)),
        #transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
    ])

    transforms_test=transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
    ])

    #train_dataset=datasetself(root_dir='D:\\PythonProject2\\celeba_data\\train',transform=transforms_train)
    train_dataset = datasetself(root_dir='D:\\PythonProject2\\celeba_data\\train\\clean_dataset', transform=transforms_train)
    test_dataset=datasetself(root_dir='D:\\PythonProject2\\celeba_data\\test',transform=transforms_test)

    train_loader=DataLoader(train_dataset,batch_size=8,shuffle=True,num_workers=0)
    test_loader=DataLoader(test_dataset,batch_size=8,shuffle=False,num_workers=0)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device:{device}")
    print(torch.cuda.get_device_name(0))
    #搭建网络
    #生成器G
    class Unetblock(nn.Module):
        def __init__(self,in_ch,out_ch,use_dropout=True):
            super().__init__()
            self.unetblock=nn.Sequential(
                nn.Conv2d(in_ch,out_ch,kernel_size=3,padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5)if use_dropout else nn.Identity()
            )
        def forward(self,x):
            return self.unetblock(x)

    class Down(nn.Module):
        def __init__(self,in_ch,out_ch,use_dropout=False):
            super().__init__()
            self.down=nn.Sequential(
                nn.MaxPool2d(2),
                Unetblock(in_ch,out_ch,use_dropout=use_dropout)
            )
        def forward(self,x):
            return self.down(x)

    class Up(nn.Module):
        def __init__(self,in_ch,out_ch,bilinear=True):
            super().__init__()
            if bilinear:
               self.up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
            else:
               self.up=nn.ConvTranspose2d(in_ch//2,in_ch//2,2,stride=2)

            self.conv=Unetblock(in_ch,out_ch,use_dropout=True)
        def forward(self,x1,x2):
            x1=self.up(x1)
            diffY=x2.size()[2]-x1.size()[2]
            diffX=x2.size()[3]-x1.size()[3]
            x1=nn.functional.pad(x1,[diffX//2,diffX-diffX//2,diffY,diffY-diffY//2])
            x=torch.cat([x2,x1],dim=1)
            return self.conv(x)

    class UNet(nn.Module):
        def __init__(self,in_channels=3,out_channels=3,features=[64,128,256,512]):
            super().__init__()
            self.inc=Unetblock(in_channels,features[0],use_dropout=False)
            self.down1=Down(features[0],features[1],use_dropout=False)
            self.down2=Down(features[1],features[2],use_dropout=False)
            self.down3=Down(features[2],features[3],use_dropout=False)
            self.down4=Down(features[3],features[3]*2,use_dropout=False)

            self.up1=Up(features[3]*2+features[3],features[3],bilinear=True)
            self.up2=Up(features[3]+features[2],features[2],bilinear=True)
            self.up3=Up(features[2]+features[1],features[1],bilinear=True)
            self.up4=Up(features[1]+features[0],features[0],bilinear=True)
            self.outc=nn.Conv2d(features[0],out_channels,1)
        def forward(self,x):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)

            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            x= self.outc(x)
            return torch.tanh(x)

    generator=UNet(in_channels=3,out_channels=3,features=[64,128,256,512]).to(device)
    #generator.load_state_dict(torch.load('pix2pix_generator1_epoch_80.pth'))

    class Patchblock(nn.Module):
        def __init__(self,in_ch,out_ch):
            super().__init__()
            self.patchblock=nn.Sequential(
                nn.Conv2d(in_ch,out_ch,kernel_size=4,stride=2,padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2,inplace=False)
            )
        def forward(self,x):
            return self.patchblock(x)

    class PatchGan(nn.Module):
        def __init__(self,in_ch=3,ndf=64,out_ch=1):
            super().__init__()
            self.inc=nn.Conv2d(in_ch*2,ndf,kernel_size=4,stride=2,padding=1)
            self.con1=Patchblock(ndf,ndf*2)
            self.con2=Patchblock(ndf*2,ndf*4)
            self.con3=nn.Conv2d(ndf*4,out_ch,kernel_size=4,stride=1,padding=1)
        def forward(self,x):
            x=self.inc(x)
            x=self.con1(x)
            x=self.con2(x)
            return self.con3(x)
    discriminator=PatchGan(in_ch=3,ndf=64,out_ch=1).to(device)
    #discriminator.load_state_dict(torch.load('pix2pix_discriminator1_epoch_40.pth'))
    #损失函数
    criterion_gan=nn.BCEWithLogitsLoss()
    criterion_L1=nn.L1Loss()# L1 损失，让生成图像更接近真实图像

    class VGGPerceptualLoss(nn.Module):
        def __init__(self, layer_ids=[2, 7, 12, 21, 30]):
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
                    loss += nn.functional.l1_loss(x, y)
            return loss

    vgg_loss = VGGPerceptualLoss().to(device)
    lambda_vgg = 20
    #优化器
    optimizer_G=optim.Adam(generator.parameters(),lr=0.00002,betas=(0.5,0.999))
    optimizer_D=optim.Adam(discriminator.parameters(),lr=0.00005,betas=(0.5,0.999))

    start_epoch = 60
    generator.load_state_dict(torch.load(f'pix2pix_generator1_epoch_{start_epoch}.pth', map_location=device))
    discriminator.load_state_dict(torch.load(f'pix2pix_discriminator1_epoch_{start_epoch}.pth', map_location=device))
    #print(f"成功加载 Epoch {start_epoch} 的 G 和 D 权重！")

    generator.eval()  # 切到评估模式
    test_line = next(iter(train_loader))[0][0:1].to(device)
    with torch.no_grad():
        test_fake = generator(test_line)
        print(f"生成图片范围: {test_fake.min():.2f} ~ {test_fake.max():.2f}")
    generator.train()  # 切回训练模式

    def lambda_rule(epoch):
        # 微调模式：前 10 轮保持 100% 的学习率（让模型快速吸收困难样本）
        # 10 轮之后，在接下来的 30 轮内线性衰减到 0（慢慢收敛，防止过拟合）
        n_epochs_decay = 30
        if epoch - start_epoch < 10:
            return 1.0
        else:
            lr_l = 1.0 - max(0, epoch - start_epoch - 10) / float(n_epochs_decay)
            return max(0.0, lr_l)
    #scheduler_G = StepLR(optimizer_G, step_size=80, gamma=0.2)
    #scheduler_D = StepLR(optimizer_D, step_size=80, gamma=0.2)
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda_rule)

    #训练
    import torch
    torch.cuda.empty_cache()
    epochs=100
    lambda_L1=100# L1 损失的权重
    lambda_gp = 10  # [修复9] 梯度惩罚权重，稳定判别器训练
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    for epoch in range(start_epoch+1,epochs+1):
        for i,(line,photo) in enumerate(train_loader):
            batch_size=line.size(0)
            line=line.to(device)
            photo=photo.to(device)
            #训练D
            optimizer_D.zero_grad()
            fake_photo=generator(line).detach()
            real_pairs=torch.cat([line,photo],dim=1)#真实配对
            real_pred=discriminator(real_pairs)
            loss_D_real=criterion_gan(real_pred,torch.ones_like(real_pred)*0.9)
            fake_pairs=torch.cat([line,fake_photo],dim=1)#假配对
            fake_pred=discriminator(fake_pairs)
            loss_D_fake=criterion_gan(fake_pred,torch.zeros_like(fake_pred))
            loss_D=(loss_D_real+loss_D_fake)*0.5

            alpha = torch.rand(batch_size, 1, 1, 1).to(device)
            interpolates = (alpha * real_pairs + (1 - alpha) * fake_pairs).detach()
            interpolates.requires_grad_(True)
            inter_pred = discriminator(interpolates)
            grad = torch.autograd.grad(
                outputs=inter_pred, inputs=interpolates,
                grad_outputs=torch.ones_like(inter_pred),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            grad = grad.view(grad.size(0), -1)
            grad_penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean()
            loss_D += lambda_gp * grad_penalty


            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizer_D.step()

            #训练G
            # ================= 训练 G =================
            optimizer_G.zero_grad()
            fake_photo = generator(line)

            # GAN Loss
            fake_pairs = torch.cat([line, fake_photo], dim=1)
            fake_pred = discriminator(fake_pairs)
            loss_G_Gan = criterion_gan(fake_pred, torch.ones_like(fake_pred))

            # L1 Loss
            loss_G_L1 = criterion_L1(fake_photo, photo)

            # VGG Loss 【关键修改】直接传原始的 [-1, 1] 数据进去，类内部会自动处理
            loss_G_VGG = vgg_loss(fake_photo, photo)

            # 总 Loss
            loss_G = loss_G_Gan + lambda_L1 * loss_G_L1 + lambda_vgg * loss_G_VGG

            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer_G.step()


        scheduler_G.step()
        scheduler_D.step()
        if epoch % 5==0:
            print(f"Epoch {epoch}: D_loss: {loss_D.item():.4f}, G_loss: {loss_G.item():.4f}")
            print(f"生成图片范围: {fake_photo.min().item():.4f} ~ {fake_photo.max().item():.4f}")
            torch.save(generator.state_dict(), f'pix2pix_generator1_epoch_{epoch}.pth')
            torch.save(discriminator.state_dict(), f'pix2pix_discriminator1_epoch_{epoch}.pth')

            generator.eval()  # 切换到评估模式
            os.makedirs('generated_p2p1', exist_ok=True)
            with torch.no_grad():
                # 从测试集拿一个 batch
                test_line, test_photo = next(iter(test_loader))
                test_line = test_line.to(device)

                # 生成假图片
                fake_photo = generator(test_line)
                fake_photo = fake_photo * 0.5 + 0.5  # 反归一化

                # 保存第一张图
                img = torchvision.transforms.ToPILImage()(fake_photo[1].cpu())
                img.save(f'generated_p2p1/epoch_{epoch}_fake.png')

            generator.train()  # 切回训练模式


        #测试
        generator.eval()
        os.makedirs('generated_p2p1', exist_ok=True)
        count=0
        with torch.no_grad():
            for line, photo in test_loader:
                line=line.to(device)
                fake_photo=generator(line)
                fake_photo=fake_photo*0.5+0.5
                for i in range(fake_photo.size(0)):
                    img=torchvision.transforms.ToPILImage()(fake_photo[i].cpu())
                    img.save( f'generated_p2p1/{count:05d}.png')
                    count+=1
        print(f"已生成 {count} 张假图片到 generated1/")

    #可视化
    def visualize_results(generator, test_loader, num_samples=5):
        generator.eval()

        # 获取一批数据
        line, photo = next(iter(test_loader))
        actual_samples = min(num_samples, line.size(0))

        # 截取并移动到设备
        line = line[:actual_samples].to(device)
        photo = photo[:actual_samples].to(device)

        # 生成假图片
        with torch.no_grad():
            fake_photo = generator(line)

        # --- 修改点：这里必须加 's'，变成 subplots ---
        # 注意：如果 actual_samples 为 1，返回的 axes 可能没有 [i, j] 维度，
        # 但通常 num_samples=5 没问题。为了安全，建议设置 squeeze=False。
        fig, axes = plt.subplots(actual_samples, 3, figsize=(9, 3 * actual_samples), squeeze=False)

        for i in range(actual_samples):
            # --- 修正点 1：处理输入线稿 (Input) ---
            input_img = line[i].cpu().permute(1, 2, 0) * 0.5 + 0.5
            # 如果是单通道 [H, W, 1]，必须 squeeze 掉最后一个维度变成 [H, W]
            # 或者使用 cmap='gray'
            if input_img.shape[2] == 1:
                axes[i, 0].imshow(input_img.squeeze(), cmap='gray')
            else:
                axes[i, 0].imshow(input_img)
            axes[i, 0].set_title('Input (Line)')
            axes[i, 0].axis('off')

            # --- 修正点 2：处理生成图 (Generated) ---
            fake_img = fake_photo[i].cpu().permute(1, 2, 0) * 0.5 + 0.5
            # 防止数值溢出导致颜色诡异，必须 clamp
            fake_img = torch.clamp(fake_img, 0, 1)
            axes[i, 1].imshow(fake_img)
            axes[i, 1].set_title('Generated')
            axes[i, 1].axis('off')

            # --- 修正点 3：处理真实图 (Real) ---
            real_img = photo[i].cpu().permute(1, 2, 0) * 0.5 + 0.5
            axes[i, 2].imshow(real_img)
            axes[i, 2].set_title('Real')
            axes[i, 2].axis('off')
        plt.tight_layout()
        plt.savefig('result.png')
        plt.show()

    visualize_results(generator, test_loader, num_samples=5)

if __name__ == '__main__':
    main()
