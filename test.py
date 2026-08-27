import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import os
import torch.optim as optim
from dataset_c import datasetself

#读取数据
transforms_train=transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
])

transforms_test=transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
])

train_dataset=datasetself(root_dir='D:\\PythonProject2\\celeba_data\\train',transform=transforms_train)
test_dataset=datasetself(root_dir='D:\\PythonProject2\\celeba_data\\test',transform=transforms_test)

#train_loader=DataLoader(train_dataset,batch_size=8,shuffle=True,num_workers=0)
test_loader=DataLoader(test_dataset,batch_size=8,shuffle=False,num_workers=0)

# 在训练前取子集
train_dataset_subset = torch.utils.data.Subset(train_dataset, range(5000))
train_loader = DataLoader(train_dataset_subset, batch_size=8, shuffle=True,num_workers=0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device:{device}")
#搭建网络
#生成器G
class Unetblock(nn.Module):
    def __init__(self,in_ch,out_ch,use_dropout=True):
        super().__init__()
        self.unetblock=nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.5)if use_dropout else nn.Identity()
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
    def __init__(self,in_channels=3,out_channels=3,features=[32,64,128,256]):
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

generator=UNet(in_channels=3,out_channels=3,features=[32,64,128,256]).to(device)

#generator.load_state_dict(torch.load('pix2pix_generator.pth'))
print("成功加载生成器权重，准备继续训练！")

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
    def __init__(self,in_ch=3,ndf=32,out_ch=1):
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
discriminator=PatchGan(in_ch=3,ndf=32,out_ch=1).to(device)

#损失函数
criterion_gan=nn.BCEWithLogitsLoss()
criterion_L1=nn.L1Loss()# L1 损失，让生成图像更接近真实图像
#优化器
optimizer_G=optim.Adam(generator.parameters(),lr=0.0002,betas=(0.5,0.999))
optimizer_D=optim.Adam(discriminator.parameters(),lr=0.0002,betas=(0.5,0.999))

#from torch.optim.lr_scheduler import StepLR
lr_policy = 'linear'
def lambda_rule(epoch):
    # 前 50 轮保持 lr=0.0002，后 50 轮线性衰减到 0
    lr_l = 1.0 - max(0, epoch - 50) / float(50)
    return lr_l
#scheduler_G = StepLR(optimizer_G, step_size=80, gamma=0.2)
#scheduler_D = StepLR(optimizer_D, step_size=80, gamma=0.2)
scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda_rule)

test_line = next(iter(train_loader))[0][0:1].to(device)
with torch.no_grad():
    test_fake = generator(test_line)
    print(f"生成图片范围: {test_fake.min():.2f} ~ {test_fake.max():.2f}")


#训练
epochs=200
lambda_L1=100# L1 损失的权重
for epoch in range(epochs):


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

        loss_D.backward()
        optimizer_D.step()

        #训练G
        for _ in range(2):
            optimizer_G.zero_grad()
            fake_photo=generator(line)
            fake_pairs=torch.cat([line,fake_photo],dim=1)
            fake_pred=discriminator(fake_pairs)
            loss_G_Gan=criterion_gan(fake_pred,torch.ones_like(fake_pred))
            loss_G_L1=criterion_L1(fake_photo,photo)
            loss_G=loss_G_Gan+lambda_L1*loss_G_L1


            loss_G.backward()
            optimizer_G.step()

    scheduler_G.step()
    scheduler_D.step()

    if epoch % 10==0:
        print(f"Epoch {epoch}: D_loss: {loss_D.item():.4f}, G_loss: {loss_G.item():.4f}")
    #if epoch < 50:
     #   print(f"Epoch {epoch}: 预热阶段, G_loss: {loss_G.item():.4f}")
    #else:
     #   print(f"Epoch {epoch}: D_loss: {loss_D.item():.4f}, G_loss: {loss_G.item():.4f}")

    torch.save(generator.state_dict(), 'pix2pix_generator.pth')

#测试
generator.eval()
os.makedirs('generated_p2ptest', exist_ok=True)
count=0
with torch.no_grad():
    for line, photo in test_loader:
        line=line.to(device)
        fake_photo=generator(line)
        fake_photo=fake_photo*0.5+0.5
        for i in range(fake_photo.size(0)):
            img=torchvision.transforms.ToPILImage()(fake_photo[i].cpu())
            img.save( f'generated_p2p/{count:05d}.png')
            count+=1
print(f"已生成 {count} 张假图片到 generated/")

#可视化
def visualize_results(generator, test_loader, num_samples=5):
    generator.eval()
    line,photo=next(iter(test_loader))
    actual_samples = min(num_samples, line.size(0))
    line=line[:,actual_samples].to(device)
    photo=photo[::actual_samples].to(device)
    with torch.no_grad():
        fake_photo=generator(line)

    fig,axes=plt.subplot(actual_samples,3,figsize=(9,3*actual_samples))
    for i in range(actual_samples):
        # 线条画
        axes[i, 0].imshow(line[i].cpu().permute(1, 2, 0) * 0.5 + 0.5)
        axes[i, 0].set_title('输入 (线条画)')
        axes[i, 0].axis('off')
        # 生成图
        axes[i, 1].imshow(fake_photo[i].cpu().permute(1, 2, 0) * 0.5 + 0.5)
        axes[i, 1].set_title('生成')
        axes[i, 1].axis('off')
        # 真实图
        axes[i, 2].imshow(photo[i].cpu().permute(1, 2, 0) * 0.5 + 0.5)
        axes[i, 2].set_title('真实')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig('result.png')
    plt.show()
visualize_results(generator,test_loader,num_samples=5)

print(f"generated 文件夹中的图片数: {len(os.listdir('generated'))}")
