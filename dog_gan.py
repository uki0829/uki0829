import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
import os
from PIL import Image
import glob
from utils import download_dog_images
from torch.nn.utils import spectral_norm
import time  # 确保在文件顶部导入 time 模块

# 设置 matplotlib 后端（如果在没有图形界面的环境中运行）
import matplotlib
matplotlib.use('Agg')

# 设置参数
latent_dim = 128
img_size = 64
batch_size = 128  # 或更大
num_epochs = 200  # 增加训练轮数
lr = 0.0002
beta1 = 0.5

def load_and_preprocess_data(data_dir, image_size=(64, 64)):
    print(f"正在从 {data_dir} 加载图像...")
    jpg_files = glob.glob(os.path.join(data_dir, "*.jpg"))
    png_files = glob.glob(os.path.join(data_dir, "*.png"))
    image_files = jpg_files + png_files
    
    print(f"找到 {len(jpg_files)} 个 .jpg 文件")
    print(f"找到 {len(png_files)} 个 .png 文件")
    print(f"总共找到 {len(image_files)} 个图像文件")

    if not image_files:
        print("警告：没有找到任何图像文件。请检查数据目录路径是否正确。")
        print("目录内容:")
        for item in os.listdir(data_dir):
            print(item)
        return []

    images = []
    for file in image_files:
        try:
            img = Image.open(file).convert('RGB')
            img = img.resize(image_size)
            img_array = np.array(img) / 255.0  # 归一化到 [0, 1]
            images.append(img_array)
        except Exception as e:
            print(f"处理图像 {file} 时出: {str(e)}")

    print(f"成功加载 {len(images)} 张图像")
    return np.array(images)

def process_images(images):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),  # 新增：随机垂直翻转
        transforms.RandomRotation(10),    # 新增：随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 新增：随机平移
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    processed_images = [transform(img) for img in images]
    return torch.stack(processed_images)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 输入是 latent_dim
            nn.ConvTranspose2d(latent_dim, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            # 状态尺寸: (1024) x 4 x 4
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 状态尺寸: (512) x 8 x 8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 状态尺寸: (256) x 16 x 16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 状态尺寸: (128) x 32 x 32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 状态尺寸: (64) x 64 x 64
            nn.ConvTranspose2d(64, 3, 3, 1, 1, bias=False),
            nn.Tanh()
            # 最终状态尺寸: (3) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 输入 3 x 64 x 64
            spectral_norm(nn.Conv2d(3, 64, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态尺寸: (64) x 32 x 32
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态尺寸: (128) x 16 x 16
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态尺寸: (256) x 8 x 8
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态尺寸: (512) x 4 x 4
            spectral_norm(nn.Conv2d(512, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

def generate_and_save_images(generator, epoch, fixed_noise):
    with torch.no_grad():
        fake = generator(fixed_noise).detach().cpu()
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title(f"生成的图片: Epoch {epoch}")
    plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True), (1,2,0)))
    plt.savefig(f"generated_epoch_{epoch}.png")
    plt.close()
    print(f"已保存epoch {epoch}的生成图像")

def compute_gradient_penalty(discriminator, real_imgs, fake_imgs):
    device = real_imgs.device
    alpha = torch.rand(real_imgs.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_imgs + ((1 - alpha) * fake_imgs)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake = torch.ones(real_imgs.size(0), 1).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(d_interpolates.size()).to(device),  # 修改这里
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_gan(generator, discriminator, dataloader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    generator.to(device)
    discriminator.to(device)

    optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            batch_size = data[0].size(0)
            real_imgs = data[0].to(device)
            
            # 标签平滑
            real_label = torch.ones(batch_size, device=device) * 0.9  # 使用 0.9 而不是 1
            fake_label = torch.zeros(batch_size, device=device)
            
            # 训练判别器
            discriminator.zero_grad()
            
            # 真实图像
            output_real = discriminator(real_imgs).view(-1)
            errD_real = -torch.mean(output_real)  # WGAN 损失
            errD_real.backward()

            # 生成的图像
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_imgs = generator(noise)
            output_fake = discriminator(fake_imgs.detach()).view(-1)
            errD_fake = torch.mean(output_fake)  # WGAN 损失
            errD_fake.backward()

            # 梯度惩罚
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs, fake_imgs)
            gradient_penalty.backward()

            errD = errD_real + errD_fake + gradient_penalty * 10
            optimizerD.step()

            # 训练生成器
            if i % 5 == 0:  # 每 5 次迭代更新一次生成器
                generator.zero_grad()
                fake_imgs = generator(noise)
                output = discriminator(fake_imgs).view(-1)
                errG = -torch.mean(output)  # WGAN 损失
                errG.backward()
                optimizerG.step()

            if i % 50 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}')

        # 保存生成的图片
        generate_and_save_images(generator, epoch, fixed_noise)

        # 每10个epoch保存一次模型
        if epoch % 10 == 0:
            torch.save(generator.state_dict(), f'generator_epoch_{epoch}.pth')
            torch.save(discriminator.state_dict(), f'discriminator_epoch_{epoch}.pth')

def auto_train_gan(generator, discriminator, dataloader, num_epochs, max_images=1000, save_interval=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    generator.to(device)
    discriminator.to(device)

    optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

    total_generated_images = 0
    epoch = 0

    while total_generated_images < max_images:
        for i, data in enumerate(dataloader, 0):
            batch_size = data[0].size(0)
            real_imgs = data[0].to(device)
            
            # 标签平滑
            real_label = torch.ones(batch_size, device=device) * 0.9  # 使用 0.9 而不是 1
            fake_label = torch.zeros(batch_size, device=device)
            
            # 训练判别器
            discriminator.zero_grad()
            
            # 真实图像
            output_real = discriminator(real_imgs).view(-1)
            errD_real = -torch.mean(output_real)  # WGAN 损失
            errD_real.backward()

            # 生成的图像
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_imgs = generator(noise)
            output_fake = discriminator(fake_imgs.detach()).view(-1)
            errD_fake = torch.mean(output_fake)  # WGAN 损失
            errD_fake.backward()

            # 梯度惩罚
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs, fake_imgs)
            gradient_penalty.backward()

            errD = errD_real + errD_fake + gradient_penalty * 10
            optimizerD.step()

            # 训练生成器
            if i % 5 == 0:  # 每 5 次迭代更新一次生成器
                generator.zero_grad()
                fake_imgs = generator(noise)
                output = discriminator(fake_imgs).view(-1)
                errG = -torch.mean(output)  # WGAN 损失
                errG.backward()
                optimizerG.step()

            if i % 50 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}')

        # 保存生成的图片
        generate_and_save_images(generator, epoch, fixed_noise)
        total_generated_images += 64  # 假设每次生成64张图片

        print(f"已生成 {total_generated_images} 张图片")

        # 每save_interval个epoch保存一次模型
        if epoch % save_interval == 0:
            torch.save(generator.state_dict(), f'generator_epoch_{epoch}.pth')
            torch.save(discriminator.state_dict(), f'discriminator_epoch_{epoch}.pth')

        epoch += 1

        # 检查是否达到最大图片数量
        if total_generated_images >= max_images:
            print(f"已达到最大图片数量 {max_images}，重新开始训练")
            total_generated_images = 0
            epoch = 0
            # 可以选择重新初始化模型或使用当前模型继续训练

def main():
    dataset_path = "/Users/zhanghefeng/Desktop/python practice/dog_images"
    
    while True:
        # 检查并下载图像（如果文件夹为空）
        if not os.listdir(dataset_path):
            print("dog_images 文件夹为空，正在下载图像...")
            download_dog_images(1000, save_dir=dataset_path)  # 下载1000张图片
        
        # 加载和预处理数据
        images = load_and_preprocess_data(dataset_path)
        
        if len(images) == 0:
            print("错误：没有找到有效的图像。请检查数据目录中是否包含 .jpg 或 .png 文件。")
            return

        # 处理图像
        processed_images = process_images(images)

        # 创建 PyTorch 数据集和数据加载器
        dataset = TensorDataset(processed_images)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        # 创建网络实例
        generator = Generator()
        discriminator = Discriminator()

        # 开始训练
        auto_train_gan(generator, discriminator, dataloader, num_epochs, max_images=5000)

        print("完成一轮训练，等待5分钟后重新开始...")
        time.sleep(300)  # 等待5分钟

if __name__ == "__main__":
    main()