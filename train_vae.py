import torch
import torch.nn.functional as F
import torch.optim as optim
from diffusers import DiffusionPipeline
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from vae_model import VAE
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
import os
# from datasets import load_dataset
from utils import load_gene_expression, plot_losses, load_image


device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# 超参数
batch_size = 128
learning_rate = 1e-3
num_epochs = 200
image_size = 256
latent_dim = 4

result_path = '/data/sunpengyu/Task_STsim/Results/Vae_7Nov2024'
data_path = '/data/sunpengyu/Task_STsim/Data/Data_24Oct2024'
sd_path = '/data/sunpengyu/Task_STsim/Model/DiT_from_scratch-main/stable_diffusion_from_scratch'
dataroot_train = f'{data_path}/image/spots_train'
dataroot_test = f'{data_path}/image/spots_test'
# gene_expression_data = f'{data_path}/gene_expression/data_train.csv'
# gene_expression_val = f'{data_path}/gene_expression/data_test.csv'

train_dataloader = load_image(dataroot_train, image_size, batch_size, workers=2)
val_dataloader = load_image(dataroot_test, image_size, batch_size, workers=2)

# gene_expression_train = load_gene_expression(gene_expression_data)
# gene_expression_val = load_gene_expression(gene_expression_val)

# 初始化模型
vae = VAE(in_channels=3, latent_dim=latent_dim, image_size=image_size).to(device)

# 优化器和学习率调度器
optimizer = optim.AdamW(vae.parameters(), lr=learning_rate, weight_decay=1e-4)  # 可以考虑加入L2正则化：weight_decay=1e-4
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=5e-5)
# scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs) # 余弦退火学习率调度器
scheduler = OneCycleLR(optimizer, max_lr=1e-3, epochs=num_epochs, steps_per_epoch=len(train_dataloader))


# 自定义损失函数
"""
这个损失函数是用于变分自编码器（VAE）的训练。它由两部分组成：重构误差（MSE）和KL散度（KLD）。  
重构误差（MSE）：衡量重构图像 recon_x 和原始图像 x 之间的差异。使用均方误差（MSE）作为度量标准，计算两个图像之间的像素差异的平方和。  
KL散度（KLD）：衡量编码器输出的潜在分布 mu 和 logvar 与标准正态分布之间的差异。KL散度用于正则化潜在空间，使其接近标准正态分布。

:param recon_x: 重构图像
:param x: 原始图像
:param mu: 编码器输出的均值
:param logvar: 编码器输出的对数方差
:return: 总损失值 =（重构误差 + KL散度） <- 也可以调整加法的比重
"""

def vae_loss_function(recon_x, x, mu, logvar, kld_weight=0.1):
    batch_size = x.size(0)
    mse = F.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # 总损失 - 用于优化
    total_loss = mse + kld_weight * kld
    # 每像素指标 - 用于监控
    mse_per_pixel = mse / (batch_size * x.size(1) * x.size(2) * x.size(3))
    kld_per_pixel = kld / (batch_size * x.size(1) * x.size(2) * x.size(3))

    return total_loss, mse, kld_weight * kld, mse_per_pixel, kld_per_pixel

# 创建保存生成测试图像的目录
os.makedirs(f'{result_path}/generate_image', exist_ok=True)

# 训练循环
for epoch in range(num_epochs):
    vae.train()
    train_loss = 0
    mse_loss_total = 0
    kl_loss_total = 0
    mse_vs_kld = 0
    for batch_idx, batch in enumerate(train_dataloader):
        # print(batch[0].shape)
        data = batch[0].to(device)
        optimizer.zero_grad()

        recon_batch, _, mu, logvar = vae(data)  # 传递给VAE模型，获取重构图像、均值和对数方差
        loss, mse, kld, mse_per_pixel, kld_per_pixel = vae_loss_function(recon_batch, data, mu, logvar)  # 计算损失

        loss.backward()
        train_loss += loss.item()
        mse_vs_kld += mse_per_pixel / kld_per_pixel
        mse_loss_total += mse_per_pixel.item()
        kl_loss_total += kld_per_pixel.item()
        optimizer.step()
        scheduler.step()  # OneCycleLR 在每个批次后调用

    # scheduler.step()  # 除了 OneCycleLR 之外，其他调度器都需要在每个 epoch 结束时调用

    avg_train_loss = train_loss / len(train_dataloader.dataset)
    avg_mse_loss = mse_loss_total / len(train_dataloader.dataset)
    avg_kl_loss = kl_loss_total / len(train_dataloader.dataset)
    avg_mse_vs_kld = mse_vs_kld / len(train_dataloader)

    print(f'====> Epoch: {epoch} | Learning rate: {scheduler.get_last_lr()[0]:.6f}')
    print(f'Total loss: {avg_train_loss:.4f}')
    print(f'MSE loss (pixel): {avg_mse_loss:.6f} | KL loss (pixel): {avg_kl_loss:.6f}')

    # 验证集上的损失
    vae.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            data = batch[0].to(device)
            recon_batch, _, mu, logvar = vae(data)
            loss,_,_,_,_ = vae_loss_function(recon_batch, data, mu, logvar)
            val_loss += loss.item()

    val_loss /= len(val_dataloader.dataset)
    print(f'Validation set loss: {val_loss:.4f}')


    # 生成一些重构图像和可视化
    pipeline = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    if epoch % 20 == 0:
        with torch.no_grad():
            # 获取实际的批次大小
            actual_batch_size = data.size(0)
            # 重构图像
            n = min(actual_batch_size, 8)
            comparison = torch.cat([data[:n], recon_batch.view(actual_batch_size, 3, image_size, image_size)[:n]])
            comparison = (comparison * 0.5) + 0.5  # 将 [-1, 1] 转换回 [0, 1]
            save_image(comparison.cpu(), f'{result_path}/generate_image/reconstruction_{epoch}.png', nrow=n)

            # 需要安装 wandb 库，如果要记录训练过程可以打开下面的注释
            # wandb.log({"reconstruction": wandb.Image(f'vae_results/reconstruction_{epoch}.png')})

torch.save(vae.state_dict(), f'{result_path}/vae_model.pth')
print("Training completed.")
# 需要安装 wandb 库，如果要记录训练过程可以打开下面的注释
# wandb.finish()