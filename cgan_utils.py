import torch
import torch.nn as nn


class AffineTransform(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        # 可学习的缩放和平移参数
        self.scale = nn.Parameter(torch.ones(1, output_dim))  # shape: [1, output_dim]
        self.shift = nn.Parameter(torch.zeros(1, output_dim)) # shape: [1, output_dim]

    def forward(self, x):
        return x * self.scale + self.shift  # 广播机制会自动扩展维度


class Generator(nn.Module):
    def __init__(self, latent_dim, condition_dim, output_dim):
        super(Generator, self).__init__()
        # 主网络部分保持不变
        self.main = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, output_dim),
            nn.Tanh()  # 保留Tanh激活
        )
        # 新增仿射变换层
        self.affine = AffineTransform(output_dim)

    def forward(self, noise, condition):
        combined = torch.cat([noise, condition], dim=1)
        raw_output = self.main(combined)       # [-1, 1]
        scaled_output = self.affine(raw_output) # 动态调整输出范围
        return scaled_output


# 判别器
class Discriminator(nn.Module):
    def __init__(self, input_dim, condition_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, data, condition):
        combined = torch.cat([data, condition], dim=1)
        return self.model(combined)


def generate_enhanced_data(generator, num_samples, latent_dim):
    enhanced_data = []
    enhanced_labels = []
    for _ in range(num_samples):
        with torch.no_grad():  # 关闭梯度计算
            noise = torch.randn((1, latent_dim))
            label = torch.randint(0, 2, (1, 1)).float()
            fake_data = generator(noise, label)
        enhanced_data.append(fake_data.detach())  # 分离计算图
        enhanced_labels.append(label.long())
    return torch.cat(enhanced_data, dim=0), torch.cat(enhanced_labels, dim=0).squeeze()


# ==================CGAN训练====================================
def train_cgan(generator, discriminator, real_data, labels, latent_dim, epochs, batch_size):
    # 在 train_cgan 函数中修改优化器部分
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))  # 降低 D 的学习率
    criterion = nn.BCELoss()
    d_losses = []  # 用于存储判别器的损失
    g_losses = []  # 用于存储生成器的损失
    for epoch in range(epochs):
        epoch_d_loss = 0
        epoch_g_loss = 0
        for i in range(0, len(real_data), batch_size):
            # =================== 真实数据 ===================
            real_batch = real_data[i:i + batch_size]  # Real batch shape: torch.Size([32, 12572])
            real_labels = labels[i:i + batch_size].unsqueeze(1).float()  # Real labels shape: torch.Size([32, 1])
            # =================== 训练判别器 ===================
            optimizer_D.zero_grad()
            # 真实数据损失
            real_validity = discriminator(real_batch, real_labels)
            d_real_loss = criterion(real_validity, torch.ones_like(real_validity))
            # 生成假数据
            noise = torch.randn((real_batch.size(0), latent_dim))
            fake_labels = torch.randint(0, 2, (real_batch.size(0), 1)).float()  # Fake labels shape: torch.size([32,1])
            fake_data = generator(noise, fake_labels)  # Fake data shape: torch.size([32,12572])
            # 假数据的损失
            fake_validity = discriminator(fake_data.detach(), fake_labels)
            d_fake_loss = criterion(fake_validity, torch.zeros_like(fake_validity))
            # 总损失和反向传播
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            # 梯度裁剪（应该在反向传播之后，优化器更新之前）
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizer_D.step()
            # =================== 训练生成器 ===================
            optimizer_G.zero_grad()
            # 重新通过判别器（不要detach）
            validity = discriminator(fake_data, fake_labels)
            g_loss = criterion(validity, torch.ones_like(validity))
            # 反向传播
            g_loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer_G.step()
            # =================== 记录损失 ===================
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
        epoch_d_loss /= (len(real_data) // batch_size)
        epoch_g_loss /= (len(real_data) // batch_size)
        d_losses.append(epoch_d_loss)
        g_losses.append(epoch_g_loss)

    return generator, discriminator
