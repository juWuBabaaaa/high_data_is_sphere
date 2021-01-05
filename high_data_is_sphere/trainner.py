import networks
import sampler
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import os


d_noise = 2
d_data = 2
dn = f"data_{d_data}_z_{d_noise}"
fp = os.path.join(os.path.abspath(''), dn)

if not os.path.exists(dn):
    os.makedirs(fp)

manual_seed = 999
torch.manual_seed(manual_seed)
batch_size = 1000
fixed_size = 10000

train_epoch = 20000
lr_d = 0.00002
lr_g = 0.0002
beta1 = 0.5

generator = networks.Generator(d_noise, d_data)
discriminator = networks.Discriminator(d_data)

s = sampler.SAMPLER()

plt.ion()
fig = plt.figure(figsize=(6, 6))
fig.canvas.set_window_title("2D Generation")
fig.suptitle(f"dimension data:{d_data}, dimension noise:{d_noise}")
if d_data == 2:     # 感觉这个if 可以写在sampler类里面，没必要在这里拿出来
    sam = s.sampler(bs=batch_size)
    sam_show = s.sampler(bs=fixed_size)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

if d_data == 3:
    sam = s.sampler_3d(bs=batch_size)
    sam_show = s.sampler_3d(bs=fixed_size)
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, 0.999))
opt_g = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1, 0.999))
criterion = torch.nn.BCELoss()
target_one = torch.ones(batch_size)
target_zero = torch.zeros(batch_size)

fixed_noise = torch.randn((fixed_size, d_noise))


def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


generator.apply(weights_init)
discriminator.apply(weights_init)

G_losses = list()
D_real_losses = list()
D_fake_losses = list()
DX = list()
DGZ1 = list()
DGZ2 = list()


for epoch in range(train_epoch):

    discriminator.zero_grad()
    samples = Variable(torch.tensor(sam.__next__(), dtype=torch.float32))
    output = discriminator(samples).view(-1)
    loss_real = criterion(output, target_one)
    loss_real.backward()
    D_x = output.mean().item()

    noise = torch.randn((batch_size, d_noise))
    fake = generator(noise)
    output = discriminator(fake.detach()).view(-1)
    loss_fake = criterion(output, target_zero)
    loss_fake.backward()
    D_G_z1 = output.mean().item()

    loss = loss_fake + loss_real
    opt_d.step()

    generator.zero_grad()
    output = discriminator(fake).view(-1)
    loss_g_fake = criterion(output, target_one)
    loss_g_fake.backward()
    D_G_z2 = output.mean().item()
    opt_g.step()

    D_fake_losses.append(loss_fake.item())
    D_real_losses.append(loss_real.item())
    G_losses.append(loss_g_fake.item())
    DX.append(D_x)
    DGZ1.append(D_G_z1)
    DGZ2.append(D_G_z2)

    if epoch % 50 == 0 and epoch > 0:

        print(f"[{epoch}|{train_epoch}]\tLoss | D_real {loss_real} D_fake {loss_fake} G_fake {loss_g_fake}")
        ax3.clear()
        ax3.plot(list(range(epoch+1)), D_real_losses, 'r', label='d_real_loss')
        ax3.plot(list(range(epoch+1)), D_fake_losses, 'b', label='d_fake_loss')
        ax3.plot(list(range(epoch+1)), G_losses, 'g', label='g_loss')
        ax3.legend(loc='upper right')
        ax3.set_xlabel('iteration')
        ax4.clear()
        ax4.plot(list(range(epoch + 1)), DX, 'r', label='D(x)')
        ax4.plot(list(range(epoch + 1)), DGZ1, 'b', label='D(G(z1))')
        ax4.plot(list(range(epoch + 1)), DGZ2, 'g', label='D(G(z2)')
        ax4.legend(loc='upper right')
        ax4.set_xlabel('iteration')

        real = sam_show.__next__()
        fake = generator(fixed_noise).detach().numpy()

        if d_data == 2:
            ax1.clear()
            ax1.plot(real[:, 0], real[:, 1], 'k,', label='real')
            ax1.set_xticks([0, 0.5, 1])
            ax1.set_yticks([0, 0.5, 1])
            ax1.legend()
            ax2.clear()
            ax2.plot(fake[:, 0], fake[:, 1], 'r,', label='fake')
            ax2.set_xticks([0, 0.5, 1])
            ax2.set_yticks([0, 0.5, 1])
            ax2.legend()

        if d_data == 3:
            ax1.clear()
            ax1.plot(real[:, 0], real[:, 1], real[:, 2], '.', label='real')
            ax1.set_xticks([0, 0.5, 1])
            ax1.set_yticks([0, 0.5, 1])
            ax1.set_zticks([-0.5, 0, 0.5])
            ax1.legend()
            ax2.clear()
            ax2.plot(fake[:, 0], fake[:, 1], fake[:, 2], 'k,', label='fake')
            ax2.plot(real[:, 0], real[:, 1], real[:, 2], 'r,', alpha=0.1, label='real')
            ax2.set_xticks([0, 0.5, 1])
            ax2.set_yticks([0, 0.5, 1])
            ax2.set_zticks([-1, 0, 1])
            ax2.set_zlim(-0.001, 0.001)
            ax2.view_init(elev=25, azim=45)
            ax2.legend(loc='upper right')
        fig.savefig(os.path.join(fp, f"{epoch}.png"))
        plt.pause(0.02)
plt.ioff()
plt.show()
