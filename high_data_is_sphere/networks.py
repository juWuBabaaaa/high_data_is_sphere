import torch
from torch import nn


class Discriminator(nn.Module):

    def __init__(self, data_dimension):

        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(data_dimension, 100),
            nn.LeakyReLU(0.1),
            nn.Linear(100, 100),
            nn.LeakyReLU(0.1),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


class Generator(nn.Module):

    def __init__(self, noise_dim, data_dimension):

        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(noise_dim, 50),
            nn.LeakyReLU(0.1),
            nn.Linear(50, 50),
            nn.LeakyReLU(0.1),
            nn.Linear(50, data_dimension),
            nn.Sigmoid()
        )

    def forward(self, x):

        return self.main(x)


if __name__ == '__main__':

    d_noise = 2
    d_data = 2

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)

    d = Discriminator(d_data)
    g = Generator(d_noise, d_data)

    d.apply(weights_init)
    g.apply(weights_init)

    noise = torch.randn(2)

    fake_sample = g(noise)
    print("generate: ", fake_sample)
    print("discriminate: ", d(fake_sample))

