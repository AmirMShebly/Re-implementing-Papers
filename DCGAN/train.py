import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from models import Discriminator, Generator, initialize_weights
from utils import get_transforms, show_images

device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 2e-4
z_dim = 100
batch_size = 128
image_size = 64
num_epochs = 5
channels_img = 1
features_discriminator = 64
features_generator = 64

dataset = datasets.MNIST(root="dataset/", train=True, transform=get_transforms(image_size, channels_img), download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

generator = Generator(z_dim, channels_img, features_generator).to(device)
discriminator = Discriminator(channels_img, features_discriminator).to(device)
initialize_weights(discriminator)
initialize_weights(generator)
opt_disc = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
opt_gen = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
loss_fn = torch.nn.BCELoss()

step = 0
for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn((batch_size, z_dim, 1, 1)).to(device)
        fake = generator(noise)

        disc_real = discriminator(real).reshape(-1)
        loss_disc_real = loss_fn(disc_real, torch.ones_like(disc_real))
        disc_fake = discriminator(fake).reshape(-1)
        loss_disc_fake = loss_fn(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        discriminator.zero_grad()
        loss_disc.backward(retain_graph=True)
        opt_disc.step()

        output = discriminator(fake).reshape(-1)
        loss_gen = loss_fn(output, torch.ones_like(output))
        generator.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}")
            with torch.no_grad():
                fake = generator(torch.randn(32, z_dim, 1, 1).to(device))
                show_images(fake[:32], real[:32], epoch)
