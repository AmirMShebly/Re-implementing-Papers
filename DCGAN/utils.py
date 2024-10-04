import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def get_transforms(image_size, channels_img):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)]),
    ])

def show_images(fake_images, real_images, epoch):
    fake_images = fake_images.detach().cpu().numpy()
    real_images = real_images.detach().cpu().numpy()

    fig, ax = plt.subplots(2, 8, figsize=(15, 4))
    
    for i in range(8):
        ax[0, i].imshow(fake_images[i][0], cmap='gray', vmin=-1, vmax=1)
        ax[0, i].axis('off')
        ax[0, i].set_title("Fake")

        ax[1, i].imshow(real_images[i][0], cmap='gray', vmin=-1, vmax=1)
        ax[1, i].axis('off')
        ax[1, i].set_title("Real")

    plt.suptitle(f'Epoch {epoch}', fontsize=16)
    plt.show()
