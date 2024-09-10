from torch import nn, optim
from dataset import SRDataset
from utils import *
from models import *
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import StepLR
import torch.backends.cudnn as cudnn
import argparse

parser = argparse.ArgumentParser(description="Train SRGAN")
parser.add_argument("--data_folder", type=str, default='./data', help="Data folder")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--epochs", type=int, default=100, help="Epoch to train srresnet")
parser.add_argument("--ckpt", type=str, default='srresnet.pth', help="Checkpoint of pretrained srresnet")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
base_lr = 5e-4
wd = 1e-5
in_channels = 3
out_channels = 64
small_kernel_size = 3
large_kernel_size = 9
n_residual_blocks = 16
scaling_factor = 4
generator = Generator().to(device)
generator.init_with_checkpoint(args.ckpt)
gen_optimizer = optim.Adam(generator.parameters(), lr=base_lr, weight_decay=wd)
discriminator = Discriminator(in_channels=3, out_channels=64).to(device)
dis_optimizer = optim.Adam(discriminator.parameters(), lr=base_lr, weight_decay=wd)
vgg19_i = 5 
vgg19_j = 4
vgg19 = TruncatedVGG19(vgg19_i, vgg19_j).to(device)
vgg19.eval()
adversarial_loss = nn.BCEWithLogitsLoss().to(device)
criterion = nn.MSELoss().to(device)
epochs = args.epochs
step_train = 100
step_val = 500
batch_size = 64
train_losses = []
val_losses = []
train_dataset = SRDataset(args.data_folder, 'train', 96, 4, 'imagenet-norm', '[-1, 1]')
val_dataset = SRDataset(args.data_folder, 'val', 96, 4, 'imagenet-norm', '[-1, 1]')
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
step_lr = StepLR(gen_optimizer, step_size=10, gamma=0.1)
best_val_loss = 1e9
patience = 10
beta = 1e-3

cudnn.benchmark = True

def train(generator, discriminator, gen_optimizer, dis_optimizer, vgg19, epoch, epochs, device, train_loader):
    generator.train()
    discriminator.train()
    epoch_train_loss = 0
    for step, (lr_images, hr_images) in enumerate(tqdm(train_loader)):
        
        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)
        gen_optimizer.zero_grad()
        sr_images = generator(lr_images)
        sr_images = convert_image(sr_images, source='[-1, 1]', target='imagenet-norm')
        hr_vgg_images = vgg19(hr_images).detach()
        sr_vgg_images = vgg19(sr_images)
        vgg_loss = criterion(sr_vgg_images, hr_vgg_images)
        dis_sr_images = discriminator(sr_images)
        gen_adver_loss = adversarial_loss(dis_sr_images, torch.ones_like(dis_sr_images)) # we want to optimize Generator so fool the Discriminator with the label one
        total_train_loss = vgg_loss + beta * gen_adver_loss
        total_train_loss.backward()
        epoch_train_loss += total_train_loss.item()
        gen_optimizer.step()
        
        dis_optimizer.zero_grad()
        dis_sr_images = discriminator(sr_images.detach())
        dis_hr_images = discriminator(hr_images)
        dis_loss = adversarial_loss(dis_sr_images, torch.zeros_like(dis_sr_images)) + adversarial_loss(dis_hr_images, torch.ones_like(dis_hr_images))
        dis_loss.backward()
        dis_optimizer.step()
        
        if step % step_train == 0 and step != 0:
            print(f'Step {step}/{len(train_loader)} | Epoch {epoch + 1}/{epochs} | Train Loss: {total_train_loss.item():.4f}')
    epoch_train_loss /= len(train_loader)
    train_losses.append(epoch_train_loss)
    return epoch_train_loss

def val(generator, epoch, epochs, device, val_loader):
    generator.eval()
    epoch_val_loss = 0
    with torch.inference_mode():
        for step, (lr_images, hr_images) in enumerate(tqdm(val_loader)):
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)
            sr_images = generator(lr_images)
            val_loss = criterion(sr_images, hr_images)
            epoch_val_loss += val_loss.item()
            if step % step_val == 0 and step != 0:
                print(f'Step {step}/{len(val_loader)} | Epoch {epoch + 1}/{epochs} | Val Loss: {val_loss.item():.4f}')
    epoch_val_loss /= len(val_loader)
    val_losses.append(epoch_val_loss)
    return epoch_val_loss

def srgan_train(generator, discriminator, gen_optimizer, dis_optimizer, vgg19, epochs, device, train_loader, val_loader):
    global best_val_loss
    early_stopping_count = 0
    for epoch in range(epochs):

        if early_stopping_count >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

        epoch_train_loss = train(generator, discriminator, gen_optimizer, dis_optimizer, vgg19, epoch, epochs, device, train_loader)
        epoch_val_loss = val(generator, epoch, epochs, device, val_loader)

        if epoch_val_loss < best_val_loss:
            save_checkpoint(generator, 'best')
            best_val_loss = epoch_val_loss
            early_stopping_count = 0
            print(f'Early Stopping decrease to {early_stopping_count}/{patience}')
        else:
            early_stopping_count += 1
            print(f'Early Stopping increase to {early_stopping_count}/{patience}')

        step_lr.step()

        print(f'{epoch + 1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}')

        save_checkpoint(generator, 'last')

srgan_train(generator, discriminator, gen_optimizer, dis_optimizer, vgg19, epochs, device, train_loader, val_loader)