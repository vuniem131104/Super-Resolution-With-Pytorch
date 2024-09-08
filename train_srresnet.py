from torch import nn, optim
from dataset import SRDataset
from utils import *
from models import SRResNet
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import StepLR

device = 'cuda' if torch.cuda.is_available() else 'cpu'
base_lr = 5e-4
wd = 1e-5
in_channels = 3
out_channels = 64
small_kernel_size = 3
large_kernel_size = 9
n_residual_blocks = 16
scaling_factor = 4
model = SRResNet(in_channels, out_channels, small_kernel_size, large_kernel_size, n_residual_blocks, scaling_factor).to(device)
optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=wd)
criterion = nn.MSELoss().to(device)
epochs = 100
step_train = 100
step_val = 500
batch_size = 64
train_losses = []
val_losses = []
train_dataset = SRDataset('./data', 'train', 96, 4, 'imagenet-norm', '[-1, 1]')
val_dataset = SRDataset('./data', 'val', 96, 4, 'imagenet-norm', '[-1, 1]')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
step_lr = StepLR(optimizer, step_size=10, gamma=0.1)
best_val_loss = 1e9
patience = 10

def train(model, optimizer, epoch, epochs, device, train_loader):
    model.train()
    epoch_train_loss = 0
    for step, (lr_images, hr_images) in enumerate(tqdm(train_loader)):
        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)
        optimizer.zero_grad()
        sr_images = model(lr_images)
        train_loss = criterion(sr_images, hr_images)
        train_loss.backward()
        epoch_train_loss += train_loss.item()
        optimizer.step()
        if step % step_train == 0 and step != 0:
            print(f'Step {step}/{len(train_loader)} | Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss.item():.4f}')
    epoch_train_loss /= len(train_loader)
    train_losses.append(epoch_train_loss)
    return epoch_train_loss

def val(model, epoch, epochs, device, val_loader):
    model.eval()
    epoch_val_loss = 0
    with torch.inference_mode():
        for step, (lr_images, hr_images) in enumerate(tqdm(val_loader)):
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)
            sr_images = model(lr_images)
            val_loss = criterion(sr_images, hr_images)
            epoch_val_loss += val_loss.item()
            if step % step_val == 0 and step != 0:
                print(f'Step {step}/{len(val_loader)} | Epoch {epoch + 1}/{epochs} | Val Loss: {val_loss.item():.4f}')
    epoch_val_loss /= len(val_loader)
    val_losses.append(epoch_val_loss)
    return epoch_val_loss

def srresnet_train(model, optimizer, epochs, device, train_loader, val_loader):
    global best_val_loss
    early_stopping_count = 0
    for epoch in range(epochs):

        if early_stopping_count >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

        epoch_train_loss = train(model, optimizer, epoch, epochs, device, train_loader)
        epoch_val_loss = val(model, epoch, epochs, device, val_loader)

        if epoch_val_loss < best_val_loss:
            save_checkpoint(model, 'best')
            best_val_loss = epoch_val_loss
            early_stopping_count = 0
            print(f'Early Stopping decrease to {early_stopping_count}/{patience}')
        else:
            early_stopping_count += 1
            print(f'Early Stopping increase to {early_stopping_count}/{patience}')

        step_lr.step()

        print(f'{epoch + 1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}')

        save_checkpoint(model, 'last')

srresnet_train(model, optimizer, epochs, device, train_loader, val_loader)