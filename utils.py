import os
import json
import torch
import torchvision.transforms.functional as FT
import random
import matplotlib.pyplot as plt
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
imagenet_std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
imagenet_mean_cuda = torch.FloatTensor([0.485, 0.456, 0.406]).to(device).unsqueeze(1).unsqueeze(2).unsqueeze(0)
imagenet_std_cuda = torch.FloatTensor([0.229, 0.224, 0.225]).to(device).unsqueeze(1).unsqueeze(2).unsqueeze(0)

def create_list_images(folder, path):

    with open(path, 'r') as f:
        data = json.load(f)

    list_images = [os.path.join(folder, item['file_name']) for item in data['images']]
    return list_images

def convert_image(image, source, target):
    assert source in {'pil', '[0, 1]', '[-1, 1]'}, "Cannot convert from source format %s!" % source
    assert target in {'pil', '[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm',
                      'y-channel'}, "Cannot convert to target format %s!" % target
    if source == 'pil':
        image = FT.to_tensor(image)
    elif source == '[0, 1]':
        pass
    elif source == '[-1, 1]':
        image = (image + 1.) / 2.

    if target == 'pil':
        image = FT.to_pil_image(image)
    elif target == '[0, 255]':
        image = image * 255.
    elif target == '[0, 1]':
        pass
    elif target == '[-1, 1]':
        image = image * 2. - 1
    elif target == 'imagenet-norm':
        if image.ndimension() == 3:
            image = (image - imagenet_mean) / imagenet_std
        elif image.ndimension() == 4:
            image = (image - imagenet_mean_cuda) / imagenet_std_cuda

    return image

class ImageTransform(object):
    def __init__(self, split, crop_size, scaling_factor, lr_img_type, hr_img_type):
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type
        self.split = split.lower()
        self.crop_size = crop_size
        self.scaling_factor = scaling_factor

    def __call__(self, image):
        # if self.split == 'train':
        left = random.randint(1, image.width - self.crop_size)
        top = random.randint(1, image.height - self.crop_size)
        right = left + self.crop_size
        bottom = top + self.crop_size
        hr_img = image.crop((left, top, right, bottom))
        # elif self.split == 'val':
        #     x_remainder = image.width % self.scaling_factor
        #     y_remainder = image.height % self.scaling_factor
        #     left = x_remainder // 2
        #     top = y_remainder // 2
        #     right = left + (image.width - x_remainder)
        #     bottom = top + (image.width - y_remainder)
        #     hr_img = image.crop((left, top, right, bottom))

        lr_img = hr_img.resize((hr_img.width // self.scaling_factor, hr_img.height // self.scaling_factor), Image.BICUBIC)

        lr_img = convert_image(lr_img, 'pil', 'imagenet-norm')
        hr_img = convert_image(hr_img, 'pil', '[-1, 1]')

        return lr_img, hr_img

def save_checkpoint(model, name):

    torch.save(model.state_dict(), f'{name}.pth')
    print(f'{name} checkpoint saved.')

def remove_module_prefix(state_dict):

    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")  # Remove prefix 'module.'
        new_state_dict[new_key] = value
    return new_state_dict

def show_result(lr_img, hr_img, sr_resnet_img, sr_gan_img=None):
    if sr_gan_img is None:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(lr_img)
        axs[0].set_title('Low Resolution Image')
        axs[0].axis('off')

        axs[1].imshow(hr_img)
        axs[1].set_title('High Resolution Image')
        axs[1].axis('off')

        axs[2].imshow(sr_resnet_img)
        axs[2].set_title('SR ResNet Image')
        axs[2].axis('off')
    else:
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs[0, 0].imshow(lr_img)
        axs[0, 0].set_title('Low Resolution Image')
        axs[0, 0].axis('off')

        axs[0, 1].imshow(hr_img)
        axs[0, 1].set_title('High Resolution Image')
        axs[0, 1].axis('off')

        axs[1, 0].imshow(sr_resnet_img)
        axs[1, 0].set_title('SR ResNet Image')
        axs[1, 0].axis('off')

        axs[1, 1].imshow(sr_gan_img)
        axs[1, 1].set_title('SR GAN Image')
        axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()




