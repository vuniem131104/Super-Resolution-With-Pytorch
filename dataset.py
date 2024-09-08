import json
import os
from PIL import Image
from torch.utils.data import Dataset
from utils import ImageTransform, create_list_images


class SRDataset(Dataset):
    def __init__(self, data_folder, split, crop_size, scaling_factor, lr_img_type, hr_img_type, test_data_name=None):
        super().__init__()

        self.data_folder = data_folder
        self.split = split.lower()
        self.crop_size = int(crop_size)
        self.scaling_factor = int(scaling_factor)
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type
        self.test_data_name = test_data_name

        assert self.split in {'train', 'val'}
        assert lr_img_type in {'[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm'}
        assert hr_img_type in {'[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm'}

        if self.split == 'train':
            assert self.crop_size % self.scaling_factor == 0, "Crop dimensions are not perfectly divisible by scaling factor! This will lead to a mismatch in the dimensions of the original HR patches and their super-resolved (SR) versions!"

        if self.split == 'train':
            self.images = create_list_images(self.data_folder, 'train_images.json')
        elif self.split == 'val':
            self.images = create_list_images(self.data_folder, 'val_images.json')

        self.transform = ImageTransform(split=self.split,
                                         crop_size=self.crop_size,
                                         scaling_factor=self.scaling_factor,
                                         lr_img_type=self.lr_img_type,
                                         hr_img_type=self.hr_img_type)

    def __getitem__(self, i):
        if not os.path.exists(self.images[i]):
            img = Image.open(self.images[0], mode='r')
        else:
            img = Image.open(self.images[i], mode='r')
        img = img.convert('RGB')
        if img.width <= 96 or img.height <= 96:
            print(self.images[i], img.width, img.height)
            img = img.resize((100, 100), Image.BICUBIC)
        lr_img, hr_img = self.transform(img)

        return lr_img, hr_img

    def __len__(self):
        return len(self.images)