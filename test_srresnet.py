from models import SRResNet
from utils import *
import matplotlib
import argparse
matplotlib.use('Qt5Agg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default='test.png', help="path to image")
    parser.add_argument("--srresnet_ckpt", type=str, default='srresnet.pth', help="path to checkpoint")
    args = parser.parse_args()
    srresnet_ckpt = torch.load(args.srresnet_ckpt, map_location='cpu')
    srresnet = SRResNet()
    srresnet.load_state_dict(srresnet_ckpt)
    hr_img = Image.open(args.image_path, mode="r")
    hr_img = hr_img.convert('RGB')
    lr_img = hr_img.resize((int(hr_img.width / 4), int(hr_img.height / 4)), Image.BICUBIC)

    # Bicubic Upsampling
    bicubic_img = lr_img.resize((hr_img.width, hr_img.height), Image.BICUBIC)

    with torch.inference_mode():
        # Super-resolution (SR) with SRResNet
        sr_img_srresnet = srresnet(convert_image(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
        sr_img_srresnet = sr_img_srresnet.squeeze(0).cpu().detach()
        sr_img_srresnet = convert_image(sr_img_srresnet, source='[-1, 1]', target='pil')
    show_result(lr_img, hr_img, sr_img_srresnet)