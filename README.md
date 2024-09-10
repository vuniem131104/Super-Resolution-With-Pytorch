# Overview
This repo is the implementation of SRRESNET and SRGAN in super resolution task for blurry images 
## SRRESNET ARCHITECTURE
![image](https://github.com/user-attachments/assets/fead4ca9-4071-4ef5-a95b-da99f52366f2)
The SRResNet is composed of the following operations –
- First, the low resolution image is convolved with a large kernel size 9x9 and a stride of 1, producing a feature map at the same resolution but with 
64 channels. A parametric ReLU (PReLU) activation is applied.
- This feature map is passed through 16 residual blocks, each consisting of a convolution with a 3x3 kernel and a stride of 1, batch normalization and PReLU activation, another but similar convolution, and a second batch normalization. The resolution and number of channels are maintained in each convolutional layer.
- The result from the series of residual blocks is passed through a convolutional layer with a 3x3 kernel and a stride of 1, and batch normalized. The resolution and number of channels are maintained. In addition to the skip connections in each residual block (by definition), there is a larger skip connection arching across all residual blocks and this convolutional layer.
- 2 subpixel convolution blocks, each upscaling dimensions by a factor of 2 (followed by PReLU activation), produce a net 4x upscaling. The number of channels is maintained.
- Finally, a convolution with a large kernel size 9x9 and a stride of 1 is applied at this higher resolution, and the result is Tanh-activated to produce the super-resolved image with RGB channels in the range [-1, 1].

### The SRResNet Update

Training the SRResNet, like any network, is composed of a series of updates to its parameters. What might constitute such an update?

Our training data will consist of high-resolution (gold) images, and their low-resolution counterparts which we create by 4x-downsampling them using bicubic interpolation. 

In the forward pass, the SRResNet produces a **super-resolved image at 4x the dimensions of the low-resolution image** that was provided to it. 

![image](https://github.com/user-attachments/assets/20a2baa9-1532-4a23-9522-ea36ee35f685)


We use the **Mean-Squared Error (MSE) as the loss function** to compare the super-resolved image with this original, gold high-resolution image that was used to create the low-resolution image.

![image](https://github.com/user-attachments/assets/5ac9201c-b9bb-4621-ac4c-75183bc41ebb)

Choosing to minimize the MSE between the super-resolved and gold images means we will change the parameters of the SRResNet in a way that, if given the low-resolution image again, it will **create a super-resolved image that is closer in appearance to the original high-resolution version**. 

The MSE loss is a type of ***content* loss**, because it is based purely on the contents of the predicted and target images. 

In this specific case, we are considering their contents in the ***RGB space*** – we will discuss the significance of this soon.

## SRGAN ARCHITECTURE 
It consists of a *Generator* and a **Discriminator** as other conventional GANS

### GENERATOR 
It will be the same as the **SRRESNET**, and we will take the pretrained **SRRESNET** to initialize for the **GENERATOR** 

### DISCRIMINATOR

As you might expect, the Discriminator is a convolutional network that functions as a **binary image classifier**.

![image](https://github.com/user-attachments/assets/3ca77488-9508-40a8-a10d-a8a7d3d8769e)

It is composed of the following operations –

- The high-resolution image (of natural or artificial origin) is convolved with a large kernel size $9\times9$ and a stride of $1$, producing a feature map at the same resolution but with $64$ channels. A leaky *ReLU* activation is applied.
  
- This feature map is passed through $7$ **convolutional blocks**, each consisting of a convolution with a $3\times3$ kernel, batch normalization, and leaky *ReLU* activation. The number of channels is doubled in even-indexed blocks. Feature map dimensions are halved in odd-indexed blocks using a stride of $2$.
  
- The result from this series of convolutional blocks is flattened and linearly transformed into a vector of size $1024$, followed by leaky *ReLU* activation.
  
- A final linear transformation yields a single logit, which can be converted into a probability score using the *Sigmoid* activation function. This indicates the **probability of the original input being a natural (gold) image**.

### GENERATOR UPDATE 
- We will update the generator by using pretrained VGG19 model from torchvision. We no longer compare the orginial high resolution images with super resolution images but we compare their outputs after forward them through **the truncated VGGV19**. \
- Moreover, we utilize the advantage of an adversirial loss by using BCEWithLogitsLoss in pytorch to compare super resolution images passing through the discriminator and it **untrue label (1)**.
![image](https://github.com/user-attachments/assets/42df0edf-6c99-4439-b211-8d6f5aa74f52)

### DISCRIMINATOR UPDATE 
- It is very straightforward because it just distinguish between high resolution images with real labels (1) and super resolution images with it real labels (0).

## TRAINING MODELS
Our models are trained on COCO2024 dataset. If you want to train on your dataset, please do the following steps:

### Train SRRESNET
```bash
python3 train_srresnet.py --data_folder <your data directory> --batch_size <your batch size> --epochs <epochs to train models>
```

### Train SRGAN
```bash
python3 train_srgan.py --data_folder <your data directory> --batch_size <your batch size> --epochs <epochs to train models>
```

## TESTING MODELS
### Test SRRESNET with images
```bash
python3 test_srresnet.py --image_path <path to your image> --srresnet_ckpt <your srresnet checkpoint>
```

### Test SRGAN with images
```bash
python3 test_srgan.py --image_path <path to your image> --srgan_ckpt <your srgan checkpoint>
```

## RESULT
Loading...
