import os
import numpy as np
import cv2
import albumentations as albu
import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base.modules import Activation


def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor),
    ]
    return albu.Compose(_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def Gau(h, w, s):
    gau = np.zeros([h, w])
    for i in range(h):
        for j in range(w):
            dx = j - (w - 1) * 0.5
            dy = i - (h - 1) * 0.5
            gau[i, j] = np.exp(- s * (dx * dx + dy * dy))
    return gau


def pred(h, w, best_model, gau, t,kernel):
    th = orimg.shape[0]
    tw = orimg.shape[1]
    oh = ((th - 1) // h + 1) * h
    ow = ((tw - 1) // w + 1) * w
    re = np.zeros([oh, ow])
    mm = np.zeros([oh, ow])
    extend_img = np.zeros((oh,ow,3))
    extend_img[:th, :tw] = orimg
    for i in range(100):
        y = i * (h // 2)
        for j in range(100):
            x = j * (w // 2)
            if y + h <= oh and x + w <= ow:
                crop = extend_img[y: y + h, x:x + w]
                sample = preprocess(image=crop)
                image = sample['image']
                x_tensor = torch.from_numpy(image).to('cuda').unsqueeze(0)

                pr_mask = best_model.predict(x_tensor)
                pr_mask = (pr_mask.squeeze().cpu().numpy().round())  # -70   70
                pr_mask = 1 / (1 + np.exp(-t * pr_mask))

                pr_mask = pr_mask * gau
                mm[y: y + h, x:x + w] += gau
                re[y: y + h, x:x + w] += pr_mask
    result = cv2.GaussianBlur(re / mm, kernel, 0)
    return result[:th, :tw]


os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
u1ro = 256  # road - red
u2ro = 320
gau1ro = Gau(u1ro, u1ro, 0.00012)  # @256  road
gau2ro = Gau(u2ro, u2ro, 0.00008)  # @320  road
u1bu = 128   # build - green
u2bu = 256
gau1bu = Gau(u1bu, u1bu, 0.0005)  # @128 build
gau2bu = Gau(u2bu, u2bu, 0.00012) # @256 build
u1wa = 256    # water - blue
u2wa = 320
gau1wa = Gau(u1wa, u1wa, 0.00012)  # @256 water
gau2wa = Gau(u2wa, u2wa, 0.00008)  # @320 water


model1ro = torch.load('../makeDataset/roadFull20_upp_vgg13.pth')  # 256*256
model2ro = torch.load('../makeDataset/roadFull30_320.pth')       # 320*320
model1bu = torch.load('../makeDataset/build40_upp_vgg13.pth')  # 128*128
model2bu = torch.load('../makeDataset/build256_E40.pth')  # 256*256
model1wa = torch.load('../makeDataset/waterE15_256.pth')  # 256*256
model2wa = torch.load('../makeDataset/waterFull20_upp_vgg13.pth')  # 320*320

model1ro.module.segmentation_head[2] = Activation(None)
model2ro.module.segmentation_head[2] = Activation(None)
if isinstance(model1ro, torch.nn.DataParallel):
    model1ro = model1ro.module
if isinstance(model2ro, torch.nn.DataParallel):
    model2ro = model2ro.module

model1bu.module.segmentation_head[2] = Activation(None)
model2bu.module.segmentation_head[2] = Activation(None)
if isinstance(model1bu, torch.nn.DataParallel):
    model1bu = model1bu.module
if isinstance(model2bu, torch.nn.DataParallel):
    model2bu = model2bu.module

model1wa.module.segmentation_head[2] = Activation(None)
model2wa.module.segmentation_head[2] = Activation(None)
if isinstance(model1wa, torch.nn.DataParallel):
    model1wa = model1wa.module
if isinstance(model2wa, torch.nn.DataParallel):
    model2wa = model2wa.module


# save_dir = '/data/baiduEles/caz/caz0'
save_dir = '/home/hua/pys/baiduEles/caz/caz4'
read_dir = '/data/baiduMaps/caz/caz4'
ENCODER = 'vgg13'  # ws:imagenet
ENCODER_WEIGHTS = 'imagenet'
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
preprocess = get_preprocessing(preprocessing_fn)

count=0
for filename in os.listdir(read_dir):
    img = cv2.imread(os.path.join(read_dir, filename))
    orimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    re1ro = pred(u1ro, u1ro, model1ro, gau1ro, 0.05, (9, 9))
    re2ro = pred(u2ro, u2ro, model2ro, gau2ro, 0.06, (9, 9))
    vred = np.amax((re1ro, re2ro), axis=0) * 255       # choose stronger

    re1bu = pred(u1bu, u1bu, model1bu, gau1bu, 0.06, (5, 5))
    re2bu = pred(u2bu, u2bu, model2bu, gau2bu, 0.06, (5, 5))
    vgreen = np.amax((re1bu, re2bu), axis=0) * 255     # choose stronger

    re1wa = pred(u1wa, u1wa, model1wa, gau1wa, 0.06, (9, 9))
    re2wa = pred(u2wa, u2wa, model2wa, gau2wa, 0.06, (9, 9))
    vblue = (re1wa + re2wa) * 128                    # average

    save_name = filename.replace('final_map', 'caz')
    colimg = cv2.merge((vblue, vgreen, vred))
    # colimg = cv2.resize(colimg, (640, 640), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(save_dir, save_name), colimg)
    print(filename, 'done')
    count+=1

print(count, 'files')
