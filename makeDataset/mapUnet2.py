import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
import torch
import segmentation_models_pytorch as smp
import time

def visualize(**images):
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


class Dataset(BaseDataset):
    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        masks = [(mask == 1) ]  #  class
        mask = np.stack(masks, axis=-1).astype('float')

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
#DATA_DIR = '/home/hua/pys/segmentation_models.pytorch/data/tri120den/'  #128
DATA_DIR = '/home/hua/pys/segmentation_models.pytorch/data/tri120den256/'   #256

x_train_dir = os.path.join(DATA_DIR, 'trainX')
y_train_dir = os.path.join(DATA_DIR, 'trainY')

x_valid_dir = os.path.join(DATA_DIR, 'validX')
y_valid_dir = os.path.join(DATA_DIR, 'validY')

x_test_dir = os.path.join(DATA_DIR, 'testX')
y_test_dir = os.path.join(DATA_DIR, 'testY')

# dataset = Dataset(x_train_dir, y_train_dir)   #classes=['car']
# image, mask = dataset[97] # get some sample ,349
# visualize( image=image,cars_mask=mask.squeeze(),)


ENCODER = 'vgg13'   #ws:imagenet
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda'

# model = smp.Unet(encoder_name=ENCODER,encoder_weights=ENCODER_WEIGHTS,activation=ACTIVATION,)
model = smp.UnetPlusPlus (encoder_name=ENCODER,encoder_weights=ENCODER_WEIGHTS,activation=ACTIVATION,)
model = torch.nn.DataParallel(model)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

train_dataset = Dataset(x_train_dir,y_train_dir,preprocessing=get_preprocessing(preprocessing_fn),)
valid_dataset = Dataset(x_valid_dir,y_valid_dir,preprocessing=get_preprocessing(preprocessing_fn),)

# train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=4)
# valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)
train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=8)
valid_loader = DataLoader(valid_dataset, batch_size=12, shuffle=False, num_workers=4)

loss = smp.utils.losses.DiceLoss()
metrics = [smp.utils.metrics.IoU(threshold=0.5)]

'''
optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001)])
train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

max_score = 0
start_time= time.time()
for i in range(0, 40):
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, 'build256_E40.pth')
        print('Model saved!')
        
    if i >=20:
        nlr = 0.0001 - (i-19)*0.0000045
        optimizer.param_groups[0]['lr'] = nlr
        print('decoder learning rate ', nlr)

print((time.time()-start_time),'s')

'''
# best_model = torch.load('../E40_unet_vgg13.pth')   # created by ym
# best_model = torch.load('./upp_E40_vgg13.pth')   # created by server
best_model = torch.load('../makeDataset/build256_E40.pth')   # created by server

test_dataset = Dataset(x_test_dir, y_test_dir,preprocessing=get_preprocessing(preprocessing_fn))
test_dataloader = DataLoader(test_dataset)
# evaluate model on test set
test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)
logs = test_epoch.run(test_dataloader)

# test dataset without transformations for image visualization
test_dataset_vis = Dataset( x_test_dir, y_test_dir)
for i in range(10):
    n = np.random.choice(len(test_dataset))
    image_vis = test_dataset_vis[n][0].astype('uint8')
    image, gt_mask = test_dataset[n]
    gt_mask = gt_mask.squeeze()
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    if isinstance(best_model,torch.nn.DataParallel):
        best_model=best_model.module

    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    visualize(
        image=image_vis,
        ground_truth_mask=gt_mask,
        predicted_mask=pr_mask
    )
