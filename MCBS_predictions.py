''' Import the libraries'''

import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import SimpleITK as sitk
import glob
import tensorflow as tf

from random import randrange
from numpy import save
from numpy import load
from numpy import savez_compressed

import torchvision.transforms.functional as F
import torch.nn as nn

import pathlib
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

''' Directory of the data '''

t1_dir  = "/home/T1.nii.gz"
t2_dir  = "/home/T2.nii.gz"
t2S_dir = "/home/T2S.nii.gz"

''' Define the functions '''

def read_image(path):
    img = sitk.ReadImage(path)
    img_as_numpy = sitk.GetArrayFromImage(img).astype('float')
    img_as_tensor = torch.as_tensor(img_as_numpy, dtype=torch.float32)
    return img_as_tensor

def resize_img(img, n):
    resize = transforms.Compose([transforms.Resize(n)])
    img = resize(img)
    return img

def zscore_normalize(img):
    mean = torch.mean(img)
    std = torch.std(img)
    normalized = (img - mean) / std
    return normalized

def extract_patches_2d(img,patch_shape,step=[1.0,1.0],batch_first=False):
    patch_H, patch_W = patch_shape[0], patch_shape[1]
    if(img.size(2)<patch_H):
        num_padded_H_Top = (patch_H - img.size(2))//2
        num_padded_H_Bottom = patch_H - img.size(2) - num_padded_H_Top
        padding_H = nn.ConstantPad2d((0,0,num_padded_H_Top,num_padded_H_Bottom),0)
        img = padding_H(img)
    if(img.size(3)<patch_W):
        num_padded_W_Left = (patch_W - img.size(3))//2
        num_padded_W_Right = patch_W - img.size(3) - num_padded_W_Left
        padding_W = nn.ConstantPad2d((num_padded_W_Left,num_padded_W_Right,0,0),0)
        img = padding_W(img)
    step_int = [0,0]
    step_int[0] = int(patch_H*step[0]) if(isinstance(step[0], float)) else step[0]
    step_int[1] = int(patch_W*step[1]) if(isinstance(step[1], float)) else step[1]
    patches_fold_H = img.unfold(2, patch_H, step_int[0])
    if((img.size(2) - patch_H) % step_int[0] != 0):
        patches_fold_H = torch.cat((patches_fold_H,img[:,:,-patch_H:,].permute(0,1,3,2).unsqueeze(2)),dim=2)
    patches_fold_HW = patches_fold_H.unfold(3, patch_W, step_int[1])   
    if((img.size(3) - patch_W) % step_int[1] != 0):
        patches_fold_HW = torch.cat((patches_fold_HW,patches_fold_H[:,:,:,-patch_W:,:].permute(0,1,2,4,3).unsqueeze(3)),dim=3)
    patches = patches_fold_HW.permute(2,3,0,1,4,5)
    patches = patches.reshape(-1,img.size(0),img.size(1),patch_H,patch_W)
    #patches = patches[:,0,:,:,:]
    if(batch_first):
        patches = patches.permute(1,0,2,3,4)
        patches = patches[0,:,:,:,:]
    #patches = patches[0,:,:,:,:]
    return patches

def reconstruct_from_patches_2d(patches,img_shape,step=[1.0,1.0],batch_first=False):
    patches = patches.unsqueeze(1)
    if(batch_first):
        patches = patches.permute(1,0,2,3,4)
    patch_H, patch_W = patches.size(3), patches.size(4)
    img_size = (patches.size(1), patches.size(2),max(img_shape[0], patch_H), max(img_shape[1], patch_W))
    step_int = [0,0]
    step_int[0] = int(patch_H*step[0]) if(isinstance(step[0], float)) else step[0]
    step_int[1] = int(patch_W*step[1]) if(isinstance(step[1], float)) else step[1]
    nrow, ncol = 1 + (img_size[-2] - patch_H)//step_int[0], 1 + (img_size[-1] - patch_W)//step_int[1]
    r_nrow = nrow + 1 if((img_size[2] - patch_H) % step_int[0] != 0) else nrow
    r_ncol = ncol + 1 if((img_size[3] - patch_W) % step_int[1] != 0) else ncol
    patches = patches.reshape(r_nrow,r_ncol,img_size[0],img_size[1],patch_H,patch_W)
    img = torch.zeros(img_size, device = patches.device)
    overlap_counter = torch.zeros(img_size, device = patches.device)
    for i in range(nrow):
        for j in range(ncol):
            img[:,:,i*step_int[0]:i*step_int[0]+patch_H,j*step_int[1]:j*step_int[1]+patch_W] += patches[i,j,]
            overlap_counter[:,:,i*step_int[0]:i*step_int[0]+patch_H,j*step_int[1]:j*step_int[1]+patch_W] += 1
    if((img_size[2] - patch_H) % step_int[0] != 0):
        for j in range(ncol):
            img[:,:,-patch_H:,j*step_int[1]:j*step_int[1]+patch_W] += patches[-1,j,]
            overlap_counter[:,:,-patch_H:,j*step_int[1]:j*step_int[1]+patch_W] += 1
    if((img_size[3] - patch_W) % step_int[1] != 0):
        for i in range(nrow):
            img[:,:,i*step_int[0]:i*step_int[0]+patch_H,-patch_W:] += patches[i,-1,]
            overlap_counter[:,:,i*step_int[0]:i*step_int[0]+patch_H,-patch_W:] += 1
    if((img_size[2] - patch_H) % step_int[0] != 0 and (img_size[3] - patch_W) % step_int[1] != 0):
        img[:,:,-patch_H:,-patch_W:] += patches[-1,-1,]
        overlap_counter[:,:,-patch_H:,-patch_W:] += 1
    img /= overlap_counter
    if(img_shape[0]<patch_H):
        num_padded_H_Top = (patch_H - img_shape[0])//2
        num_padded_H_Bottom = patch_H - img_shape[0] - num_padded_H_Top
        img = img[:,:,num_padded_H_Top:-num_padded_H_Bottom,]
    if(img_shape[1]<patch_W):
        num_padded_W_Left = (patch_W - img_shape[1])//2
        num_padded_W_Right = patch_W - img_shape[1] - num_padded_W_Left
        img = img[:,:,:,num_padded_W_Left:-num_padded_W_Right]
    return img

def pred_patches(upsample_patch):
    
    upsample = upsample_patch
    patch_pred = torch.zeros(0,1,256,256)
    for f in range(len(upsample)):
        
        one_patch = upsample[f,:,:,:]
        model.eval()
        with torch.no_grad():
            prediction = model([one_patch.to(device)])

        mask = prediction[0]['masks']
        mask = mask.cpu()
        threshold, upper, lower = 0.1, 1, 0
        bmask=np.where(mask>threshold, upper, lower)

        if len(mask) !=0:
            mm0 = bmask[0 ,:,:, :]
            for f in range(len(bmask)):
                m = bmask[f ,:,:, :]
                mm0 = mm0 + m 
                #binarize
                threshold, upper, lower = 0.1, 1, 0
                fuse=np.where(mm0>threshold, upper, lower)
                fuse = torch.from_numpy(fuse)
                fuse = fuse.unsqueeze(0)
        elif len(mask) == 0:
            fuse = torch.zeros(1,256,256)
            fuse = fuse.unsqueeze(0)

        patch_pred = torch.cat((patch_pred,fuse),0)
        
    return patch_pred 

def create_pred(img):
    # Create patches
    patches = extract_patches_2d(img, [64,64], batch_first=True)
    # Upsample
    m = nn.Upsample(scale_factor=4, mode='nearest')
    upsample = m(patches)
    # Prediction
    slice_pred = pred_patches(upsample)
    # Dowsample
    d = nn.Upsample(scale_factor=0.25, mode='nearest')
    downsample = d(slice_pred)
    # Reconstruct image
    vol = reconstruct_from_patches_2d(downsample, [512,512], batch_first=False)
    # Create tensor with anatomical data
    t2S = img[:,2,:,:].unsqueeze(0)

    pred = torch.cat((t2S, vol),1)
    return pred

def clean_pred (img, pred, max_value):
    s,x,y = img.shape
    max_tensor = tf.fill([s,x,y],255)
    max_numpy = max_tensor.numpy().astype('float')
    max_tensor = torch.as_tensor(max_numpy, dtype=torch.float32)
    img_modified = torch.where(pred>0, img, max_tensor)
    pred_modified = pred.detach().clone()
    for ss in range(s):
        for xx in range(x):
            for yy in range(y):
                if img_modified[ss,xx,yy] > max_value:
                    pred_modified[ss,xx,yy] = 0
    return pred_modified

def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    torchvision.models.detection.mask_rcnn.model_urls['maskrcnn_resnet50_fpn_coco'] = 'file:///home/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth'
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

import torch.nn as nn # Basic building blocks (containers, different layers ect.)

def double_conv(in_channels, out_channels): 
    return nn.Sequential( # a sequential container, elements are kept in order
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        # 3 is the size of the kernel? Padding is zero padding  
        nn.ReLU(inplace=True),
        # Activation function ReLU = applies the rectified linear unit function element wise 
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   

class UNet(nn.Module):
  # nn.Module is the base class for all neural network modules

    def __init__(self, n_class, n_channels):
      #Initialization, number of classes, number of channels
        super().__init__()
        # The U-shape         
        self.dconv_down1 = double_conv(n_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
        
        self.maxpool = nn.MaxPool2d(2) # a 2x2 kernel 
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out

''' Predictions of the data with MaskRCNN '''

tensor_mcbs = torch.zeros(0,2,512,512)

# Read the image
t1 = read_image(t1_dir)
t2 = read_image(t2_dir)
t2S = read_image(t2S_dir)
s,x,y = t1.size()

# Load the model and set a threshold
if s < 55:  # model for 35 slices or 2D multi slice acquisitions with anisotropic voxels
    model = get_instance_segmentation_model(2)
    model.load_state_dict(torch.load('./MCB_MaskRCNN_low.pt'))
    model = model.to(device)
    model_UNet = UNet(1,4)
    model_UNet.load_state_dict(torch.load('./MCB_3DUNet_low.pt'))
    model_UNet = model_UNet.to(device)
    thres_value = 1.3113
else:
    model = get_instance_segmentation_model(2)
    model.load_state_dict(torch.load('./MCB_MaskRCNN_high.pt'))
    model = model.to(device)
    model_UNet = UNet(1,4)
    model_UNet.load_state_dict(torch.load('./MCB_3DUNet_high.pt'))
    model_UNet = model_UNet.to(device)
    thres_value = 2.1118

# Resize the image
t1 = resize_img(t1,512).unsqueeze(1)
t2 = resize_img(t2,512).unsqueeze(1)
t2S = resize_img(t2S,512).unsqueeze(1)

# Z-score normalization
t1 = zscore_normalize(t1)
t2 = zscore_normalize(t2)
t2S = zscore_normalize(t2S)

# Concatenate the modalities of the image
input_tensor = torch.cat(( t1,t2,t2S),1)

# Create a prediction with Mask-RCNN
pred_tensor = torch.zeros(0,2,512,512)
for i in range(len(input_tensor)):
    input_ = input_tensor[i,:,:,:].unsqueeze(0)
    pred_ = create_pred(input_)
    pred_tensor = torch.cat((pred_tensor, pred_),0)

# Clean the predictions
t2S = pred_tensor[:,0,:,:]
pred = pred_tensor[:,1,:,:]

clean_pred_ = clean_pred(t2S.squeeze(), pred.squeeze(), thres_value)
clean_pred_tensor = torch.cat((t2S.unsqueeze(1), clean_pred_.unsqueeze(1)),1)

# Introduce an empty slice between each image
s,n,x,y = clean_pred_tensor.shape
tensor_mcbs = torch.cat((clean_pred_tensor, torch.zeros(1,n,x,y)),0)

''' Predictions of the data with UNet '''

tensor_UNet = torch.zeros(len(tensor_mcbs)-1,4,512,512)

for i in range(len(tensor_mcbs)):
    t2s = tensor_mcbs[i,0,:,:]
    pred = tensor_mcbs[i,1,:,:]
    
    if torch.sum(t2s)!=0:
        pre = tensor_mcbs[i-1,0,:,:]
        pos = tensor_mcbs[i+1,0,:,:]

        imgs_ = torch.cat((t2s.unsqueeze(0),
                              pre.unsqueeze(0),
                              pos.unsqueeze(0),
                              pred.unsqueeze(0)),0)
        imgs_ = imgs_.float().unsqueeze(0)
        
        tensor_UNet[i,:,:,:] = imgs_

train_dataloader_UNet = DataLoader(tensor_UNet, batch_size=1, num_workers=0, shuffle=False)

pred_tensor_UNet = torch.zeros(0,1,512,512)

model_UNet.eval()

for i, (data) in enumerate(train_dataloader_UNet):
    with torch.no_grad():
        data = data.to(device, dtype=torch.float)
        # Prediction
        prediction = model_UNet(data)
        prediction = prediction.cpu()
        
        pred_tensor_UNet = torch.cat((pred_tensor_UNet, prediction),0)

''' Threshold the predictions '''

# Clean the predictions
t2S = tensor_UNet[:,0,:,:]
pred = pred_tensor_UNet

clean_pred_tensor_UNet = clean_pred(t2S.squeeze(), pred.squeeze(), thres_value)

# Create a mask of the brain
t2S_image = sitk.ReadImage(t2S_dir)
t2S_mask = sitk.BinaryThreshold(t2S_image, lowerThreshold=-1, upperThreshold=0, insideValue=1, outsideValue=0)

dilate_xy = 10
dilate_z = int( dilate_xy / t2S_image.GetSpacing()[2] )  # divide by slice thickness
t2S_mask_dilated = sitk.BinaryDilate(t2S_mask, [dilate_xy, dilate_xy, dilate_z])

t2S_mask_as_numpy = sitk.GetArrayFromImage(t2S_mask_dilated).astype('int')
t2S_mask_as_tensor = torch.as_tensor(t2S_mask_as_numpy, dtype=torch.int32)
t2S_mask_as_tensor = resize_img(t2S_mask_as_tensor,512)

# Apply the mask
mask_pred_tensor = torch.where(t2S_mask_as_tensor==0., clean_pred_tensor_UNet, torch.zeros(t2S_mask_as_tensor.shape, dtype=torch.float32))

# Apply the threshold
final_pred_tensor = mask_pred_tensor > 0.001

''' Save predicted tensors '''

final_pred_numpy = final_pred_tensor.numpy().astype('int')
img = sitk.GetImageFromArray(final_pred_numpy)
sitk.WriteImage(img, '/home/predictions.nii.gz')
