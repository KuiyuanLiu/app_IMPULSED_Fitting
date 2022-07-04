# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 23:47:15 2020
@author: zzl
"""
import torch
import torchvision
from seg_GAN import net_G
from PIL import Image
import numpy as np
from skimage import io,transform
import os
import time
import sys

###Initil model
use_gpu = False  # bool
model = net_G(norm_layer = torch.nn.InstanceNorm2d) #BatchNorm2d/InstanceNorm2d

def load_network(network,save_path=''):        
    try:
        network.load_state_dict(torch.load(save_path))
    except:   
        pretrained_dict = torch.load(save_path)                
        model_dict = network.state_dict()
        try:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}                    
            network.load_state_dict(pretrained_dict)
        except:
            print('Pretrained network %s has fewer layers')
            for k, v in pretrained_dict.items():                      
                if v.size() == model_dict[k].size():
                    model_dict[k] = v
            if sys.version_info >= (3,0):
                not_initialized = set()
            else:
                from sets import Set
                not_initialized = Set()                    
            for k, v in model_dict.items():
                if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                    not_initialized.add(k.split('.')[0])                    
            print(sorted(not_initialized))
            network.load_state_dict(model_dict) 
load_network(model,save_path='cell_seg.pth')
model.eval()
if use_gpu:
    model.cuda()
print('########## Model Loaded ##########')

def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0      
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)

def tile_big_img(oimg_path,tile_size=2560):
    oimg = io.imread(oimg_path)
    h,w,c = oimg.shape
    h_max = int(h/tile_size)
    w_max = int(w/tile_size)
    tile_list = []
    for ii in range(h_max):
        for jj in range(w_max):
            tile_img = oimg[ii*tile_size:(ii+1)*tile_size,
                            jj*tile_size:(jj+1)*tile_size,:]
            tile_list.append(tile_img)
            #print('Tile {}/{},{}/{} done'.format(ii+1,h_max,jj+1,w_max))
    return tile_list,[h_max,w_max]    
    
    
def test_forward(img_list,seg_size):
    result = []
    data_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize(
                                                 (0.5, 0.5, 0.5),
                                                 (0.5, 0.5, 0.5))])
    for i in range(len(img_list)):
        Img = Image.fromarray(img_list[i]).convert('RGB')
        Img = Img.resize(seg_size)
        Img = data_transform(Img)    
        if use_gpu:
           Img = Img.cuda() 
        with torch.no_grad():
           output = model(Img.unsqueeze(0))   
        result_array = tensor2im(output[0,:,:,:]) 
        result.append(result_array)
        #print('Finished '+str(i+1)+'/'+str(len(img_list)))
    return result

def merge_tiles(result_list,shape,tile_size=2560):
    h_max,w_max = shape
    h = tile_size*h_max
    w = tile_size*w_max
    fresult = np.zeros([h,w,3],dtype=np.uint8)
    for iii in range(h_max):
        for jjj in range(w_max):
            result_img = result_list[iii*w_max+jjj]
            result_img = np.uint8(transform.resize(result_img, 
                                                   [tile_size,tile_size])*255)
            fresult[iii*tile_size:(iii+1)*tile_size,
                    jjj*tile_size:(jjj+1)*tile_size,:] = result_img
    return fresult

bimg_path = 'dataset5/original/'
pred_save_path = 'dataset5/cell_seg/'
bimg_list = os.listdir(bimg_path)
seg_size = [2560,2560]

for i in range(len(bimg_list)):
    s_time = time.time()
    tile_name = bimg_list[i]
    tiles = tile_big_img(bimg_path+tile_name)
    if len(tiles[0]) == 0:
        oimg = io.imread(bimg_path+tile_name)
        h,w,c = oimg.shape
        tiles = [[oimg],[1,1]]
    
    img_list,shape = tiles
    result_list = test_forward(img_list,seg_size)
    #fimg = merge_tiles(img_list,shape)
    fresult = merge_tiles(result_list,shape)
    osize = img_list[0].shape
    save_result = np.uint8(transform.resize(fresult,[osize[0],osize[1]])*255)
    io.imsave('{}/{}'.format(pred_save_path,tile_name),save_result)
    e_time = time.time()
    print('{}/{} done, taken {} S'.format(i+1,len(bimg_list),e_time-s_time))


