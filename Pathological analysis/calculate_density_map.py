# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 15:55:41 2021
@author: lenovoM425
"""
import numpy as np
import os
from skimage import io
import matplotlib.pyplot as plt

seg_mask_path = 'dataset4/cell_seg/'
save_path = 'dataset4/cell_density/'
img_list = os.listdir(seg_mask_path)

def segmented_process(M, blk_size=(8,8)):
    fmask = np.zeros((int(M.shape[0]/blk_size[0]),int(M.shape[1]/blk_size[1])))
    for i in range(0, M.shape[0], blk_size[0]):
        for j in range(0, M.shape[1], blk_size[1]):            
            dvalue = M[i:i+blk_size[0], j:j+blk_size[1]].sum()/(blk_size[0]*blk_size[1])
            if dvalue>0:
                fmask[int(i/blk_size[0]),int(j/blk_size[1])] = dvalue                
    return fmask

for i in range(len(img_list)):
    img_name_ = img_list[i]
    mask = io.imread(seg_mask_path+img_list[i])
    mask1 = np.uint8(mask[:,:,0]>128)
    z = segmented_process(mask1,(2,2))    # 1378 = 2*13*53
    io.imsave(save_path+img_name_,z)
    plt.imshow(z,cmap='CMRmap')
    plt.colorbar()
    plt.savefig(save_path+img_name_[:-4]+'_dmap.png')
    plt.close()
    print('{}/{} done'.format(i+1,len(img_list)))

# import cv2
# heatmap = cv2.normalize(z, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_SUMMER)
# cv2.imshow("Heatmap", heatmap)
# cv2.imwrite('zz.png',heatmap)












