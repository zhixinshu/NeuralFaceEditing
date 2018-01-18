#!/usr/bin/env python
import zx_7
import numpy as np
from skimage.transform import resize
import h5py
import os

def to_bc01(b01c):
    return np.transpose(b01c, (0, 3, 1, 2))


def to_b01c(bc01):
    return np.transpose(bc01, (0, 2, 3, 1))


def zx_imgs(dataset, size, crop):
    imgs, lights, names_idx, names = zx_7.ZX(dataset).arrays()
    #new_imgs = []
    #for img in imgs:
    #    img = img[crop:-crop, crop:-crop]
    #    img = resize(img, (size, size, 3), order=3)
    #    new_imgs.append(img)
    #imgs = np.array(new_imgs)
    imgs = np.array(imgs)
    lights = np.array(lights)
    return imgs, lights


def create_zx(dataset):
    imgs, lights = zx_imgs(dataset, size=64, crop=0)
    print imgs.shape
    print lights.shape
    # Shuffle images
    #idxs = np.random.permutation(np.arange(len(imgs)))
    #imgs = imgs[idxs]
    imgs = to_bc01(imgs)
    print imgs.shape
    print lights.shape
    return imgs, lights


if __name__ == '__main__':
    print("start create")
    dataset=os.getenv('DATASET')
    dfilename_imgs = 'zx_7_d10_inmc_' + dataset + '.hdf5'
    dfilename_lights = 'zx_7_d3_lrgb_' + dataset + '.hdf5'
    d_img, d_light = create_zx(dataset)
    print("end create")
    f1 = h5py.File(dfilename_imgs, 'w')
    f1.create_dataset('zx_7', data=d_img)
    f1.close()
    f2 = h5py.File(dfilename_lights, 'w')
    f2.create_dataset('zx_7', data=d_light)
    f2.close()
