import os
import time
import logging
import numpy as np
from PIL import Image
from skimage.transform import resize
from .util import download, checksum, archive_extract, checkpoint

log = logging.getLogger(__name__)

class ZX(object):
    def __init__(self, dataset='lfwsomething', data_root='/nfs/bigdisk/zhshu/FaceDatasets/Imgs/something/'):
        data_root = os.getenv('DATAROOT')
        dataset = os.getenv('DATASET')
        self.dataset = dataset
        # names of directories
        self.name = dataset
        self.data_dir = self.name
        # paths of directories
        self.data_dir_img = os.path.join(data_root, self.data_dir,'crop_resize')
        self.data_dir_mask = os.path.join(data_root, self.data_dir,'mask/crop_resize')
        self.data_dir_img_mo = os.path.join(data_root, self.data_dir,'crop_resize_maskout')
        self.data_dir_coord_mo = os.path.join(data_root, self.data_dir,'coord/crop_resize_maskout')
        self.data_dir_normal_mo = os.path.join(data_root, self.data_dir,'normal/crop_resize_maskout')
        self.data_dir_coord = os.path.join(data_root, self.data_dir,'coord/crop_resize')
        self.data_dir_normal = os.path.join(data_root, self.data_dir,'normal/crop_resize')
        self.data_dir_Lvecs = os.path.join(data_root, self.data_dir,'Lvecs')
        self._npz_path = os.path.join(data_root,self.data_dir, self.name+'.npz')
        self._install()
        self._load()

    def arrays(self):
        return self.imgs, self.lights, self.names_idx, self.names

    def _install(self):
        #load directories
        log.info('Read images and convert images to NumPy arrays')
        name_dict = {}
        imgs = []
        lights = []
        img_idx = 0
        for root, dirs, files in os.walk(self.data_dir_img): # parse files in the img_ directory
            for filename in files:
                pre, ext = os.path.splitext(filename)
                #print filename
                #if ext.lower() != '.jpg' | ext.lower() != '.png' | ext.lower() != '.jpeg':
                #    continue
                filepath_img    = os.path.join(self.data_dir_img , filename)
                filepath_coord  = os.path.join(self.data_dir_coord , filename)
                filepath_normal = os.path.join(self.data_dir_normal , filename)
                filepath_mask   = os.path.join(self.data_dir_mask, filename)
                filepath_img_mo    = os.path.join(self.data_dir_img_mo, filename)
                filepath_coord_mo  = os.path.join(self.data_dir_coord_mo, filename)
                filepath_normal_mo = os.path.join(self.data_dir_normal_mo, filename)
                filepath_Lvecs = os.path.join(self.data_dir_Lvecs, pre+'.csv')
                #img = np.array(resize(Image.open(filepath_img),(64,64,3),order=3))
                img = resize(np.array(Image.open(filepath_img)),(64,64,3),order=3)
                mask = resize(np.array(Image.open(filepath_mask)),(64,64),order=0)
                mask = mask.reshape(64,64,1)        
                normal = resize(np.array(Image.open(filepath_normal)),(64,64,3),order=3)
                coord = resize(np.array(Image.open(filepath_coord)),(64,64,3),order=3)
                light = np.genfromtxt(filepath_Lvecs, delimiter=',') 
                imgs.append(np.append(img,np.append(normal,np.append(mask,coord,2),2),2))  
                lights.append(light)
                name = filename
                if name not in name_dict:
                    name_dict[name] = []
                name_dict[name].append(img_idx)
                img_idx += 1
                if img_idx % 100 == 0:
                    print img_idx
        imgs = np.array(imgs)
        lights = np.array(lights)
        print imgs.shape
        print lights.shape
        names = sorted(name_dict.keys())
        names_idx = np.empty(len(imgs))
        for name_idx, name in enumerate(names):
            for img_idx in name_dict[name]:
                names_idx[img_idx] = name_idx
        with open(self._npz_path, 'wb') as f:
            np.savez(f, imgs=imgs, lights=lights, names_idx=names_idx, names=names)

    def _load(self):
        with open(self._npz_path, 'rb') as f:
            dic = np.load(f)
            self.imgs = dic['imgs']
            self.lights = dic['lights']
            self.names_idx = dic['names_idx']
            self.names = dic['names']
