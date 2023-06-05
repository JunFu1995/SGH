import torch.utils.data as data
from PIL import Image
import os
import random
import torch 
import torchvision
import scipy.io as scio 
import numpy as np 
import pandas as pd 

from six.moves import cPickle as pickle #for performance
import numpy as np
import numpy 
import torchvision.transforms.functional as TF
def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di



def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class CVIU(data.Dataset):
    # img size is not equal 
    def __init__(self, root: str, imgIndex: list, srIndex: list, transform=None, patch_num: int = 1, training: bool = True):
        img_all = []
        mos_all = []
        imgPath_all = []

        mat_file = os.path.join(root, 'sr_metric_data.mat')
        data = scio.loadmat(mat_file)

        img = data['SR_image_names']
        mos = data['subject_scores_mean40']
        srName = [] # 9
        picName = [] # 30
        for i in img:
            i = i.tolist()[0][0]
            i = i.strip().split('\\')
            img_all.append(i[-4:])
        for i in img_all[:30]:
            picName.append(i[-1])
        for i in img_all[::180]:
            srName.append(i[0]) 
        res = np.split(mos, 9, axis=1)
        res = np.concatenate(res, axis=0)
        mos = res[:,0].tolist()

        picName = [picName[i] for i in imgIndex]
        srName = [srName[i] for i in srIndex]


        scale_all = [int(i[1][-1]) for i in img_all]
        scale_f = []
        for i in range(len(img_all)):
            if img_all[i][0] in srName and img_all[i][-1] in picName:
                #print(img_all[i])
                imgPath = os.path.join(root, *img_all[i])
                if img_all[i][0] == 'Shan08' and img_all[i][1] == 'sf3':
                    continue
                for _ in range(patch_num):
                    imgPath_all.append(imgPath)
                    mos_all.append(mos[i])  
                    scale_f.append(scale_all[i])          

        #self.stride = stride
        self.transform = transform
        self.img_all = imgPath_all
        self.mos_all = mos_all
        self.scale_all = scale_f
        self.root = root 

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # imgPath = self.img_all[index] 
        # refsPath = os.path.join(self.root, 'Ref', imgPath.split('/')[-1].split('.')[0] + '.jpg')

        # return imgPath, refsPath
        imgPath, mos, z = self.img_all[index], self.mos_all[index], self.scale_all[index]
        img = pil_loader(imgPath)

        if self.transform is not None:
            img = self.transform(img)
        return imgPath, img, mos, z
    def __len__(self):
        return len(self.mos_all)

class QADS(data.Dataset):
    # 尺度信息
    def __init__(self, root: str, imgIndex: list, srIndex: list, transform=None, patch_num: int = 32, training: bool = True):
        img_all = []
        mos_all = []
        scale_all = []

        txt_file = os.path.join(root, 'mos_with_names.txt')
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                mos, imgName = line.strip().split()
                imgInd, s, srInd = imgName.split('.')[0].split('_')
                imgInd = int(imgInd[-2:])
                srInd = int(srInd)
                if imgInd in imgIndex and srInd in srIndex:
                    for _ in range(patch_num):
                        imgPath = os.path.join(root, 'super-resolved_images', imgName)
                        img_all.append(imgPath)
                        mos_all.append(float(mos))
                        scale_all.append(int(s))

        self.transform = transform
        self.img_all = img_all
        self.mos_all = mos_all
        self.scale_all = scale_all
        self.root = root 

        self.training = training 


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # imgPath = self.img_all[index] 
        # refsPath = os.path.join(self.root, 'source_images', imgPath.split('/')[-1].split('.')[0].split('_')[0] + '.bmp')
  
        # return imgPath, refsPath

        imgPath, mos, z = self.img_all[index], self.mos_all[index], self.scale_all[index]
        img = pil_loader(imgPath)

        if self.transform is not None:
            img = self.transform(img)
        return imgPath, img, mos, z
    def __len__(self):
        return len(self.mos_all)

class Waterloo(data.Dataset):
    # img size is not equal 505x505, 511x511
    def __init__(self, root: str, imgIndex: list, srIndex: list, transform=None, patch_num: int = 32, training: bool = True):
        img_all = []
        mos_all = []
        imgPath_all = []
        scale_all = []

        xlsx_file = os.path.join(root, 'rakningScores.xlsx')
        df = pd.read_excel(xlsx_file)

        data = df.iloc[1:27,3:].values.tolist()
        count = 0 
        ind = 1 
        while count < len(data):
            for img, mos in zip(data[count], data[count+1]):
                img_all.append(['factor2', str(int(ind)), int(img)]) # srf, imgid, srid
                mos_all.append(mos)
                scale_all.append(2)
            count += 2 
            ind += 1

        data = df.iloc[29:55,3:].values.tolist()
        count = 0 
        ind = 1 
        while count < len(data):
            for img, mos in zip(data[count], data[count+1]):
                img_all.append(['factor4', str(int(ind)), int(img)]) # srf, imgid, srid
                mos_all.append(mos)
                scale_all.append(4)
            count += 2 
            ind += 1

        data = df.iloc[57:83,3:].values.tolist()
        count = 0 
        ind = 1 
        while count < len(data):
            for img, mos in zip(data[count], data[count+1]):
                img_all.append(['factor8', str(int(ind)), int(img)]) # srf, imgid, srid
                mos_all.append(mos)
                scale_all.append(8)
            count += 2 
            ind += 1

        srName = list(range(1, 9)) # 8
        picName = list(range(1, 14)) # 13
        picName = [picName[i] for i in imgIndex]
        srName = [srName[i] for i in srIndex]

        mos_f = []
        scale_f = []
        for i in range(len(img_all)):
            if int(img_all[i][1]) in picName and img_all[i][-1] in srName:
                #print(img_all[i])
                imgPath = os.path.join(root, *img_all[i][:-1], '%d.bmp'%img_all[i][-1])
                for _ in range(patch_num):
                    imgPath_all.append(imgPath)
                    mos_f.append(mos_all[i])    
                    scale_f.append(scale_all[i])        

        #self.stride = stride
        self.transform = transform
        self.img_all = imgPath_all
        self.mos_all = mos_f
        self.scale_all = scale_f

        self.training = training
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        imgPath, mos, z = self.img_all[index], self.mos_all[index], self.scale_all[index]
        img = pil_loader(imgPath)

        if self.transform is not None:
            img = self.transform(img)
        return imgPath, img, mos, z
    def __len__(self):
        return len(self.mos_all)
