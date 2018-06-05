import numpy as np
import struct
from torch.utils.data import Dataset
import random
import os.path as osp

class BatchLoader(Dataset):
    def __init__(self, dataRoot = './', phase = 'TRAIN'):
        if phase == 'TRAIN':
            fname_img = osp.join(dataRoot, 'train-images-idx3-ubyte')
            fname_lbl = osp.join(dataRoot, 'train-labels-idx1-ubyte')
        elif phase == 'TEST':
            fname_img = osp.join(dataRoot, 't10k-images-idx3-ubyte')
            fname_lbl = osp.join(dataRoot, 't10k-labels-idx1-ubyte')
        else:
            raise ValueError('phase should be TEST or TRAIN')

        with open(fname_lbl, 'rb') as flbl:
            magic, num = struct.unpack('>II', flbl.read(8) )
            lbl = np.fromfile(flbl, dtype = np.int8)

        with open(fname_img, 'rb') as fimg:
            magic, num, rows, cols = struct.unpack('>IIII', fimg.read(16) )
            img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

        lbl = lbl.astype(dtype = np.int32)
        img = img.astype(dtype = np.float32) / 255.0

        self.labels = lbl
        self.imgs = img
        self.num = num

        self.perm = list(range(num) )
        random.seed(0)
        random.shuffle(self.perm)


    def __len__(self):
        return len(self.perm)

    def __getitem__(self, ind):
        label = self.labels[ind]
        img = self.imgs[ind:ind+1, :, :]
        batchDict = {
                'label': label,
                'img': img
                }
        return batchDict




