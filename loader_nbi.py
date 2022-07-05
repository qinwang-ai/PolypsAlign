from torch.utils.data import DataLoader, RandomSampler, Dataset
import os
import glob
import numpy as np
import random
from scipy.ndimage import rotate
import imageio
from utils.get_bbox import get_bbox_from_image
from skimage.transform import resize

from skimage.color import rgb2gray
from skimage import exposure

class PolyDataset(Dataset):
    def __init__(self, is_train):
        split_id = 0
        if is_train:
            self.white_list = open('splits/train_wht_pub%d.txt' % split_id).readlines()
            self.white_list1 = open('splits/train_wht_prt%d.txt' % split_id).readlines()
        else:
            self.white_list = open('splits/val_wht_pub%d.txt' % split_id).readlines()
            self.white_list1 = open('splits/val_wht_prt%d.txt' % split_id).readlines()
        self.white_list.extend(self.white_list1)

        self.is_train = is_train

        # stripping
        self.white_list = list(map(lambda x: x.strip(), self.white_list))

        self.nbi_list = []
        for white_index in range(0, len(self.white_list)):
            wht_path = self.white_list[white_index]
            postfix = wht_path.split('.')[-1]
            nbi_paths = glob.glob(wht_path.split('-')[0].replace('white_light', 'NBI')+'-*.'+postfix)
            for p in nbi_paths:
                if p not in self.nbi_list:
                    self.nbi_list.append(p)

        # get labels
        self.nbi_label = list(map(lambda x: 'abnormal' not in x, self.nbi_list))
        self.nbi_label = np.array(self.nbi_label, dtype=np.int8)
        print(np.unique(self.nbi_label, return_counts=True))

    def __getitem__(self, nbi_index):
        nbi_path = self.nbi_list[nbi_index]
        # read
        _nbi_img = imageio.imread(nbi_path)

        # cropping
        nbi_filename = nbi_path.split('/')[-1]
        nbi_bbox = get_bbox_from_image(nbi_filename, 'nbi')
        # center_x = nbi_bbox[0] + nbi_bbox[2]//2
        # center_y = nbi_bbox[1] + nbi_bbox[3]//2
        # margin = 30
        # edge = (max(nbi_bbox[2], nbi_bbox[3]) // 2 + margin)
        # x = max(center_x - edge, 0)
        # y = max(center_y - edge, 0)
        # mx = min(center_x + edge, _nbi_img.shape[0])
        # my = min(center_y + edge, _nbi_img.shape[1])

        x, y, mx, my = nbi_bbox[0], nbi_bbox[1], nbi_bbox[0]+nbi_bbox[2],  nbi_bbox[1]+nbi_bbox[3]

        nbi_img = _nbi_img[x:mx, y:my, :]

        # augmentation
        if self.is_train:
            angle = random.randint(-180, 180)
            nbi_img = rotate(nbi_img, angle)
            if random.random() > 0.5:
                nbi_img = np.flip(nbi_img, 0)

            if random.random() > 0.5:
                nbi_img = np.flip(nbi_img, 1)

        # convert to gray
        # white_img = rgb2gray(white_img)
        # white_img = exposure.equalize_hist(white_img)
        # white_img = np.expand_dims(white_img, axis=-1)
        # white_img = np.repeat(white_img, 3, -1)

        # resize
        nbi_img = resize(nbi_img, (448, 448))

        # swap axis
        nbi_img = np.swapaxes(nbi_img, 0,-1)

        label = self.nbi_label[nbi_index]

        return nbi_img, label, nbi_filename

    def __len__(self):
        return len(self.nbi_list)


if __name__ == '__main__':
    bs = 8
    dataset = PolyDataset(is_train=False)
    loader = DataLoader(dataset, batch_size=bs, num_workers=bs, shuffle=True)
    for item in loader:
        nbi_img, label = item
        print(nbi_img.shape)


