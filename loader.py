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
    def __init__(self, is_train, enable_aug=True):
        self.enable_aug = enable_aug
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

        # get labels
        self.white_label = list(map(lambda x: 'abnormal' not in x, self.white_list))
        self.white_label = np.array(self.white_label, dtype=np.int8)

        unique, counts = np.unique(self.white_label, return_counts=True)
        print('white:', dict(zip(unique, counts)))


    def __getitem__(self, white_index):
        wht_path = self.white_list[white_index]
        # wht_path = '/home/abc/datasets/PolyDataset2/white_light/abnormal/20210103136-4.JPG'
        postfix = wht_path.split('.')[-1]
        nbi_paths = glob.glob(wht_path.split('-')[0].replace('white_light', 'NBI')+'-*.'+postfix)
        if len(nbi_paths) == 0:
            raise Exception("%s NBI is empty" % wht_path)
        # read
        nbi_path = random.choice(nbi_paths)
        white_img = imageio.imread(wht_path)
        nbi_img = imageio.imread(nbi_path)
        wht_filename = wht_path.split('/')[-1]
        nbi_filename = nbi_path.split('/')[-1]

        # cropping
        wht_bbox = get_bbox_from_image(wht_filename, 'wht')
        nbi_bbox = get_bbox_from_image(nbi_filename, 'nbi')
        # print(white_img.shape, 'a', nbi_filename)
        white_img = white_img[wht_bbox[0]:wht_bbox[0]+wht_bbox[2], wht_bbox[1]: wht_bbox[1]+wht_bbox[3]]
        # print(white_img.shape, 'b', nbi_filename)
        nbi_img = nbi_img[nbi_bbox[0]:nbi_bbox[0]+nbi_bbox[2], nbi_bbox[1]: nbi_bbox[1]+nbi_bbox[3]]

        # augmentation
        if self.is_train and self.enable_aug:
            angle = random.randint(-180, 180)
            white_img = rotate(white_img, angle)
            nbi_img = rotate(nbi_img, angle)

            if random.random() > 0.5:
                nbi_img = np.flip(nbi_img, 0)
                white_img = np.flip(white_img, 0)

            if random.random() > 0.5:
                nbi_img = np.flip(nbi_img, 1)
                white_img = np.flip(white_img, 1)

        # convert to gray
        # white_img = rgb2gray(white_img)
        # white_img = exposure.equalize_hist(white_img)
        # white_img = np.expand_dims(white_img, axis=-1)
        # white_img = np.repeat(white_img, 3, -1)

        # resize
        white_img = resize(white_img, (448, 448))
        nbi_img = resize(nbi_img, (448, 448))

        # swap axis
        white_img = np.swapaxes(white_img, 0,-1)
        nbi_img = np.swapaxes(nbi_img, 0,-1)

        label = self.white_label[white_index]

        return white_img, nbi_img, label, wht_filename

    def __len__(self):
        return len(self.white_list)


if __name__ == '__main__':
    bs = 8
    dataset = PolyDataset(is_train=False)
    loader = DataLoader(dataset, batch_size=bs, num_workers=bs, shuffle=True)
    for item in loader:
        white_img, nbi_img, label = item
        print(white_img.shape)


