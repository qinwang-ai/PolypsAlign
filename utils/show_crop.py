import imageio
from utils.get_bbox import get_bbox_from_image
import cv2
import matplotlib.pyplot as plt
import glob
from skimage.transform import resize
import random
def main():
    nbi_paths = glob.glob('/home/abc/datasets/PolyDataset/NBI/*/*.png')
    # check
    for i in range(len(nbi_paths)):
        nbi_path = nbi_paths[i]
        nbi_filename = nbi_path.split('/')[-1]
        try:
            nbi_bbox = get_bbox_from_image(nbi_filename, 'nbi')
        except:
            print(nbi_filename)
    print('pass checking')


    random.shuffle(nbi_paths)
    for i in range(len(nbi_paths)):
        nbi_path = nbi_paths[i]
        nbi_filename = nbi_path.split('/')[-1]
        nbi_bbox = get_bbox_from_image(nbi_filename, 'nbi')
        nbi_img = imageio.imread(nbi_path)
        nbi_img = nbi_img[nbi_bbox[0]:nbi_bbox[0]+nbi_bbox[2], nbi_bbox[1]: nbi_bbox[1]+nbi_bbox[3]]
        nbi_img = resize(nbi_img, (448, 448))
        plt.figure()
        plt.title(nbi_filename)
        plt.imshow(nbi_img)
        plt.show()

if __name__ == '__main__':
    main()
