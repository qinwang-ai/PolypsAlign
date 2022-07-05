from skimage.color import rgb2gray
from imageio import imread,imsave
import matplotlib.pyplot as plt


def main(wht_path, nbi_path):
    wht_img = imread(wht_path)
    nbi_img = imread(nbi_path)
    wht_gray = rgb2gray(wht_img)
    nbi_gray = rgb2gray(nbi_img)

    plt.figure()
    plt.imshow(wht_gray)
    plt.show()

    plt.figure()
    plt.imshow(nbi_gray)
    plt.show()

if __name__ == '__main__':
    main('','')
