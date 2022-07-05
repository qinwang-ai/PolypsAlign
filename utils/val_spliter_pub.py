import glob
import numpy as np
import random

if __name__ == '__main__':
    fold = 4
    white_list = glob.glob("/home/abc/datasets/PolyDataset/white_light/*/*.png")
    nbi_list = glob.glob("/home/abc/datasets/PolyDataset/NBI/*/*.png")
    white_list_prefix = list(map(lambda x:x.split("-")[0], white_list))
    nbi_list_prefix = list(map(lambda x:x.split("-")[0], nbi_list))

    white_list_prefix = np.unique(white_list_prefix)
    nbi_list_prefix = np.unique(nbi_list_prefix)

    random.shuffle(white_list_prefix)
    random.shuffle(nbi_list_prefix)


# ------------------------- white begin --------------------------
    train_wht_list_prefix = white_list_prefix[:int(len(white_list_prefix)*0.8)]
    val_wht_list_prefix = white_list_prefix[int(len(white_list_prefix)*0.8):]

    train_wht_list = []
    for prefix in train_wht_list_prefix:
        for item in white_list:
            if prefix in item:
                train_wht_list.append(item)

    val_wht_list = []
    for prefix in val_wht_list_prefix:
        for item in white_list:
            if prefix in item:
                val_wht_list.append(item)

    f = open('../splits/train_wht_pub%d.txt' % fold, 'w')
    f.write('\n'.join(train_wht_list))
    f = open('../splits/val_wht_pub%d.txt' % fold, 'w')
    f.write('\n'.join(val_wht_list))

# --------------------- begin nbi -----------------------

    # train_nbi_list_prefix = nbi_list_prefix[:int(len(nbi_list_prefix)*0.8)]
    # val_nbi_list_prefix = nbi_list_prefix[int(len(nbi_list_prefix)*0.8):]
    #
    # train_nbi_list = []
    # for prefix in train_nbi_list_prefix:
    #     for item in nbi_list:
    #         if prefix in item:
    #             train_nbi_list.append(item)
    #
    # val_nbi_list = []
    # for prefix in val_nbi_list_prefix:
    #     for item in nbi_list:
    #         if prefix in item:
    #             val_nbi_list.append(item)
    #
    # f = open('train_nbi.txt', 'w')
    # f.write('\n'.join(train_nbi_list))
    # f = open('val_nbi.txt', 'w')
    # f.write('\n'.join(val_nbi_list))

