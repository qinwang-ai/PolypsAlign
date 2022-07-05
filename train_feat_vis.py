import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from loader import PolyDataset
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from models.resnet50_vis import resnet50
from models.discriminator import Discriminator
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib.pyplot as plt

def try_get_pretrained(teacher, student, raw_student, scratch):
    raw_stu_path = '../module/poly_raw_student.pth'
    tea_path = '../module/poly_teacher.pth'
    stu_path = '../module/poly_student_our.pth'
    dis_path = '../module/poly_discriminator_our.pth'
    if not scratch:
        if os.path.exists(stu_path):
            student.load_state_dict(torch.load(stu_path))
            print('load student compeleted')

        if os.path.exists(raw_stu_path):
            raw_student.load_state_dict(torch.load(raw_stu_path))
            print('load raw_stu_path compeleted')

        if os.path.exists(tea_path):
            teacher.load_state_dict(torch.load(tea_path))
            print('load teacher compeleted')

        if os.path.exists(dis_path):
            discriminator.load_state_dict(torch.load(dis_path))
            print('load discriminator compeleted')
    return teacher, student, raw_student

def show_img(pred_tea, title):
    pred_f_img = pred_tea.squeeze().cpu().detach().numpy()
    pred_f_img = pred_f_img.transpose(1, 2, 0)
    pred_mean = pred_f_img.mean(axis=0).mean(axis=1)
    pred_indcs = np.argsort(pred_mean)

    ans = np.zeros((14,14,3))
    ans[:, :, 0] = pred_f_img[:, :, pred_indcs[-1]]
    ans[:, :, 1] = pred_f_img[:, :, pred_indcs[-2]]
    ans[:, :, 2] = pred_f_img[:, :, pred_indcs[-3]]

    ans = (ans - ans.min()) / (ans.max() - ans.min())
    ans = (ans * 255).astype(np.int8)

    plt.figure()
    plt.imshow(ans, cmap='jet')
    plt.title(title)
    plt.show()

def get_ce_loss(img, label, network):
    pred, pred_f = network(img)
    errCE = ce_loss_func(pred, label)
    pred_label = torch.argmax(pred, dim=-1)
    acc = (pred_label == label).sum().float() / pred_label.shape[0]

    return errCE, acc, pred, pred_f

def save_model():
    print('update model..')
    stu_path = '../module/poly_student_our.pth'
    dis_path = '../module/poly_discriminator_our.pth'
    torch.save(student.state_dict(), stu_path)
    torch.save(discriminator.state_dict(), dis_path)

def get_prop(feat, net):
    return net.get_prop_from_feat(feat)

def train(teacher, student, epochs=1000, is_test=True):
    optimizer_stu = torch.optim.Adam(student.parameters(), lr=1e-4, weight_decay=1e-8)
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=1e-4, weight_decay=1e-8)
    prev_best = 0

    raw_student.eval()
    if is_test:
        phases = ('test',)
    else:
        phases = ('train', 'test')

    discriminator.train()

    for epoch in range(1, epochs):
        val_acc = ''
        for phase in iter(phases):
            if phase == 'train':
                teacher.train()
                student.train()
                ldr = loader
            else:
                teacher.eval()
                student.eval()
                ldr = val_loader
            summary = []

            i = 0
            for batch in tqdm(ldr):
                white_img, nbi_img, label, filename = \
                    batch[0].cuda().float(), batch[1].cuda().float(), batch[2].cuda().long(), batch[3][0]

                # show white img
                input_img_wl = white_img.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
                input_img_nbi = nbi_img.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
                plt.figure()
                plt.imshow(input_img_wl)
                plt.title('input_img_wl '+ filename)
                plt.show()

                plt.figure()
                plt.imshow(input_img_nbi)
                plt.title('input_img_nbi')
                plt.show()


                # vis
                _, _, pred_prob_pos, pred_tea = get_ce_loss(nbi_img, label, teacher)
                show_img(pred_tea, 'NBI features')

                errCE_wht, acc_wht, _, pred_ali = get_ce_loss(white_img, label, student)
                show_img(pred_ali, 'ALI features')

                errCE_wht, acc_wht, _, pred_wli = get_ce_loss(white_img, label, raw_student)
                show_img(pred_wli, 'WLI features')

                i+=1
                if i == 3:
                    exit(0)



if __name__ == '__main__':
    # is_test = False
    is_test = True

    dataset = PolyDataset(is_train=True, enable_aug=False)
    val_dataset = PolyDataset(is_train=True, enable_aug=False)
    bs = 16
    loader = DataLoader(dataset, batch_size=bs, num_workers=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=bs, shuffle=True)
    ce_loss_func = nn.CrossEntropyLoss()
    kl_func = nn.KLDivLoss()

    student = resnet50(pretrained=True, num_classes=2).cuda()
    raw_student = resnet50(pretrained=True, num_classes=2).cuda()
    teacher = resnet50(pretrained=True, num_classes=2).cuda()
    discriminator = Discriminator().cuda()
    discriminator.initialize()
    teacher, student, raw_student = try_get_pretrained(teacher, student, raw_student, scratch=False)
    train(teacher, student, is_test=is_test)

