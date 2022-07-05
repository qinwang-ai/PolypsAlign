import os
import torch
from torch.utils.data import DataLoader
from loader import PolyDataset
from tqdm import tqdm
import torch.nn as nn
import numpy as np
# from torchvision.models import resnet50, vgg16
from models.resnet50 import resnet50
from models.discriminator import Discriminator
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def try_get_pretrained(teacher, student, scratch):
    stu_path = '../module/poly_student.pth'
    tea_path = '../module/poly_teacher.pth'
    dis_path = '../module/poly_discriminator.pth'
    if not scratch:
        if os.path.exists(stu_path):
            student.load_state_dict(torch.load(stu_path))
            print('load student compeleted')

        if os.path.exists(tea_path):
            teacher.load_state_dict(torch.load(tea_path))
            print('load teacher compeleted')

        if os.path.exists(dis_path):
            discriminator.load_state_dict(torch.load(dis_path))
            print('load discriminator compeleted')
    return teacher, student

def get_ce_loss(img, label, network):
    pred, pred_f = network(img)
    errCE = ce_loss_func(pred, label)
    pred_label = torch.argmax(pred, dim=-1)
    acc = (pred_label == label).sum().float() / pred_label.shape[0]
    return errCE, acc, pred_f

def save_model():
    print('update model..')
    stu_path = '../module/poly_student.pth'
    dis_path = '../module/poly_discriminator.pth'
    torch.save(student.state_dict(), stu_path)
    torch.save(discriminator.state_dict(), dis_path)

def train(teacher, student, epochs=1000, is_test=True):
    optimizer_stu = torch.optim.Adam(student.parameters(), lr=1e-4, weight_decay=1e-8)
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=1e-4, weight_decay=1e-8)
    prev_best = 0

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

            for batch in tqdm(ldr):
                white_img, nbi_img, label, filename = \
                    batch[0].cuda().float(), batch[1].cuda().float(), batch[2].cuda().long(), batch[3][0]

                _, _, pred_tea = get_ce_loss(nbi_img, label, teacher)
                errCE_wht, acc_wht, pred_stu = get_ce_loss(white_img, label, student)
                if phase == 'train':
                    # train discriminator
                    optimizer_dis.zero_grad()
                    p_real = discriminator(pred_tea.detach())
                    p_fake = discriminator(pred_stu.detach())
                    real_label = torch.ones(p_real.shape[0]).cuda().long()
                    fake_label = torch.zeros(p_fake.shape[0]).cuda().long()
                    real_err = ce_loss_func(p_real, real_label)
                    fake_err = ce_loss_func(p_fake, fake_label)
                    acc_real = (p_real.argmax(dim=-1) == real_label).sum() / p_real.shape[0]
                    acc_fake = (p_fake.argmax(dim=-1) == fake_label).sum() / p_fake.shape[0]
                    gan_acc = (acc_real + acc_fake)/2
                    (real_err + fake_err).backward()
                    optimizer_dis.step()

                    # train student
                    optimizer_stu.zero_grad()
                    p_real = discriminator(pred_stu)
                    distil_err = ce_loss_func(p_real, torch.ones(p_real.shape[0]).cuda().long())
                    (errCE_wht + distil_err).backward()
                    optimizer_stu.step()
                else:
                    val_acc += "%f\n" % (acc_wht)
                summary.append((errCE_wht.item(), distil_err.item(), gan_acc.item(), acc_wht.item()))
            summary = np.array(summary).mean(axis=0)

            if phase == 'train':
                print('Epoch %d' % epoch, 'CE_loss: %0.2f, Distil_loss: %0.2f, gan_acc: %0.2f, acc: %0.2f' % (summary[0], summary[1], summary[2], summary[-1]))
            else:
                if summary[-1] > prev_best:
                    prev_best = summary[-1]
                    if not is_test:
                        save_model()
                print('[EVAL] Epoch %d' % epoch, 'CE_loss: %0.2f, acc: %0.2f, best_acc: %0.3f' % (summary[0], summary[-1], prev_best))
        if is_test:
            f = open('./nbi_acc.txt', 'w')
            f.write(val_acc)
            break


if __name__ == '__main__':
    is_test = False
    # is_test = True

    dataset = PolyDataset(is_train=True)
    val_dataset = PolyDataset(is_train=False)
    bs = 16
    loader = DataLoader(dataset, batch_size=bs, num_workers=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=bs, shuffle=False)
    ce_loss_func = nn.CrossEntropyLoss()
    kl_func = nn.KLDivLoss()

    student = resnet50(pretrained=True, num_classes=2).cuda()
    teacher = resnet50(pretrained=True, num_classes=2).cuda()
    discriminator = Discriminator().cuda()
    discriminator.initialize()
    teacher, student = try_get_pretrained(teacher, student, scratch=False)
    train(teacher, student, is_test=is_test)

