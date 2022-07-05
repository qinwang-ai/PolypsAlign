import os
import torch
from torch.utils.data import DataLoader
from loader import PolyDataset
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from torchvision.models import resnet50, vgg16, inception_v3
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def try_get_pretrained(teacher, student, scratch):
    stu_path = '../module/poly_raw_student_inc3.pth'
    if not scratch:
        if os.path.exists(stu_path):
            student.load_state_dict(torch.load(stu_path))
            print('load student compeleted')
    return teacher, student

def get_ce_loss(img, label, network, phase):
    if phase == 'train':
        pred, pred_aux = network(img)
    else:
        pred = network(img)
    errCE = ce_loss_func(pred, label)
    pred_label = torch.argmax(pred, dim=-1)
    acc = (pred_label == label).sum().float() / pred_label.shape[0]
    return errCE, acc


def save_model():
    print('update model..')
    stu_path = '../module/poly_raw_student_inc3.pth'
    torch.save(student.state_dict(), stu_path)


def train(teacher, student, epochs=100, is_test=True):
    # optimizer_tea = torch.optim.Adam(teacher.parameters(), lr=1e-3, weight_decay=1e-8)
    optimizer_stu = torch.optim.Adam(student.parameters(), lr=1e-4, weight_decay=1e-8)
    prev_best = 0

    if is_test:
        phases = ('test',)
    else:
        phases = ('train', 'test')

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
                optimizer_stu.zero_grad()
                # optimizer_tea.zero_grad()

                errCE_wht, acc_wht = get_ce_loss(white_img, label, student, phase)
                # errCE_wht, acc_wht = get_ce_loss(nbi_img, label, teacher)
                if phase == 'train':
                    errCE_wht.backward()
                    optimizer_stu.step()
                else:
                    val_acc += "%f\n" % (acc_wht)
                summary.append((errCE_wht.item(), acc_wht.item()))
            summary = np.array(summary).mean(axis=0)

            if phase == 'train':
                print('Epoch %d' % epoch, 'CE_loss: %0.2f, acc: %0.2f' % (summary[0], summary[1]))
            else:
                if summary[-1] > prev_best:
                    prev_best = summary[-1]
                    if not is_test:
                        save_model()
                print('[EVAL] Epoch %d' % epoch, 'CE_loss: %0.2f, acc: %0.2f, best_acc: %0.3f' % (summary[0], summary[1], prev_best))
        if is_test:
            f = open('./white_acc.txt', 'w')
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

    student = inception_v3(pretrained=True, num_classes=2).cuda()
    teacher = inception_v3(pretrained=True, num_classes=2).cuda()
    teacher, student = try_get_pretrained(teacher, student, scratch=True)
    train(teacher, student, is_test=is_test, epochs=500)





