import os
import torch
from torch.utils.data import DataLoader
from loader import PolyDataset
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from torchvision.models import resnet50
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def try_get_pretrained(teacher, student, scratch):
    # stu_path = '../module/poly_student.pth'
    tea_path = '../module/poly_teacher.pth'
    if not scratch:
        # student.load_state_dict(torch.load(stu_path))
        teacher.load_state_dict(torch.load(tea_path))
    return teacher, student

def get_ce_loss(img, label, network):
    pred = network(img)
    errCE = ce_loss_func(pred, label)
    pred_label = torch.argmax(pred, dim=-1)
    acc = (pred_label == label).sum().float() / pred_label.shape[0]
    return errCE, acc, pred

def save_model():
    print('update model..')
    stu_path = '../module/poly_student.pth'
    torch.save(student.state_dict(), stu_path)

def train(teacher, student, epochs=500, is_test=True):
    optimizer_tea = torch.optim.Adam(teacher.parameters(), lr=1e-3, weight_decay=1e-8)
    optimizer_stu = torch.optim.Adam(student.parameters(), lr=1e-4, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_stu, step_size=50, gamma=0.1)
    prev_best = 0
    teacher.eval()

    if is_test:
        phases = ('test',)
    else:
        phases = ('train', 'test')

    for epoch in range(1, epochs):
        scheduler.step()
        for phase in iter(phases):

            if phase == 'train':
                student.train()
                ldr = loader
            else:
                student.eval()
                ldr = val_loader
            summary = []

            for batch in tqdm(ldr):
                white_img, nbi_img, label  = \
                    batch[0].cuda().float(), batch[1].cuda().float(), batch[2].cuda().long()
                optimizer_stu.zero_grad()
                # optimizer_tea.zero_grad()

                errCE_wht, acc_wht, pred_stu = get_ce_loss(white_img, label, student)
                _, _, pred_tea = get_ce_loss(nbi_img, label, teacher)

                # KL
                errKL = F.kl_div(F.log_softmax(pred_tea.detach(), dim=1), F.softmax(pred_stu, dim=1))
                # errKL = F.kl_div(F.log_softmax(pred_stu, dim=1), F.softmax(pred_tea.detach(), dim=1))
                # errKL = 0.5 * (errKL_tea + errKL_stu)

                # Distill
                err = errCE_wht + 10*errKL

                if phase == 'train':
                    err.backward()
                    optimizer_stu.step()
                summary.append((errCE_wht.item(), errKL.item(), acc_wht.item()))
            summary = np.array(summary).mean(axis=0)

            if phase == 'train':
                print('Epoch %d' % epoch, 'CE_loss: %0.2f, KL_loss: %0.4f, acc: %0.2f' % (summary[0], summary[1], summary[2]))
            else:
                if summary[-1] > prev_best:
                    prev_best = summary[-1]
                    if not is_test:
                        save_model()
                print('[EVAL] Epoch %d' % epoch, 'CE_loss: %0.2f, acc: %0.2f, best_acc: %0.3f' % (summary[0], summary[-1], prev_best))


if __name__ == '__main__':
    is_test = False
    # is_test = True
    dataset = PolyDataset(is_train=True)
    val_dataset = PolyDataset(is_train=False)
    bs = 16
    loader = DataLoader(dataset, batch_size=bs, num_workers=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, num_workers=bs, shuffle=False)
    ce_loss_func = nn.CrossEntropyLoss()
    kl_func = nn.KLDivLoss()

    student = resnet50(pretrained=True, num_classes=2).cuda()
    teacher = resnet50(pretrained=True, num_classes=2).cuda()
    teacher, student = try_get_pretrained(teacher, student, scratch=False)
    train(teacher, student, is_test=is_test)

