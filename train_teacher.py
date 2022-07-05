import os
import torch
from torch.utils.data import DataLoader
from loader_nbi import PolyDataset
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from torchvision.models import resnet50
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def try_get_pretrained(teacher, student, scratch):
    tea_path = '../module/poly_teacher.pth'
    if not scratch:

        if os.path.exists(tea_path):
            teacher.load_state_dict(torch.load(tea_path))
            print('load teacher compeleted')
    return teacher, student

def get_ce_loss(img, label, network):
    pred = network(img)
    errCE = ce_loss_func(pred, label)
    pred_label = torch.argmax(pred, dim=-1)
    acc = (pred_label == label).sum().float() / pred_label.shape[0]
    return errCE, acc

def save_model():
    print('update model..')
    tea_path = '../module/poly_teacher.pth'
    torch.save(teacher.state_dict(), tea_path)

def train(teacher, student, epochs=400, is_test=True):
    optimizer_tea = torch.optim.Adam(teacher.parameters(), lr=1e-4, weight_decay=1e-8)
    prev_best = 0

    if is_test:
        phases = ('test',)
    else:
        phases = ('train', 'test')

    loss_train = []
    loss_eval = []
    for epoch in range(1, epochs):
        val_acc = ''
        for phase in iter(phases):
            if phase == 'train':
                teacher.train()
                ldr = loader
            else:
                teacher.eval()
                ldr = val_loader
            summary = []

            for batch in tqdm(ldr):
                nbi_img, label, filename = \
                    batch[0].cuda().float(), batch[1].cuda().long(), batch[2][0]
                optimizer_tea.zero_grad()

                errCE_wht, acc_wht = get_ce_loss(nbi_img, label, teacher)
                if phase == 'train':
                    errCE_wht.backward()
                    optimizer_tea.step()
                else:
                    val_acc += "%f\n" % (acc_wht)
                summary.append((errCE_wht.item(), acc_wht.item()))
            summary = np.array(summary).mean(axis=0)

            if phase == 'train':
                print('Epoch %d' % epoch, 'CE_loss: %0.2f, acc: %0.2f' % (summary[0], summary[1]))
                loss_train.append(summary[-1])
            else:
                if summary[-1] > prev_best:
                    prev_best = summary[-1]
                    if not is_test:
                        save_model()
                print('[EVAL] Epoch %d' % epoch, 'CE_loss: %0.2f, acc: %0.2f, best_acc: %0.3f' % (summary[0], summary[1], prev_best))
                loss_eval.append(summary[-1])
        f = open('./loss_array.py', 'w')
        f.write('loss_train=%s\n' % str(loss_train))
        f.write('loss_eval=%s\n' % str(loss_eval))
        f.close()


if __name__ == '__main__':
    # is_test = False
    is_test = True

    dataset = PolyDataset(is_train=True)
    val_dataset = PolyDataset(is_train=False)
    bs = 16
    loader = DataLoader(dataset, batch_size=bs, num_workers=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=bs, shuffle=False)
    ce_loss_func = nn.CrossEntropyLoss()
    kl_func = nn.KLDivLoss()

    student = resnet50(pretrained=True, num_classes=2).cuda()
    teacher = resnet50(pretrained=True, num_classes=2).cuda()
    teacher, student = try_get_pretrained(teacher, student, scratch=False)
    train(teacher, student, is_test=is_test)

