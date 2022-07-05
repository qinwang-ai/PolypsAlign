import torch

def get_feat_from_vgg(vgg, x):
    x = vgg.features(x)
    return x

def get_prop_from_feat(self, f):
    x = self.avgpool(f)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x

