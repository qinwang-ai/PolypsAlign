# Dataset
Download PolypsDataset from  https://drive.google.com/drive/folders/1e2t5HhQf08sTAE_CPRNVgpi6YUKgQSHn

# Training:
- 'train.py' for full package model Training
- 'train_backbone.py' where 'backbone' can be vgg, resnet50, and inceptionV3

- 'train_wo_ali.py' for ablation study of alignment loss 

- 'train_wo_cont.py' for ablation study of contrastive loss 

# Others:
- 'splits' folder contains 5-fold cross-validation settings.
- 'utils/val_spliter.py' is utilized for generating settings for training.
