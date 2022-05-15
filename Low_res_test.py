from Dataloader import ConfAndISM
import torch
from torchvision import transforms
from Network import unet
from torchvision.utils import save_image
import numpy as np

device = 'cuda'
test_dataset = ConfAndISM(csv_file='test_dataset.csv', root_dir='Test Images', transform=transforms.ToTensor())

Unet1 = unet(1, 1, 5, 1, 1).to(device)

epoch=[1,2,5,10,25,50,100,200,400,1000]

for i in epoch:
    Unet1.load_state_dict(torch.load(f'trained_unet_at epoch_{i} (lr=0.0001, bs=1, ks=3).pt'))

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1)

    batch = next(iter(test_loader))

    conf, ism = batch
    conf, ism = conf.to(device), ism.to(device)
    preds1 = Unet1(conf)

    trans = transforms.ToPILImage()

    preds1_img = trans(preds1[0])

    preds1_img.save(f'/home/tim/prediction at epoch {i}.tif')






