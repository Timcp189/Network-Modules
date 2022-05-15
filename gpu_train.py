import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision import transforms
from torch.utils.data import Dataset
import time
from Network import unet
from Dataloader import ConfAndISM
from RunManager import RunManager
from RunBuilder import RunBuilder

train_dataset = ConfAndISM(csv_file='dataset.csv', root_dir='All', transform=transforms.ToTensor())

#hyper-parameters that can be modified depending on how the network needs to be trained
params = OrderedDict(
    lr=[0.0001]
    , batch_size=[1]
    , kernel_size=[5]
    , stride=[1]
    , groups=[1]
    , device=['cuda']
    , optim=[torch.optim.Adam]
    , loss=[nn.MSELoss()]
    , epochs=[1000]
)

m = RunManager()
for run in RunBuilder.get_runs(params):
    device = torch.device(run.device)
    Unet = unet(1, 1, run.kernel_size, run.stride, run.groups).to(device) 
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=run.batch_size, shuffle=True)
    optimizer = run.optim(Unet.parameters(), lr=run.lr)

    m.begin_run(run, Unet, train_loader)
    for epoch in range(run.epochs):
        m.begin_epoch()
        t1 = time.time()
        for batch in train_loader:
            conf = batch[0].to(device)
            ism = batch[1].to(device)
            preds = Unet(conf)
            m.track_accuracy(preds, ism)
            criterion = run.loss
            loss = criterion(preds, ism)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            m.track_loss(loss, preds)
        m.end_epoch()
        t2 = time.time()
        print(t2-t1)
        '''
        the for loop below saves the network at the epoch values in the list.
        This is useful for loading the network at what ever stage of training
        in order to look at the progression of the predictions.
        '''
        if epoch in [0, 1, 4, 9, 24, 49, 99, 199, 399, 999]: 
            m.save(Unet, f'trained_unet_at epoch_{epoch+1} (lr=0.0001, bs=1, ks=3).pt')
    m.end_run()
m.csv('unet data')
