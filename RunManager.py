from collections import OrderedDict
import torchvision
import time
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

class RunManager():

    def __init__(self):
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_accuracy = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None

        self.network = None
        self.device = None
        self.loader = None
        self.tb = None

    def get_accuracy(self, preds, labels):
        m = nn.MSELoss(reduction='sum')
        MSE = m(preds, labels)
        RMSE = np.sqrt(MSE.item())
        return RMSE


    def begin_run(self, run, network, loader):

        self.run_start_time = time.time()

        self.run_params = run
        self.run_count += 1

        self.network = network
        self.loader = loader
        self.tb = SummaryWriter(comment=f'-{run}')

        images, labels = next(iter(self.loader))
        grid = torchvision.utils.make_grid(images)

        self.tb.add_image('images', grid)
        self.tb.add_graph(
            self.network
            , images.to(getattr(run, 'device', 'cpu'))
        )

    def end_run(self):
        self.tb.close()
        self.epoch_count = 0
        self.epoch_accuracy = 0

    def begin_epoch(self):
        self.epoch_start_time = time.time()

        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_accuracy = 0

    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_accuracy

        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)
        #self.tb.add_scalar('Loss vs Accuracy (10^2)', loss, accuracy*100)

        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results['loss'] = loss
        results['accuracy'] = accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k, v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)

    def track_loss(self, loss, batch):
        self.epoch_loss += loss.item() * batch[0].shape[0]

    def track_accuracy(self, preds, labels):
        self.epoch_accuracy = self.get_accuracy(preds, labels)

    def save(self, model, path):
            torch.save(model.state_dict(), path)

    def csv(self, fileName):

        pd.DataFrame.from_dict(
            self.run_data,
            orient='columns'
        ).to_csv(f'{fileName}.csv')