import os
from scipy import ndimage as ni
import pandas as pd
from torch.utils.data import Dataset
from skimage import io
import numpy as np

class ConfAndISM(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        image = image.astype('float32')
        image = ni.zoom(image, [2, 2], order=1)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = self.transform(image)

        label_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1])
        label = io.imread(label_path)
        label = label.astype('float32')
        label = (label - np.min(label)) / (np.max(label) - np.min(label))
        label = self.transform(label)

        return image, label