from torch.utils.data import Dataset
import jpeg4py as jpeg
import os
import numpy as np
import torchvision
from skimage import io
import jpeg4py as jpeg
import torch

class HelenDataset(Dataset):
    # HelenDataset

    def __init__(self, txt_file, root_dir, transform=None):
        """
        Args:
            txt_file (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.name_list = np.loadtxt(os.path.join(root_dir, txt_file), dtype="str", delimiter=',')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        img_name = self.name_list[idx, 1].strip()
        img_path = os.path.join(self.root_dir, 'images',
                                img_name + '.jpg')
        labels_path = [os.path.join(self.root_dir, 'labels',
                                    img_name,
                                    img_name + "_lbl%.2d.png") % i
                       for i in range(11)]
        image = jpeg.JPEG(img_path).decode()
        labels = np.array([io.imread(labels_path[i]) for i in range(11)])
        sample = {'image': image, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample