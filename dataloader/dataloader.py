import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from mnist import MNIST
from PIL import Image

class ClassificationDataset(Dataset):
    def __init__(self, data_dir: str, mode: str = 'train', split: float = 0.8, aug: bool = True):

        if aug:
            self.augmentation = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((28, 28)),
                transforms.ToTensor()
            ])
        else:
            self.augmentation = transforms.Compose([
                transforms.ToTensor()
            ])

        mndata = MNIST(data_dir)
        if mode == 'train':
            images, labels = mndata.load_training()
            self.filelist = images[0:int(split*len(images))]
            self.labels = labels[0:int(split*len(labels))]
        elif mode == 'val':
            images, labels = mndata.load_training()
            self.filelist = images[int(split*len(images)):]
            self.labels = labels[int(split*len(labels)):]
        elif mode == 'test':
            images, labels = mndata.load_testing()
            self.filelist = images
            self.labels = labels
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx: torch.Tensor):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        image = self.filelist[idx]
        image = (np.array(image)).reshape((28, 28))
        image = np.stack([image,image,image], axis = 2)
        image = Image.fromarray(np.uint8(image))
        image = self.augmentation(image)
        label = self.labels[idx]
        return image, label
