"""Custom faces dataset."""
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random

class FacesDataset(Dataset):
    """Faces dataset.

    Attributes:
        root_path: str. Directory path to the dataset. This path has to
        contain a subdirectory of real images called 'real' and a subdirectory
        of not-real images (fake / synthetic images) called 'fake'.
        transform: torch.Transfrandom.randorm. Transform or a bunch of transformed to be
        applied on every image.
    """
    def __init__(self, root_path: str, transform=None, triplet=False):
        """Initialize a faces dataset."""
        self.root_path = root_path
        self.real_image_names = os.listdir(os.path.join(self.root_path, 'real'))
        self.fake_image_names = os.listdir(os.path.join(self.root_path, 'fake'))
        self.transform = transform
        self.triplet = triplet

    def __getitem__(self, index):
        """Get a sample and label from the dataset."""

        if index < len(self.real_image_names):
            image_path = os.path.join(self.root_path, 'real', self.real_image_names[index])
            pos_idx = random.randint(0, len(self.real_image_names) - 1)
            neg_idx = random.randint(0, len(self.fake_image_names) - 1)
            while self.real_image_names[pos_idx].split('~')[0] == self.real_image_names[index].split('~')[0]:
                pos_idx = random.randint(0, len(self.real_image_names) - 1)
            pos_path = os.path.join(self.root_path, 'real', self.real_image_names[pos_idx])
            neg_path = os.path.join(self.root_path, 'fake', self.fake_image_names[neg_idx])

            label = 0
        else:
            image_path = os.path.join(self.root_path, 'fake',
                                      self.fake_image_names[index - len(self.real_image_names)])
            neg_idx = random.randint(0, len(self.real_image_names) - 1)
            pos_idx = random.randint(0, len(self.fake_image_names) - 1)
            while self.fake_image_names[pos_idx].split('~')[0] == self.fake_image_names[index - len(self.real_image_names)].split('~')[0]:
                pos_idx = random.randint(0, len(self.fake_image_names) - 1)
            pos_path = os.path.join(self.root_path, 'fake', self.fake_image_names[pos_idx])
            neg_path = os.path.join(self.root_path, 'real', self.real_image_names[neg_idx])
            label = 1

        if self.triplet:
            anchor = self.load_image(image_path)
            pos = self.load_image(pos_path)
            neg = self.load_image(neg_path)
            tensor_image = [anchor, pos, neg]
        else:
            tensor_image = self.load_image(image_path)
        return tensor_image, label

    def load_image(self, image_path):
        image = Image.open(image_path)
        if self.transform is not None:
            tensor_image = self.transform(image)
        else:
            converter = transforms.PILToTensor()
            tensor_image = converter(image)
        return tensor_image

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.real_image_names) + len(self.fake_image_names)
