
import random

from PIL import Image
from torchvision import datasets
from torchvision.datasets.folder import default_loader


class AdvancedImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, filter_fn=None, shuffle=False):
        super(AdvancedImageFolder, self).__init__(root, transform, target_transform, loader)

        if filter_fn is not None:
            self.imgs = list(filter(filter_fn, self.imgs))

        if shuffle:
            random.shuffle(self.imgs)

def greyscale_image_loader(path):
    with open(path, 'rb') as file:
        with Image.open(file) as img:
            return img.convert('L')
