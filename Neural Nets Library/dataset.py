
import random

from PIL import Image
from torchvision import datasets
from torchvision.datasets.folder import default_loader


class AdvancedImageFolder(datasets.ImageFolder):
    """
    A custom data loader where the images are typically arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

        However, we add the ability to filter subdirectories and shuffle.
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        filter_fn (callable, optional): A function that takes a pair (class_name, class_index)
        shuffle (bool, optional): Randomly rearrange the list of (image path, class_index) tuples
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, filter_fn=None, shuffle=False):
        super(AdvancedImageFolder, self).__init__(root, transform, target_transform, loader)

        if filter_fn is not None:
            self.imgs = list(filter(filter_fn, self.imgs))

        if shuffle:
            random.shuffle(self.imgs)

def greyscale_image_loader(path):
    """
    Returns a converted copy of the image at path.
    The conversion here involves translating a color image to black and white (mode “L”), 
    the library uses the ITU-R 601-2 luma transform.
    """
    with open(path, 'rb') as file:
        with Image.open(file) as img:
            return img.convert('L')
