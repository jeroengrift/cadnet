import torch
import numpy as np
from torchvision.transforms import functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, anns, ann_ids, filename):
        for t in self.transforms:
            image, anns, filename = t(image, anns, ann_ids, filename)
        return image, anns, ann_ids, filename
    

class ToTensor(object):
    def __call__(self, image, anns, ann_ids, filename):        
        for key, val in anns.items():
            if isinstance(val, np.ndarray):
                anns[key] = torch.from_numpy(val)
        return F.to_tensor(image), anns, ann_ids, filename


class Normalize(object):
    def __init__(self, mean, std, to_255=False):
        self.mean = mean
        self.std = std
        self.to_255 = to_255

    def __call__(self, image, anns, filename):
        if self.to_255:
            image *= 255.0
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, anns, filename
