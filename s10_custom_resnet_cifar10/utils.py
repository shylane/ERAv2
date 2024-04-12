import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch_lr_finder import LRFinder
from tqdm import tqdm
import numpy

# Train data transformations
def train_data():
    train_transforms= v2.Compose([
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),
        # v2.CutOut(8, 8),
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        # torch.fliplr(),
        CutoutAfterToTensor(n_holes=1, length=8, fill_color=torch.tensor([0.4914, 0.4822, 0.4465])),
        v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    return CIFAR10('../data', train=True, download=True, transform=train_transforms)


# Test data transformations
def test_data():
    test_transforms= v2.Compose([
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    return CIFAR10('../data', train=False, download=True, transform=test_transforms)

def find_lr(train_loader,test_loader,device,model,optimizer,criterion):
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, val_loader=test_loader, end_lr=10, num_iter=100, step_mode="exp")
    lr_finder.plot(log_lr=False,skip_end=0)
    lr_finder.reset()
    
# Cutout implementation by David Stutz
# Refer https://davidstutz.de/2-percent-test-error-on-cifar10-using-pytorch-autoagument/
class CutoutAfterToTensor(object):
    def __init__(self, n_holes, length, fill_color=torch.tensor([0,0,0])):
        self.n_holes = n_holes
        self.length = length
        self.fill_color = fill_color
 
    def __call__(self, img):
        h = img.shape[1]
        w = img.shape[2]
        mask = numpy.ones((h, w), numpy.float32)
        for _ in range(self.n_holes):
            rng = numpy.random.default_rng(42)
            y = rng.integers(h)
            x = rng.integers(w)
            y1 = numpy.clip(y - self.length // 2, 0, h)
            y2 = numpy.clip(y + self.length // 2, 0, h)
            x1 = numpy.clip(x - self.length // 2, 0, w)
            x2 = numpy.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask + (1 - mask) * self.fill_color[:, None, None]
        return img

