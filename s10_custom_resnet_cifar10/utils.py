import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch_lr_finder import LRFinder
from tqdm import tqdm
import numpy
# from torch.optim.lr_scheduler import OneCycleLR

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

# # One Cycle Policy
# def one_cycle_policy(optimizer, epochs, max_at_epoch, steps, lr_min, lr_max):
#     scheduler = OneCycleLR(
#         optimizer,
#         max_lr=lr_max,
#         steps_per_epoch=steps,
#         epochs=epochs,
#         pct_start=max_at_epoch / epochs
#     )
#     return scheduler


# def criterion():
#     return F.nll_loss

# def optimizer(model):
#     return optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# def scheduler(model):
#     return optim.lr_scheduler.StepLR(optimizer(model), step_size=15, gamma=0.1, verbose=True)

# def GetCorrectPredCount(pPrediction, pLabels):
#     return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer, criterion, train_losses, train_acc):
    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss
        loss = criterion(pred, target)
        train_loss+=loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        
        correct += GetCorrectPredCount(pred, target)
        processed += len(data)

        pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    return train_acc + [100*correct/processed], train_losses + [train_loss/len(train_loader)]

def test(model, device, test_loader, criterion, test_losses, test_acc):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_acc+[100. * correct / len(test_loader.dataset)], test_losses+[test_loss]
