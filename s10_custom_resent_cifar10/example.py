from utils import train_data, test_data, find_lr
from model import CustomResNet
from torch.utils.data import DataLoader
from train import train_custom_resnet
import torch

import matplotlib.pyplot as plt
import numpy as np


SEED = 42

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if cuda else "cpu")

# dataloader arguments
dataloader_args = dict(shuffle=True, batch_size=512, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

test_loader = DataLoader(test_data(), **dataloader_args)
train_loader = DataLoader(train_data(), **dataloader_args)

model = CustomResNet().to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()

# find_lr(model = model,
#         optimizer = optimizer,
#         criterion = criterion,
#         train_loader=train_loader,
#         test_loader=test_loader,
#         device=device)

# Train the custom ResNet model
lr_max=1.83e-02
# Train the custom ResNet model
train_custom_resnet(
    model = model,
    optimizer = optimizer,
    criterion = criterion,
    train_loader=train_loader,
    test_loader=test_loader,
    num_epochs=24,
    max_at_epoch=5,
    # lr_min=1e-4,
    lr_max=lr_max,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)