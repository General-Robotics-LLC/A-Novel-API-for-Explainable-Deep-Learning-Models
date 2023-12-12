import warnings
warnings.filterwarnings("ignore")

# General import
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import os

# Import from repo
from models.resnet import ResNet18
from train import *
from ExplainableNode import ExplainableNode  # Import ExplainableNode

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)

config = {
    'epochs': 80,
    'batch_size': 128
}

data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandAugment(num_ops=2, magnitude=7),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    # torchvision.transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0, inplace=False)
])

# Load Datasets
full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=data_transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=data_transform)

# Split Full Train Dataset into Train and Validation
train_idx, val_idx = train_test_split(list(range(len(full_train_dataset))), test_size=0.1)
train_dataset = Subset(full_train_dataset, train_idx)
val_dataset = Subset(full_train_dataset, val_idx)

# Data Loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

# Model, Loss, and Optimizer
model = ResNet18(num_classes=10, use_variant=False).to(device) # Turn use_variant = True to replicate modified Perrformance
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=0.02)
scheduler = ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.7)

# ExplainableNode instance for monitoring, set to None if not using
explainable_node = ExplainableNode(should_compute_psm=True, should_compute_lcc=False)
if explainable_node is not None:
    explainable_node.register_hooks(model, [model.layer1, model.layer2, model.layer3, model.layer4],
                                    ['layer1', 'layer2', 'layer3', 'layer4', 'layer5'])

# Run exper
experiment(model, train_loader, val_loader, optimizer,
           scheduler, criterion, config, device='cuda',
           explainable_node=explainable_node, test_loader=test_loader)


