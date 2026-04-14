import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import models, transforms, datasets

train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(0, shear=10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

from torch.nn.modules.linear import Linear
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_asl = models.efficientnet_b0(weights='IMAGENET1K_V1')

model_asl.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(1280, 29)
)

model_asl.to(device)
