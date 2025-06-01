import os
from datetime import datetime
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision import transforms
from dataset import VideoDataset
from model import ResNet3D, BasicBlock3D, inflate_weights_2d_to_3d

from torch.utils.tensorboard import SummaryWriter

#Data paths and model parameters ==> these should be adjusted according to your setup
DATA_PATH = "/media/marcel/Data1/video_analytics/data"
MODEL_PATH = "models"

num_epochs = 15
learning_rate = 1e-4
num_classes = 25 

def get_train_transforms(spatial_size=112):
    return transforms.Compose([
        transforms.RandomResizedCrop(spatial_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
    ])

def get_val_transforms(spatial_size=112):
    return transforms.Compose([
        transforms.Resize(int(spatial_size * 1.15)),
        transforms.CenterCrop(spatial_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
    ])

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0, 0
    print(len(loader.dataset))
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        print(f"Step Loss: {total_loss / (len(loader.dataset) + 1e-8):.4f}")
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train TSN model on video dataset")

    parser.add_argument('--init', help='Initialization of the 3d ResNet (inflation, random)', type=str, default="random")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"runs/rgb/run_{run_name}"
    writer = SummaryWriter(log_dir=log_dir)

    model = ResNet3D(BasicBlock3D, [2, 2, 2, 2], num_classes=num_classes)
    if args.init == "inflation":
        resnet2d = resnet18(pretrained=True)
        inflate_weights_2d_to_3d(model, resnet2d)
    model = model.to(device)

    train_dataset = VideoDataset(
        data_path= DATA_PATH,
        file_dir="mini_UCF",
        transform=get_train_transforms()
    )
    val_dataset = VideoDataset(
        data_path=DATA_PATH,
        file_dir="mini_UCF",
        transform=get_val_transforms(),
        mode='val',
        multi_view=False
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

        writer.add_scalar('train_loss', train_loss, epoch)
        print(f"train_loss: {train_loss:.4f}")

        writer.add_scalar('validation_loss', val_loss, epoch)
        writer.add_scalar('validation_acc', val_acc, epoch)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.2f}%")

