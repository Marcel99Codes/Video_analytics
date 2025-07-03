import argparse
import random
from resnet3d import ResNet3D, BasicBlock3D
from utils import get_dataset, get_augmentation_pipeline
import matplotlib.pyplot as plt
from losses import nt_xent_loss, triplet_loss, info_nce_loss
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os

from torch.utils.tensorboard import SummaryWriter

DATA_ROOT_PATH = "/media/marcel/Data1/video_analytics/data/mini_UCF"
NUM_WORKER = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize(video_tensor):
    aug = get_augmentation_pipeline()
    view1 = aug(video_tensor)
    view2 = aug(video_tensor)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(view1[:, 0].permute(1, 2, 0))
    axs[0].set_title("View 1")
    axs[1].imshow(view2[:, 0].permute(1, 2, 0))
    axs[1].set_title("View 2")
    plt.show()


def finetune(model, dataloader_train, dataloader_val, num_classes=25, epochs=30,writer=None):
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Linear(512, num_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        classifier.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch in dataloader_train:
            x = batch['video'].to(device)
            labels = batch['label'].to(device)
            with torch.no_grad():
                features = model(x)
            logits = classifier(features)

            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = logits.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = train_loss / len(dataloader_train)
        accuracy = correct / total
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")
        if writer:
            writer.add_scalar('Finetune/Train_Loss', avg_loss, epoch)
            writer.add_scalar('Finetune/Train_Accuracy', accuracy, epoch)


def pretrain(model, dataloader, optimizer, loss_fn, epochs=10, writer=None):
    augmentation = get_augmentation_pipeline()
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            x = batch['video'].to(device)
            view1 = torch.stack([augmentation(x[i]) for i in range(x.size(0))]).to(device)
            view2 = torch.stack([augmentation(x[i]) for i in range(x.size(0))]).to(device)

            z1 = model(view1)
            z2 = model(view2)

            loss = loss_fn(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if writer:
                writer.add_scalar('Pretrain/loss_step', loss.item(), epoch * len(dataloader) + batch_idx)

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        if writer:
            writer.add_scalar('Pretrain/Loss', avg_loss, epoch)

    torch.save(model.state_dict(), "models/pretrained_resnet3d.pt")


if __name__ == "__main__":
    print("Start")
    writer = SummaryWriter(log_dir="runs/ex4")

    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="Visualize augmented views")
    parser.add_argument("--sample_path", type=str, default="/media/marcel/Data1/video_analytics/data/mini_UCF")

    parser.add_argument("--pretrain", action="store_true", help="Run self-supervised pretraining")
    parser.add_argument("--finetune", action="store_true", help="Run supervised finetune")
    parser.add_argument("--loss", choices=["ntxent", "triplet", "infonce"], default="ntxent")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    if args.visualize:
        dataset = get_dataset(mode="single", path=args.sample_path)
        random_idx = random.randint(0, len(dataset) - 1)
        print(f"Visualizing sample index: {random_idx}")
        video_tensor = dataset[random_idx]['video']
        visualize(video_tensor)

    elif args.pretrain:
        print("Start pretrain")
        dataset = get_dataset(mode="pretrain", path=DATA_ROOT_PATH)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=NUM_WORKER)

        model = ResNet3D(BasicBlock3D, [2, 2, 2, 2], num_classes=512, return_features=True).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = {"ntxent": nt_xent_loss, "triplet": triplet_loss, "infonce": info_nce_loss}[args.loss]

        pretrain(model, loader, optimizer, loss_fn, epochs=args.epochs, writer=writer)

    elif args.finetune:
        train_ds = get_dataset(mode="finetune_train", path=DATA_ROOT_PATH)
        val_ds = get_dataset(mode="finetune_val", path=DATA_ROOT_PATH)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=NUM_WORKER)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=NUM_WORKER)

        model = ResNet3D(BasicBlock3D, [2, 2, 2, 2], num_classes=512, return_features=True).to(device)

        finetune(model, train_loader, val_loader, num_classes=25, epochs=args.epochs, writer=writer)

    print("Finished")
