import argparse
from datetime import datetime
import random
from resnet3d import ResNet3D, BasicBlock3D
from utils import get_dataset, get_augmentation_pipeline
import matplotlib.pyplot as plt
from losses import nt_xent_loss, triplet_loss, info_nce_loss
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import os

from torch.utils.tensorboard import SummaryWriter

PRETRAIN_PATH = "./models/pretrained_resnet3d.pt"
NUM_WORKER = 8

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


def finetune(model, pretrained_path, dataloader_train, dataloader_val, num_classes=25, epochs=30, writer=None):
    # Load the pretrained model
    if os.path.exists(pretrained_path) == False:
        raise FileNotFoundError(f"Pretrained model not found at {pretrained_path}")
    else:
        model.load_state_dict(torch.load(pretrained_path))

    # Freeze the backbone
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Linear(512, num_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Trainings loop
    for epoch in range(epochs):
        classifier.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch in dataloader_train:
            x = batch['video'].to(device)
            labels = batch['label'].to(device)

            with torch.no_grad():
                features = model(x)  # Extract frozen features
            logits = classifier(features)

            if (labels >= 25).any() or (labels < 0).any():
                print("Bad label detected:", labels)


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
        print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        if writer:
            writer.add_scalar('Finetune/Train_Loss', avg_loss, epoch)
            writer.add_scalar('Finetune/Train_Accuracy', accuracy, epoch)

    # Validation 
    classifier.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0

    with torch.no_grad():
        for batch in dataloader_val:
            x = batch['video'].to(device)
            labels = batch['label'].to(device)

            features = model(x)
            logits = classifier(features)
            loss = criterion(logits, labels)

            val_loss += loss.item()
            _, preds = logits.max(1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_avg_loss = val_loss / len(dataloader_val)
    val_accuracy = val_correct / val_total
    print(f"Accuracy: {val_accuracy:.4f}")


def pretrain(model, dataloader, optimizer, loss_fn, epochs=10, writer=None):
    augmentation = get_augmentation_pipeline()
    model.train()

    # Training loop
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
            print(f"Current Loss: {loss.item()}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        if writer:
            writer.add_scalar('Pretrain/Loss', avg_loss, epoch)

        torch.save(model.state_dict(), PRETRAIN_PATH + "_" + str(epoch))
    torch.save(model.state_dict(), PRETRAIN_PATH)


if __name__ == "__main__":
    print("Start")

    run_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_data_path", type=str, default="/media/marcel/Data1/video_analytics/data/mini_UCF")
    parser.add_argument("--visualize", action="store_true", help="Visualize augmented views")
    parser.add_argument("--sample_path", type=str, default="/media/marcel/Data1/video_analytics/data/mini_UCF")

    parser.add_argument("--pretrain", action="store_true", help="Run self-supervised pretraining")
    parser.add_argument("--finetune", action="store_true", help="Run supervised finetune")
    parser.add_argument("--dataset_samples", type=int, default=300)

    parser.add_argument("--loss", choices=["ntxent", "triplet", "infonce"], default="ntxent")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    if args.visualize:
        print("Start visualize")
        dataset = get_dataset(mode="single", path=args.sample_path)
        random_idx = random.randint(0, len(dataset) - 1)
        print(f"Visualizing sample index: {random_idx}")
        video_tensor = dataset[random_idx]['video']
        visualize(video_tensor)

    elif args.pretrain:
        print("Start pretrain")
        dataset_full = get_dataset(mode="pretrain", path=args.root_data_path)
        subset_size = args.dataset_samples
        subset_indices = random.sample(range(len(dataset_full)), subset_size)
        dataset = Subset(dataset_full, subset_indices)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=NUM_WORKER)

        model = ResNet3D(BasicBlock3D, [2, 2, 2, 2], num_classes=512, return_features=True).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = {"ntxent": nt_xent_loss, "triplet": triplet_loss, "infonce": info_nce_loss}[args.loss]

        pretrain(model, loader, optimizer, loss_fn, epochs=args.epochs, writer=writer)

    elif args.finetune:
        print("Start finetuning")
        train_ds = get_dataset(mode="finetune_train", path=args.root_data_path)
        subset_size = args.dataset_samples
        subset_indices = random.sample(range(len(train_ds)), subset_size)
        train_dataset = Subset(train_ds, subset_indices)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=NUM_WORKER)
        
        val_ds = get_dataset(mode="finetune_val", path=args.root_data_path)
        subset_size = args.dataset_samples
        subset_indices = random.sample(range(len(train_ds)), subset_size)
        val_dataset = Subset(val_ds, subset_indices)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=NUM_WORKER)

        model = ResNet3D(BasicBlock3D, [2, 2, 2, 2], num_classes=512, return_features=True).to(device)

        finetune(model, PRETRAIN_PATH, train_loader, val_loader, num_classes=25, epochs=args.epochs, writer=writer)

    print("Finished")
