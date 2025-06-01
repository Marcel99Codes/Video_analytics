import os
import argparse
import torch
import torch.nn.functional as F
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import VideoDataset
from model import TSN

from torch.utils.tensorboard import SummaryWriter

# Data paths and model parameters ==> these should be adjusted according to your setup
DATA_PATH = "/media/marcel/Data1/video_analytics/data"
MODEL_PATH = "models"

os.makedirs(MODEL_PATH, exist_ok=True)

# Dynamic class count
num_classes = len(open(os.path.join(DATA_PATH, 'classes.txt')).readlines())
num_segments = 4
batch_size = 8
num_epochs = 10
learning_rate = 1e-3

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transforms
rgb_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

flow_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Datasets and Dataloaders
train_dataset_rgb = VideoDataset(DATA_PATH, 'mini_UCF', mode='train', num_segments=num_segments, transform=rgb_transform, flow=False)
val_dataset_rgb = VideoDataset(DATA_PATH, 'mini_UCF', mode='val', num_segments=num_segments, transform=rgb_transform, flow=False)
train_loader_rgb = DataLoader(train_dataset_rgb, batch_size=batch_size, shuffle=True)
val_loader_rgb = DataLoader(val_dataset_rgb, batch_size=batch_size, shuffle=False)

train_dataset_flow = VideoDataset(DATA_PATH, 'mini_UCF_flow', mode='train', num_segments=num_segments, transform=flow_transform, flow=True)
val_dataset_flow = VideoDataset(DATA_PATH, 'mini_UCF_flow', mode='val', num_segments=num_segments, transform=flow_transform, flow=True)
train_loader_flow = DataLoader(train_dataset_flow, batch_size=batch_size, shuffle=True)
val_loader_flow = DataLoader(val_dataset_flow, batch_size=batch_size, shuffle=False)

# Train function
def train_model(model, train_loader, val_loader, name, writer):
    model = model.to(device)
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print(len(train_loader), len(val_loader))
        print(f"Epoch {epoch+1}/{num_epochs}")

        count = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = ce_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            writer.add_scalar('step_loss', loss.item(), count)
            print(f"Step Loss: {loss.item():.4f}")
            count += 1

        writer.add_scalar('train_loss', running_loss/len(train_loader), epoch)
        print(f"train_loss: {running_loss/len(train_loader):.4f}")

        val_loss, val_acc = eval_model(model, val_loader, name, True)
        writer.add_scalar('validation_loss', val_loss, epoch)
        writer.add_scalar('validation_acc', val_acc, epoch)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.2f}%")

    #Save model
    file_path = os.path.join(MODEL_PATH, f"{name}_model.pth")
    torch.save(model.state_dict(), file_path)


# Evaluate function
def eval_model(model, val_loader, name, valid_during_training=False):
    if valid_during_training == False:
        file_path = os.path.join(MODEL_PATH, f"{name}_model.pth")
        model.load_state_dict(torch.load(file_path, map_location=device))

    model = model.to(device)
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    ce_loss = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = ce_loss(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy

# Late fusion evaluation
def late_fusion_eval(model_rgb, model_flow, loader_rgb, loader_flow):
    model_rgb.load_state_dict(torch.load(os.path.join(MODEL_PATH, "rgb_model.pth"), map_location=device))
    model_flow.load_state_dict(torch.load(os.path.join(MODEL_PATH, "flow_random_model.pth"), map_location=device))

    model_rgb = model_rgb.to(device).eval()
    model_flow = model_flow.to(device).eval()

    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for (inputs_rgb, labels_rgb), (inputs_flow, labels_flow) in zip(loader_rgb, loader_flow):
            inputs_rgb, labels_rgb = inputs_rgb.to(device), labels_rgb.to(device)
            inputs_flow, labels_flow = inputs_flow.to(device), labels_flow.to(device)

            outputs_rgb = F.softmax(model_rgb(inputs_rgb), dim=1)
            outputs_flow = F.softmax(model_flow(inputs_flow), dim=1)

            # Late fusion: average predictions
            fused_outputs = (outputs_rgb + outputs_flow) / 2.0
            _, predicted = torch.max(fused_outputs, 1)

            correct += (predicted == labels_rgb).sum().item()
            total += labels_rgb.size(0)

            for i in range(labels_rgb.size(0)):
                label = labels_rgb[i]
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

    accuracy = 100.0 * correct / total
    class_accuracies = [100.0 * c / t if t > 0 else 0.0 for c, t in zip(class_correct, class_total)]

    return accuracy, class_accuracies


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TSN model on video dataset")

    parser.add_argument('--train', help='Start training',action="store_true")
    parser.add_argument('--validate', help='Start validation', action="store_true")
    parser.add_argument('--model', help='Select the model to train (all, rgb, flow-random, flow-imagenet)', type=str, default='all')
    parser.add_argument('--fusion', help='Validate fusion', action="store_true")

    args = parser.parse_args()

    #Train models
    if args.train == True:
        if args.model == 'all' or args.model == 'rgb':
            # Train RGB model
            print("Training RGB model...")
            run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_dir = f"runs/rgb/run_{run_name}"
            writer = SummaryWriter(log_dir=log_dir)
            model = TSN(num_classes, num_segments, modality='rgb')
            train_model(model, train_loader_rgb, val_loader_rgb, name='rgb', writer=writer)
        if args.model == 'all' or args.model == 'flow-random':
            # Train Optical Flow model
            print("Training Optical Flow model...")
            run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_dir = f"runs/flow-random/run_{run_name}"
            writer = SummaryWriter(log_dir=log_dir)
            model = TSN(num_classes, num_segments, modality='flow', flow_init='random')
            train_model(model, train_loader_flow, val_loader_flow, name='flow_random', writer=writer)
        if args.model == 'all' or args.model == 'flow-imagenet':
            # Train Optical Flow model
            print("Training Optical Flow model...")
            run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_dir = f"runs/flow-imagenet/run_{run_name}"
            writer = SummaryWriter(log_dir=log_dir)
            model = TSN(num_classes, num_segments, modality='flow', flow_init='imagenet')
            train_model(model, train_loader_flow, val_loader_flow, name='flow_imagenet', writer=writer)

    # Validate models
    if args.validate == True:
        if args.model == 'all' or args.model == 'rgb':
            #Validate RGB model
            print("Validate RGB model...")
            model = TSN(num_classes, num_segments, modality='rgb')
            loss, acc = eval_model(model, val_loader_rgb, name='rgb', valid_during_training=False)
            print(f"Validation Loss: {loss:.4f}")
            print(f"Validation Accuracy: {acc:.2f}%")
        if args.model == 'all' or args.model == 'flow-random':
            # Validate flow-random model
            print("Validate flow-random model...")
            model = TSN(num_classes, num_segments, modality='flow', flow_init='random')
            loss, acc = eval_model(model, val_loader_flow, name='flow_random', valid_during_training=False)
            print(f"Validation Loss: {loss:.4f}")
            print(f"Validation Accuracy: {acc:.2f}%")
        if args.model == 'all' or args.model == 'flow-imagenet':
            # Validate flow-imagenet model
            print("Validate flow-imagenet model...")
            model = TSN(num_classes, num_segments, modality='flow', flow_init='imagenet')
            loss, acc = eval_model(model, val_loader_flow, name='flow_imagenet', valid_during_training=False)
            print(f"Validation Loss: {loss:.4f}")
            print(f"Validation Accuracy: {acc:.2f}%")
        
    if args.fusion == True:
        print("Performing late fusion evaluation...")
        model_rgb = TSN(num_classes, num_segments, modality='rgb')
        model_flow = TSN(num_classes, num_segments, modality='flow', flow_init='random')

        fusion_acc, class_accuracies = late_fusion_eval(model_rgb, model_flow, val_loader_rgb, val_loader_flow)

        print(f"Late Fusion Accuracy: {fusion_acc:.2f}%")
        print("Per-class accuracies:")
        with open(os.path.join(DATA_PATH, 'classes.txt')) as f:
            class_names = [line.strip() for line in f.readlines()]
        for i, acc in enumerate(class_accuracies):
            print(f"{class_names[i]:<20}: {acc:.2f}%")
