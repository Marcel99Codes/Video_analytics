import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
import torchvision.transforms as T
import random


class MiniUCFDataset(Dataset):
    def __init__(self, root_dir, mode="pretrain", num_frames=32):
        """
        root_dir: path to mini_UCF directory
        mode: "pretrain", "finetune_train", "finetune_val", "single"
        num_frames: fixed number of frames per video
        """
        self.root_dir = root_dir
        self.mode = mode
        self.num_frames = num_frames

        # get all video file paths and labels (if supervised)
        self.video_paths = []
        self.labels = []

        # Map folder names to labels for finetune
        class_names = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

        # For pretrain mode, use all videos
        # For finetune, split train/val by folders or predefined split
        # Here we just use all videos for simplicity
        for cls in class_names:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for filename in os.listdir(cls_dir):
                if filename.endswith(".avi") or filename.endswith(".mp4"):
                    path = os.path.join(cls_dir, filename)
                    self.video_paths.append(path)
                    self.labels.append(self.class_to_idx[cls])

        # For "single" mode (for visualization), just load one video - simplify here
        if mode == "single":
            # Use first video for demo
            self.video_paths = self.video_paths[:]
            self.labels = self.labels[:]

    def __len__(self):
        return len(self.video_paths)

    def temporal_uniform_sample(self, video):
        # video shape (C, T, H, W)
        C, T, H, W = video.shape
        num_frames = self.num_frames
        if T == num_frames:
            return video
        elif T > num_frames:
            indices = torch.linspace(0, T - 1, steps=num_frames).long()
            return video[:, indices, :, :]
        else:
            # Pad by repeating last frame
            pad_len = num_frames - T
            last_frame = video[:, -1:, :, :].repeat(1, pad_len, 1, 1)
            return torch.cat([video, last_frame], dim=1)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # Read video using torchvision: returns (frames, audio, info)
        # video: (T, H, W, C)
        video, _, _ = read_video(video_path, pts_unit='sec')

        # Convert to float tensor and permute to (C, T, H, W)
        video = video.permute(3, 0, 1, 2).float() / 255.0

        # Uniform sample or pad frames to fixed length
        video = self.temporal_uniform_sample(video)

        sample = {"video": video, "label": label}
        return sample


def get_dataset(mode="pretrain", path=None):
    """
    Factory method for datasets
    """
    if mode in ["pretrain", "finetune_train", "finetune_val"]:
        # Use fixed 32 frames for all modes
        return MiniUCFDataset(root_dir=path, mode=mode, num_frames=32)
    elif mode == "single":
        return MiniUCFDataset(root_dir=path, mode="single", num_frames=32)
    else:
        raise ValueError(f"Unknown dataset mode: {mode}")


def get_augmentation_pipeline():
    # Example augmentations for video tensors
    # Input tensor shape: (B, C, T, H, W) or (C, T, H, W)
    # We'll create a function that can work on single tensor (C, T, H, W)
    # Here a simple brightness jitter and random horizontal flip per frame

    brightness = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    horizontal_flip = T.RandomHorizontalFlip(p=0.5)
    gaussian_blur = T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))


    def augment(video):
        # video: tensor (C, num_frames, H, W)
        C, num_frames, H, W = video.shape
        augmented_frames = []
        for t in range(num_frames):
            frame = video[:, t, :, :]  # (C, H, W)
            # Convert to PIL Image
            pil_img = T.ToPILImage()(frame)
            pil_img = brightness(pil_img)
            pil_img = horizontal_flip(pil_img)
            pil_img = gaussian_blur(pil_img)
            # Back to tensor
            tensor_img = T.ToTensor()(pil_img)
            augmented_frames.append(tensor_img)
        augmented_video = torch.stack(augmented_frames, dim=1)  # (C, T, H, W)
        return augmented_video

    return augment
