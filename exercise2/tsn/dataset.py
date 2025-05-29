import os
import cv2
import random
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class VideoDataset(Dataset):
    def __init__(self, data_path, root_dir, mode='train', num_segments=4, transform=None, flow=False):
        self.data_path = data_path
        self.root_path = os.path.join(data_path, root_dir)
        self.mode = mode
        self.num_segments = num_segments
        self.transform = transform
        self.flow = flow
        self.videos = self._make_dataset()

    def _make_dataset(self):
        list_file = os.path.join(self.data_path, 'train.txt' if self.mode == 'train' else 'validation.txt')
        
        class_file = os.path.join(self.data_path, 'classes.txt')
        with open(class_file, 'r') as f:
            class_to_idx = {}
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                idx, name = parts
                class_to_idx[name] = int(idx)

        with open(list_file, 'r') as f:
            lines = f.readlines()
        
        video_label_pairs = []
        for line in lines:
            video_rel_path = line.strip()
            if not video_rel_path:
                continue 
            class_name = video_rel_path.split('/')[0]
            label = class_to_idx[class_name]
            video_path = os.path.join(self.root_path, video_rel_path)
            video_label_pairs.append((video_path, label))
        
        return video_label_pairs

    def _load_rgb_frames(self, video_path):
        video_path += ".avi"  # Add extension if you're pointing to folders
        cap = cv2.VideoCapture(video_path)
        frames = []
        success, frame = cap.read()
        while success:
            # Convert to PIL Image to apply transform
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(img)
            success, frame = cap.read()
        cap.release()
        return frames

    def _load_flow_frames(self, video_path):
        flow_x = sorted(glob(os.path.join(video_path, 'flow_x_*.jpg')))
        flow_y = sorted(glob(os.path.join(video_path, 'flow_y_*.jpg')))
        return list(zip(flow_x, flow_y))

    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        video_path, label = self.videos[idx]

        # Load frames depending on flow or RGB mode
        if self.flow:
            frames = self._load_flow_frames(video_path)
        else:
            frames = self._load_rgb_frames(video_path)

        total_frames = len(frames)
        if total_frames == 0:
            raise RuntimeError(f"No frames found in video {video_path}")

        seg_len = total_frames // self.num_segments
        if seg_len == 0:
            # If video is too short, just repeat last frame indices
            indices = [min(i, total_frames - 1) for i in range(self.num_segments)]
        else:
            indices = []
            for i in range(self.num_segments):
                start_idx = i * seg_len
                end_idx = (i + 1) * seg_len - 1
                # Clamp end_idx to last frame index to avoid index error
                end_idx = min(end_idx, total_frames - 1)
                if self.mode == 'train':
                    rand_idx = random.randint(start_idx, end_idx)
                else:
                    # take middle frame in segment for val/test
                    rand_idx = start_idx + (end_idx - start_idx) // 2
                indices.append(rand_idx)

        if self.flow:
            snippet = []
            for idx in indices:
                flow_stack = []
                for offset in range(-2, 3):
                    i = min(max(0, idx + offset), total_frames - 1)
                    fx = Image.open(frames[i][0])
                    fy = Image.open(frames[i][1])
                    if self.transform:
                        fx = self.transform(fx)
                        fy = self.transform(fy)
                    flow_stack.extend([fx, fy])
                snippet.append(torch.cat(flow_stack, dim=0))
            data = torch.stack(snippet)
        else:
            snippet = [frames[i].convert('RGB') for i in indices]
            if self.transform:
                snippet = [self.transform(f) for f in snippet]
            data = torch.stack(snippet)

        return data, label





