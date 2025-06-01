import os
import cv2
import random
from PIL import Image
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, data_path, file_dir, mode='train', transform=None, clip_len=16, spatial_size=112, multi_view=False):
        self.data_path = data_path
        self.file_path = os.path.join(data_path, file_dir)
        self.mode = mode
        self.clip_len = clip_len
        self.spatial_size = spatial_size
        self.multi_view = multi_view
        self.transform = transform
        self.videos, self.class_to_idx = self._make_dataset()

    def _make_dataset(self):
        if self.mode == "train":
            list_file = os.path.join(self.data_path, 'train.txt')
        else:
            list_file = os.path.join(self.data_path, 'validation.txt')

        class_file = os.path.join(self.data_path, 'classes.txt')

        class_to_idx = {}
        with open(class_file, 'r') as f:
            for line in f:
                idx, name = line.strip().split()
                class_to_idx[name] = int(idx)

        video_label_pairs = []
        with open(list_file, 'r') as f:
            for line in f:
                video_rel_path = line.strip()
                if not video_rel_path:
                    continue
                class_name = video_rel_path.split('/')[0]
                label = class_to_idx[class_name]
                video_label_pairs.append((os.path.join(self.file_path, video_rel_path + ".avi"), label))
        return video_label_pairs, class_to_idx

    def _load_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        success, frame = cap.read()
        while success:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(img)
            success, frame = cap.read()
        cap.release()
        return frames

    def _sample_clip_indices(self, num_frames):
        if num_frames < self.clip_len:
            indices = list(range(num_frames)) + [num_frames - 1] * (self.clip_len - num_frames)
        else:
            if self.mode == 'train':
                start = random.randint(0, num_frames - self.clip_len)
                indices = list(range(start, start + self.clip_len))
            else:
                if self.multi_view:
                    step = max((num_frames - self.clip_len) // (4 - 1), 1)
                    indices_list = []
                    for v in range(4):
                        start = v * step
                        indices_list.append(list(range(start, start + self.clip_len)))
                    return indices_list
                else:
                    start = (num_frames - self.clip_len) // 2
                    indices = list(range(start, start + self.clip_len))
        return [indices]

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path, label = self.videos[idx]
        frames = self._load_frames(video_path)
        num_frames = len(frames)

        if num_frames == 0:
            raise RuntimeError(f"No frames found in video: {video_path}")

        clip_indices_list = self._sample_clip_indices(num_frames)
        clips = []

        for clip_indices in clip_indices_list:
            clip = [frames[i] for i in clip_indices]
            clip = [self.transform(frame) for frame in clip] 
            clip = torch.stack(clip, dim=0)
            clip = clip.permute(1, 0, 2, 3)
            clips.append(clip)

        if len(clips) == 1:
            return clips[0], label
        else:
            return clips, label
