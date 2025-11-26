import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class VideoFolderDataset(Dataset):
    def __init__(self, root, class_to_label=None, seq_len=30, transform=None, sort_key=None):
        self.root = Path(root)
        self.seq_len = seq_len
        # detect classes automatically if not provided
        if class_to_label is None:
            classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
            class_to_label = {c:i for i,c in enumerate(classes)}
        self.class_to_label = class_to_label

        self.items = []
        for class_name, label in self.class_to_label.items():
            class_dir = self.root / class_name
            if not class_dir.exists(): continue
            for video_dir in sorted(class_dir.iterdir(), key=sort_key):
                if not video_dir.is_dir(): 
                    continue
                # optional: verify contains images
                self.items.append((video_dir, label))

        self.transform = transform or T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.items)

    def _load_sequence(self, folder):
        # expects files named 01.jpg ... padded two digits
        frames = []
        # try robust file listing if some names differ
        files = sorted([p for p in folder.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png')])
        if len(files) >= self.seq_len:
            files = files[:self.seq_len]
        # if len(files) < seq_len: we'll repeat last frame
        for i in range(self.seq_len):
            if i < len(files):
                img = Image.open(files[i]).convert('RGB')
            else:
                # repeat last available
                img = Image.open(files[-1]).convert('RGB')
            img_t = self.transform(img)  # C,H,W
            frames.append(img_t)
        seq = torch.stack(frames, dim=0)  # (seq_len, C, H, W)
        return seq

    def __getitem__(self, idx):
        folder, label = self.items[idx]
        seq = self._load_sequence(folder)
        return seq, torch.tensor(label, dtype=torch.long)