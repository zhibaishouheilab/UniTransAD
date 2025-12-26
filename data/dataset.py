import torch
import torch.utils.data as data
import numpy as np
from PIL import ImageEnhance, Image
import random
import os
from torchvision import transforms

# --- Helper Functions for Augmentation ---
def _norm(img):
    img -= img.min(1, keepdim=True)[0]
    img /= img.max(1, keepdim=True)[0]
    return img

def _color_enhance(image):
    bright_intensity = random.randint(8, 12) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(8, 12) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image

class BrainOmniADataset(data.Dataset):
    def __init__(self, 
                 img_size, 
                 image_root, 
                 sequences,  # Renamed from 'modal'
                 augment=False,
                 if_addlabel=False,
                 edge_root=None):
        """
        Dataset loader for Brain-OmniA benchmark.
        
        Args:
            img_size: Target image size (resize).
            image_root: Directory containing .npy files.
            sequences: List of sequence names (e.g. ['t1', 't2', 'flair']).
            augment: Whether to apply data augmentation.
            if_addlabel: If True, treats the last channel of npy as label.
            edge_root: Directory containing edge maps (optional).
        """
        self.sequences = sequences
        self.image_root = image_root
        # Mapping sequences to indices
        self.seq_indices = [i for i in range(len(self.sequences))]
        
        self.images = [os.path.join(self.image_root, f) 
                       for f in os.listdir(self.image_root) if f.endswith('.npy')]
        # Sort by numerical index in filename to ensure consistency
        self.images.sort(key=lambda x: int(os.path.basename(x).split(".npy")[0]))
        
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size)
        ])
        
        # Label transform uses Nearest Neighbor
        self.label_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.NEAREST)
        ])

        self.data_len = len(self.images)
        self.augment = augment
        self.if_addlabel = if_addlabel
        self.edge_root = edge_root
        print(f'[{self.__class__.__name__}] Number of slices: {self.data_len}')

    def __getitem__(self, index):
        # Load image npy
        npy_path = self.images[index]
        npy = np.load(npy_path)  # Shape: (C, H, W)
        
        label_img = None
        if self.if_addlabel:
            label_img = npy[-1, :, :]  # Last channel is label
            seq_data = [npy[i, :, :] for i in self.seq_indices]
        else:
            seq_data = [npy[i, :, :] for i in self.seq_indices]
        
        # Load edge npy if provided
        edge_data = None
        if self.edge_root is not None:
            edge_npy_path = os.path.join(self.edge_root, os.path.basename(npy_path))
            edge_npy = np.load(edge_npy_path)
            edge_data = [edge_npy[i, :, :] for i in self.seq_indices]

        # -------------- Data Augmentation --------------
        if self.augment:
            flip_flag = random.randint(0, 2)    # 0: None, 1: Vertical, 2: Horizontal
            rotate_flag = random.randint(0, 3)  # 0: 0, 1: 90, 2: 180, 3: 270
            
            # Augment sequences
            seq_data = self._apply_consistent_augmentation(seq_data, flip_flag, rotate_flag)
            
            # Augment label (Geometry only)
            if self.if_addlabel and label_img is not None:
                label_img = self._apply_label_augmentation(label_img, flip_flag, rotate_flag)
            
            # Augment edges (Geometry only)
            if edge_data is not None:
                edge_data = [self._apply_label_augmentation(ed, flip_flag, rotate_flag) for ed in edge_data]

        # -------------- Transform to Tensor / Resize --------------
        seq_data_tensors = [self.img_transform(item) for item in seq_data]
        
        edge_data_tensors = []
        if edge_data is not None:
            edge_data_tensors = [self.img_transform(item) for item in edge_data]
        
        label_tensor = None
        if self.if_addlabel and label_img is not None:
            label_tensor = self.label_transform(label_img)
        
        # -------------- Return --------------
        # Structure: [modal1, modal2, ..., edge1, edge2, ..., label]
        result = seq_data_tensors
        if edge_data is not None:
            result += edge_data_tensors
        
        if self.if_addlabel and label_tensor is not None:
            result.append(label_tensor)
            
        return result

    def __len__(self):
        return self.data_len

    def _apply_consistent_augmentation(self, seq_data, flip_flag, rotate_flag):
        """Geometry transform + Color Enhance for MRI sequences."""
        # Flip
        if flip_flag == 1:
            seq_data = [np.flip(img, 0).copy() for img in seq_data]
        elif flip_flag == 2:
            seq_data = [np.flip(img, 1).copy() for img in seq_data]

        # Rotate
        seq_data = [np.rot90(img, rotate_flag).copy() for img in seq_data]

        # Color Enhance (requires PIL conversion)
        # First ensure range is 0-255 uint8
        seq_data = [np.clip(img * 255, 0, 255).astype(np.uint8) for img in seq_data]
        new_seqs = []
        for img in seq_data:
            pil_img = Image.fromarray(img).convert('L')
            pil_img = _color_enhance(pil_img)
            new_seqs.append(pil_img)
        return new_seqs

    def _apply_label_augmentation(self, label_img, flip_flag, rotate_flag):
        """Geometry transform only for Labels/Edges. No Color Enhance."""
        if flip_flag == 1:
            label_img = np.flip(label_img, 0).copy()
        elif flip_flag == 2:
            label_img = np.flip(label_img, 1).copy()
        label_img = np.rot90(label_img, rotate_flag).copy()
        return label_img

def get_dataloader(batchsize, shuffle, 
                   sequences=['t1', 't2', 't1ce', 'flair'], 
                   pin_memory=True,
                   img_size=256, 
                   img_root='../data/train/', 
                   num_workers=16, 
                   augment=False,
                   if_addlabel=False,
                   edge_root=None):
    """Factory function for DataLoader."""
    dataset = BrainOmniADataset(
        img_size=img_size, 
        image_root=img_root,
        sequences=sequences, 
        augment=augment,
        if_addlabel=if_addlabel,
        edge_root=edge_root
    )
    data_loader = data.DataLoader(
        dataset=dataset, 
        batch_size=batchsize, 
        shuffle=shuffle,
        pin_memory=pin_memory, 
        num_workers=num_workers
    )
    return data_loader