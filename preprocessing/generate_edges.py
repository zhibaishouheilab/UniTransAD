import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)
        Gx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        Gy = torch.tensor([[ 1.0,  2.0,  1.0], [ 0.0,  0.0,  0.0], [-1.0, -2.0, -1.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0).unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)
        self.Repad = nn.ReplicationPad2d(padding=1)
    
    def forward(self, img):
        x = self.Repad(img)
        x = self.filter(x)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        
        # Robust normalization (Quantile based)
        nonzero = x[x > 0]
        if nonzero.numel() > 0:
            lower = torch.quantile(nonzero, 0.50)
            upper = torch.quantile(nonzero, 0.95)
        else:
            lower, upper = 0.0, 1.0
            
        x = torch.clamp(x, lower, upper)
        if upper - lower > 0:
            x = (x - lower) / (upper - lower)
        return x

def generate_edges(args):
    os.makedirs(args.output_folder, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sobel = Sobel().to(device)
    
    files = sorted([f for f in os.listdir(args.input_folder) if f.endswith('.npy')])
    print(f"Generating edges for {len(files)} files in {args.input_folder}...")
    
    for fname in tqdm(files):
        file_path = os.path.join(args.input_folder, fname)
        data = np.load(file_path) # (C, H, W)
        
        # Process all channels except the last one (Label)
        # Assuming data shape (C, H, W)
        channels = data[:-1]
        label = data[-1]
        
        # Prepare input: (B, C_in, H, W) -> handle each channel as a batch or separate item
        # Here we process channel by channel to apply separate normalization
        edge_maps = []
        
        for i in range(channels.shape[0]):
            img_tensor = torch.from_numpy(channels[i]).float().unsqueeze(0).unsqueeze(0).to(device) # (1, 1, H, W)
            edge_tensor = sobel(img_tensor)
            edge_maps.append(edge_tensor.squeeze().cpu().numpy())
            
        edge_maps = np.array(edge_maps)
        
        # Stack edges + original label
        out_data = np.concatenate([edge_maps, np.expand_dims(label, 0)], axis=0)
        
        np.save(os.path.join(args.output_folder, fname), out_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    args = parser.parse_args()
    generate_edges(args)