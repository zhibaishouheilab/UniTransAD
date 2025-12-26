import numpy as np
import os

def normalize_intensity(image, mask=None, percentile_lower=0.2, percentile_upper=99.8):
    """
    对图像进行强度归一化：
    1. 根据前景(mask)计算分位数。
    2. 截断异常值。
    3. 缩放到 [0, 1]。
    """
    if mask is None:
        # 简单的背景掩膜生成：非左上角像素值即为前景
        mask = image != image[0, 0, 0] if image.ndim == 3 else image != image[0, 0]
    
    # 避免全黑图像导致的报错
    if np.sum(mask) == 0:
        return image

    # 提取前景像素
    foreground = image[mask != 0].ravel()
    if len(foreground) == 0:
        return image

    cut_off_lower = np.percentile(foreground, percentile_lower)
    cut_off_upper = np.percentile(foreground, percentile_upper)
    
    res = np.copy(image)
    res[(res < cut_off_lower) & (mask != 0)] = cut_off_lower
    res[(res > cut_off_upper) & (mask != 0)] = cut_off_upper
    
    # 归一化到 0-1
    max_val = res.max()
    if max_val != 0:
        res = res / max_val
    
    return res

def save_slices(tensor, save_dir, file_prefix, start_slice=50, num_slices=60, crop_h=(10, 210), crop_w=(25, 225)):
    """
    通用的切片保存函数。
    tensor: (C, H, W, D)
    save_dir: 保存路径
    file_prefix: 文件名前缀 (通常是数字索引)
    """
    # 维度检查
    if tensor.shape[-1] < start_slice + num_slices:
        print(f"Warning: Volume depth {tensor.shape[-1]} is smaller than requested range.")
        return

    for j in range(num_slices):
        current_slice_idx = start_slice + j
        
        # 提取切片: (C, H_crop, W_crop)
        # 注意：这里保留了原始代码的 Crop 逻辑 (240x240 -> 200x200)
        tensor_slice = tensor[:, crop_h[0]:crop_h[1], crop_w[0]:crop_w[1], current_slice_idx]
        
        save_name = f"{file_prefix + j + 1}.npy"
        np.save(os.path.join(save_dir, save_name), tensor_slice)