import numpy as np
from matplotlib import pylab as plt
import nibabel as nib
import random
import glob
import os
from PIL import Image
import imageio

def normalize(image, mask=None, percentile_lower=0.2, percentile_upper=99.8):

    if mask is None:
        mask = image != image[0, 0, 0]
    cut_off_lower = np.percentile(image[mask != 0].ravel(), percentile_lower)
    cut_off_upper = np.percentile(image[mask != 0].ravel(), percentile_upper)
    res = np.copy(image)
    res[(res < cut_off_lower) & (mask != 0)] = cut_off_lower
    res[(res > cut_off_upper) & (mask != 0)] = cut_off_upper
    res = res / res.max()  # 0-1

    return res

def visualize(t1_data,t2_data,flair_data,t1ce_data,gt_data):

    plt.figure(figsize=(8, 8))
    plt.subplot(231)
    plt.imshow(t1_data[:, :], cmap='gray')
    plt.title('Image t1')
    plt.subplot(232)
    plt.imshow(t2_data[:, :], cmap='gray')
    plt.title('Image t2')
    plt.subplot(233)
    plt.imshow(flair_data[:, :], cmap='gray')
    plt.title('Image flair')
    plt.subplot(234)
    plt.imshow(t1ce_data[:, :], cmap='gray')
    plt.title('Image t1ce')
    plt.subplot(235)
    plt.imshow(gt_data[:, :])
    plt.title('GT')
    plt.show()

def visualize_to_gif(t1_data, t2_data, t1ce_data, flair_data):
    transversal = []
    coronal = []
    sagittal = []
    slice_num = t1_data.shape[2]
    for i in range(slice_num):
        sagittal_plane = np.concatenate((t1_data[:, :, i], t2_data[:, :, i],
                              t1ce_data[:, :, i],flair_data[:, :, i]),axis=1)
        coronal_plane = np.concatenate((t1_data[i, :, :], t2_data[i, :, :],
                              t1ce_data[i, :, :],flair_data[i, :, :]),axis=1)
        transversal_plane = np.concatenate((t1_data[:, i, :], t2_data[:, i, :],
                              t1ce_data[:, i, :],flair_data[:, i, :]),axis=1)
        transversal.append(transversal_plane)
        coronal.append(coronal_plane)
        sagittal.append(sagittal_plane)
    imageio.mimsave("./transversal_plane.gif", transversal, duration=0.01)
    imageio.mimsave("./coronal_plane.gif", coronal, duration=0.01)
    imageio.mimsave("./sagittal_plane.gif", sagittal, duration=0.01)
    return

    
if __name__ == '__main__':
    base_folder = '/mnt/sdb/zq/brain_sas_baseline/datasets/IXI/Registered'
    
    # 获取所有图像路径
    t2_list = sorted(glob.glob(os.path.join(base_folder, 'T2/*-T2_registered.nii.gz')))
    t1_list = sorted(glob.glob(os.path.join(base_folder, 'T1/*-T1_registered.nii.gz')))
    pd_list = sorted(glob.glob(os.path.join(base_folder, 'PD/*-PD_registered.nii.gz')))
    
    train_path = '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/IXI/train/'
    test_path = '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/IXI/test/'

    # 创建字典，键为共同前缀，值为文件路径
    t1_dict = {}
    for path in t1_list:
        # 提取共同前缀（去除-T1_stripped_registered.nii.gz部分）
        prefix = os.path.basename(path).replace('-T1_registered.nii.gz', '')
        t1_dict[prefix] = path
    
    t2_dict = {}
    for path in t2_list:
        prefix = os.path.basename(path).replace('-T2_registered.nii.gz', '')
        t2_dict[prefix] = path
    
    pd_dict = {}
    for path in pd_list:
        prefix = os.path.basename(path).replace('-PD_registered.nii.gz', '')
        pd_dict[prefix] = path
    
    # 找到三个字典共有的前缀（即三个模态都存在的图像）
    common_prefixes = set(t1_dict.keys()) & set(t2_dict.keys()) & set(pd_dict.keys())
    common_prefixes = sorted(list(common_prefixes))  # 排序保持一致性
    
    print(f"找到 {len(common_prefixes)} 个三个模态都存在的图像")
    
    # 创建保存目录
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    # 划分训练集和测试集（这里使用9:1的比例，可根据需要调整）
    train_ratio = 1
    train_len = int(len(common_prefixes) * train_ratio)
    
    count_number = 0
    
    for i, prefix in enumerate(common_prefixes):
        print(f'预处理第 {i+1}/{len(common_prefixes)} 个受试者: {prefix}')
        
        # 获取三个模态的路径
        t1_path = t1_dict[prefix]
        t2_path = t2_dict[prefix]
        pd_path = pd_dict[prefix]
        
        # 加载图像数据
        t1_img = nib.load(t1_path)
        t2_img = nib.load(t2_path)
        pd_img = nib.load(pd_path)
        
        # 获取图像数据数组
        pd_data = pd_img.get_fdata()
        t2_data = t2_img.get_fdata()
        t1_data = t1_img.get_fdata()
        
        # 归一化处理
        t2_data = normalize(t2_data)
        pd_data = normalize(pd_data)
        t1_data = normalize(t1_data)
        
        # 堆叠成张量 (模态, 高度, 宽度, 深度)
        tensor = np.stack([t1_data, t2_data, pd_data])  

        # 保存切片
        if i < train_len:
            save_path = train_path
            index_offset = i * 60
        else:
            save_path = test_path
            index_offset = (i - train_len) * 60
            
        for j in range(60):
            # 提取切片
            tensor_slice = tensor[:, 10:210, 25:225, 50 + j]
            # 保存为npy文件
            np.save(f"{save_path}{index_offset + j + 1}.npy", tensor_slice)