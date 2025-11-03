import os
import numpy as np
import glob
import shutil
import argparse

def is_label_empty(npy_file):
    """
    判断 npy 文件中最后一层（label 层）是否为空，即不包含大于0的数值。
    若数据加载失败或数据维度不足（少于3维），返回 None。
    否则返回布尔值：
        True 表示 label 层中不包含任何大于0的数值（为空），
        False 表示 label 层中至少存在一个大于0的数值。
    """
    try:
        data = np.load(npy_file)
    except Exception as e:
        print(f"加载文件 {npy_file} 时发生错误: {e}")
        return None

    if data.ndim < 3:
        print(f"文件 {npy_file} 数据维度不足，期望至少为3维数组。")
        return None

    label = data[-1]
    # 如果 label 中没有任何元素大于 0，则认为 label 为空，返回 True
    return not np.any(label > 20)

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--source_path", type=str, default="/mnt/sdb/zq/brain_sas_baseline/datasets/npy/BraTS2021/train/", help="输入 npy 文件所在文件夹")
    parser.add_argument("--dest_path", type=str, default="/mnt/sdb/zq/brain_sas_baseline/datasets/npy/BraTS2021/train_normal", help="保存处理后 npy 文件的文件夹")
    args = parser.parse_args()
    # 修改为你的源文件夹路径（包含 npy 文件的目录）
    source_path = args.source_path
    # 修改为目标文件夹路径，用于保存 label 为空的 npy 文件
    dest_path = args.dest_path

    # 若目标文件夹不存在，则创建
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    
    # 查找源路径下所有 npy 文件
    npy_files = glob.glob(os.path.join(source_path, "*.npy"))
    
    if not npy_files:
        print("在源路径下没有找到 npy 文件。")
        return
    
    total_files = 0        # 有效处理的文件数量
    selected_files = 0     # label 为空的文件数量

    for npy_file in npy_files:
        result = is_label_empty(npy_file)
        # 如果返回 None，说明文件加载或数据格式有问题，则跳过该文件
        if result is None:
            continue

        total_files += 1
        if result:
            selected_files += 1
            shutil.copy(npy_file, dest_path)
            print(f"已复制文件: {os.path.basename(npy_file)}")
        else:
            print(f"跳过文件: {os.path.basename(npy_file)}，其中包含大于0的label数据。")
    
    print("-" * 40)
    print(f"总处理有效文件数: {total_files}")
    print(f"label 为空的文件数量: {selected_files}")
    if total_files > 0:
        print(f"占比: {selected_files / total_files:.4f}")
    else:
        print("没有有效数据进行统计。")

if __name__ == '__main__':
    main()
