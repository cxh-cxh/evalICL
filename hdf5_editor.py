import h5py
import argparse
import os

def list_hdf5_contents(filename):
    """列出HDF5文件内容"""
    with h5py.File(filename, 'r') as f:
        print("\nHDF5文件结构:")
        f.visititems(print_item)

def print_item(name, obj):
    """递归打印HDF5项目"""
    indent = '    ' * name.count('/')
    if isinstance(obj, h5py.Group):
        print(f"{indent}📁 组: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}📊 数据集: {name} (形状: {obj.shape}, 类型: {obj.dtype})")

def delete_item(filename, path):
    """删除HDF5中的项目"""
    with h5py.File(filename, 'a') as f:
        if path in f:
            del f[path]
            print(f"成功删除: {path}")
        else:
            print(f"未找到项目: {path}")

def move_rename_item(filename, source_path, target_path):
    """移动或重命名HDF5中的项目"""
    with h5py.File(filename, 'a') as f:
        if source_path in f:
            f.move(source_path, target_path)
            print(f"成功将 {source_path} 移动/重命名为 {target_path}")
        else:
            print(f"未找到源项目: {source_path}")

def create_group(filename, path):
    """创建新组"""
    with h5py.File(filename, 'a') as f:
        if path in f:
            print(f"组已存在: {path}")
        else:
            f.create_group(path)
            print(f"成功创建组: {path}")

def create_dataset(filename, path, shape, dtype='float32'):
    """创建新数据集"""
    with h5py.File(filename, 'a') as f:
        if path in f:
            print(f"数据集已存在: {path}")
        else:
            f.create_dataset(path, shape=shape, dtype=dtype)
            print(f"成功创建数据集: {path} (形状: {shape}, 类型: {dtype})")

def main():
    parser = argparse.ArgumentParser(description='HDF5文件编辑器')
    parser.add_argument('filename', help='HDF5文件名')
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 列出内容命令
    list_parser = subparsers.add_parser('list', help='列出HDF5文件内容')
    
    # 删除命令
    delete_parser = subparsers.add_parser('delete', help='删除项目')
    delete_parser.add_argument('path', help='要删除的项目路径')
    
    # 移动/重命名命令
    move_parser = subparsers.add_parser('move', help='移动或重命名项目')
    move_parser.add_argument('source', help='源路径')
    move_parser.add_argument('target', help='目标路径')
    
    # 创建组命令
    group_parser = subparsers.add_parser('create_group', help='创建新组')
    group_parser.add_argument('path', help='新组路径')
    
    # 创建数据集命令
    dataset_parser = subparsers.add_parser('create_dataset', help='创建新数据集')
    dataset_parser.add_argument('path', help='新数据集路径')
    dataset_parser.add_argument('shape', type=int, nargs='+', help='数据集形状')
    dataset_parser.add_argument('--dtype', default='float32', 
                               choices=['float32', 'float64', 'int8', 'int16', 'int32', 'int64'],
                               help='数据类型 (默认: float32)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.filename):
        print(f"错误: 文件 {args.filename} 不存在")
        return
    
    if args.command == 'list':
        list_hdf5_contents(args.filename)
    elif args.command == 'delete':
        delete_item(args.filename, args.path)
    elif args.command == 'move':
        move_rename_item(args.filename, args.source, args.target)
    elif args.command == 'create_group':
        create_group(args.filename, args.path)
    elif args.command == 'create_dataset':
        create_dataset(args.filename, args.path, tuple(args.shape), args.dtype)
    else:
        # 如果没有命令，默认显示文件内容
        list_hdf5_contents(args.filename)

if __name__ == '__main__':
    main()