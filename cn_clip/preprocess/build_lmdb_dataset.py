# -*- coding: utf-8 -*-
# 这个脚本用于将图像和图像-文本对的注释序列化为 LMDB 文件。
# LMDB（Lightning Memory-Mapped Database）是一种快速的、基于内存映射的键值存储。
# 使用 LMDB 可以在训练过程中高效加载数据集并随机访问样本。

import argparse  # 用于解析命令行参数
import os  # 用于与文件系统交互
from tqdm import tqdm  # 用于显示进度条
import lmdb  # 用于创建和管理 LMDB 数据库
import json  # 用于处理 JSON 数据
import pickle  # 用于序列化和反序列化 Python 对象

# 解析命令行参数的函数
def parse_args():
    parser = argparse.ArgumentParser()  # 创建一个参数解析器
    parser.add_argument(
        "--data_dir", type=str, required=True, help="存储图像 TSV 文件和文本 JSONL 注释的目录"
        # --data_dir: 包含数据集文件（图像和注释）的目录路径。
    )
    parser.add_argument(
        "--splits", type=str, required=True, help="指定要处理的数据集分割，多个分割用逗号分隔（例如：train,valid,test）"
        # --splits: 要处理的数据集分割的逗号分隔列表（例如：train, valid, test）。
    )
    parser.add_argument(
        "--lmdb_dir", type=str, default=None, help="指定存储输出 LMDB 文件的目录。如果未提供，则默认为 {args.data_dir}/lmdb"
        # --lmdb_dir: 保存生成的 LMDB 文件的路径。默认为 data_dir 下名为 'lmdb' 的子目录。
    )
    return parser.parse_args()  # 解析并返回参数

if __name__ == "__main__":
    args = parse_args()  # 解析命令行参数
    assert os.path.isdir(args.data_dir), "data_dir 不存在！请检查输入参数..."
    # 确保提供的数据目录存在

    # 读取并处理指定的数据集分割
    specified_splits = list(set(args.splits.strip().split(",")))  # 分割输入的分割字符串并去重
    print("要处理的数据集分割: {}".format(", ".join(specified_splits)))

    # 设置输出 LMDB 目录
    if args.lmdb_dir is None:
        args.lmdb_dir = os.path.join(args.data_dir, "lmdb")  # 如果未提供 lmdb_dir，则默认为 {data_dir}/lmdb

    for split in specified_splits:  # 处理每个数据集分割
        # 为当前分割的 LMDB 文件创建目录
        lmdb_split_dir = os.path.join(args.lmdb_dir, split)
        if os.path.isdir(lmdb_split_dir):
            print("将覆盖已有的 LMDB 文件 {}".format(lmdb_split_dir))
        os.makedirs(lmdb_split_dir, exist_ok=True)  # 如果目录不存在，则创建

        # 创建用于存储图像的 LMDB
        lmdb_img = os.path.join(lmdb_split_dir, "imgs")  # 图像 LMDB 的路径
        '''
        map_size 是你为这个数据库预分配的虚拟内存上限
        在使用 LMDB（Lightning Memory-Mapped Database）时，它会将整个数据库映射到内存地址空间中。这个参数决定了你的数据库最大能长到多大。
        LMDB 是基于内存映射（Memory-mapped files）的。当你初始化数据库时，操作系统需要预留一段连续的虚拟地址空间。
        如果你的数据量超过了 map_size，LMDB 会抛出 MapFullError 错误。如果你设置得很大（比如你在代码里写的 1024^4Byte，即 1TB），
        它并不会立即占用 1TB 的物理磁盘空间或内存，而仅仅是定义了一个上限
        '''
        env_img = lmdb.open(lmdb_img, map_size=1024**4)  # 打开一个用于图像的 LMDB 环境，设置较大的 map_size
        txn_img = env_img.begin(write=True)  # 开始一个用于图像 LMDB 的写事务。输出：一个 txn 对象，它是你操作数据库的“笔”

        # 创建用于存储图像-文本对的 LMDB
        lmdb_pairs = os.path.join(lmdb_split_dir, "pairs")  # 图像-文本对 LMDB 的路径
        env_pairs = lmdb.open(lmdb_pairs, map_size=1024**4)  # 打开一个用于图像-文本对的 LMDB 环境，设置较大的 map_size
        txn_pairs = env_pairs.begin(write=True)  # 开始一个用于图像-文本对 LMDB 的写事务

        # 处理当前分割的文本注释
        pairs_annotation_path = os.path.join(args.data_dir, "MR_{}_queries.jsonl".format(split))
        # 当前分割的图像-文本对注释 JSONL 文件的路径
        with open(pairs_annotation_path, "r", encoding="utf-8") as fin_pairs:
            write_idx = 0  # 初始化写入的对数计数器
            for line in tqdm(fin_pairs):  # 使用进度条迭代 JSONL 文件中的每一行
                line = line.strip()  # 去除行首尾的空白字符
                obj = json.loads(line)  # 从行中解析 JSON 对象
                if split == "test":
                    obj['item_ids'] = []
                for field in ("query_id", "query_text", "item_ids"):
                    assert field in obj, "字段 {} 在行 {} 中不存在。请检查文本注释 JSONL 文件的完整性。"
                    # 确保 JSON 对象中存在所需字段
                if split == "test":
                    dump = pickle.dumps((obj['item_ids'], obj['query_id'], obj['query_text']))  # 序列化 (image_id, text_id, text) 元组
                    txn_pairs.put(key="{}".format(write_idx).encode('utf-8'), value=dump)  # 将序列化数据存储到 LMDB 中
                    write_idx += 1  # 计数器加一
                    if write_idx % 5000 == 0:  # 每写入 5000 条记录提交一次事务以节省内存
                        txn_pairs.commit()
                        txn_pairs = env_pairs.begin(write=True)  # 开始一个新事务
                else:
                    for image_id in obj["item_ids"]:  # 遍历与文本关联的图像 ID 列表
                        
                        '''
                        pickle.dumps(...):
                        干什么：把 Python 的元组 (图片ID, 问题ID, 问题内容) 变成一串二进制字节流。
                        为什么要这么干：LMDB 只能存“字节”，不能直接存 Python 的列表或字典。
                        
                        txn_pairs.put(key, value):
                        干什么：像存字典一样存入数据库。
                        参数：key 是数据的索引（这里用的是数字序号），value 是刚才序列化的二进制。注意：Key 和 Value 都必须是 bytes 类型，所以要用 .encode('utf-8')
                        '''
                        dump = pickle.dumps((image_id, obj['query_id'], obj['query_text']))  # 序列化 (image_id, text_id, text) 元组
                        txn_pairs.put(key="{}".format(write_idx).encode('utf-8'), value=dump)  # 将序列化数据存储到 LMDB 中
                        write_idx += 1  # 计数器加一
                        if write_idx % 5000 == 0:  # 每写入 5000 条记录提交一次事务以节省内存
                            txn_pairs.commit()
                            txn_pairs = env_pairs.begin(write=True)  # 开始一个新事务
                        
            # 在数据库里存入一个**“计数标签”**，记录这个数据集里总共有多少条数据
            # 将样本总数存储到 LMDB 中
            txn_pairs.put(key=b'num_samples', value="{}".format(write_idx).encode('utf-8'))
            
            
            txn_pairs.commit()  # 提交最终事务
            env_pairs.close()  # 关闭图像-文本对的 LMDB 环境
        print("完成序列化 {} {} 分割的图像-文本对到 {}。".format(write_idx, split, lmdb_pairs))

        # 处理当前分割的图像数据
        base64_path = os.path.join(args.data_dir, "MR_{}_imgs.tsv".format(split))
        # 当前分割的 TSV 文件路径，包含图像数据（图像 ID 和 base64 编码的图像字符串）
        with open(base64_path, "r", encoding="utf-8") as fin_imgs:
            write_idx = 0  # 初始化写入的图像计数器
            for line in tqdm(fin_imgs):  # 使用进度条迭代 TSV 文件中的每一行
                line = line.strip()  # 去除行首尾的空白字符
                image_id, b64 = line.split("\t")  # 将行分割为图像 ID 和 base64 编码的图像字符串
                txn_img.put(key="{}".format(image_id).encode('utf-8'), value=b64.encode("utf-8"))  # 将图像数据存储到 LMDB 中
                write_idx += 1  # 计数器加一
                if write_idx % 1000 == 0:  # 每写入 1000 条记录提交一次事务以节省内存
                    txn_img.commit()
                    txn_img = env_img.begin(write=True)  # 开始一个新事务
            # 将图像总数存储到 LMDB 中
            txn_img.put(key=b'num_images', value="{}".format(write_idx).encode('utf-8'))
            
            
            txn_img.commit()  # 提交最终事务
            env_img.close()  # 关闭图像的 LMDB 环境。关闭文件句柄，释放资源。
        print("完成序列化 {} {} 分割的图像到 {}。".format(write_idx, split, lmdb_img))

    print("完成！")