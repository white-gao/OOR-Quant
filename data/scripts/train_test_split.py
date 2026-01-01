#!/usr/bin/env python3
"""训练/测试集分割脚本

从多个 parquet 文件中随机选择 N 条样本作为测试集，剩余数据作为训练集。
两个数据集在保存前都会进行 shuffle。
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def load_all_parquet_files(file_paths: List[str], engine: str = 'pyarrow') -> pd.DataFrame:
    """加载所有 parquet 文件并合并。
    
    Args:
        file_paths: parquet 文件路径列表
        engine: parquet 引擎，'pyarrow' 或 'fastparquet'
        
    Returns:
        合并后的 DataFrame
    """
    if not file_paths:
        logger.warning("没有找到 parquet 文件")
        return pd.DataFrame()
    
    logger.info(f"找到 {len(file_paths)} 个 parquet 文件，开始加载...")
    
    dataframes = []
    for file_path in tqdm(file_paths, desc="加载文件"):
        try:
            df = pd.read_parquet(file_path, engine=engine)
            logger.debug(f"  加载 {file_path}: {len(df)} 行")
            dataframes.append(df)
        except Exception as e:
            logger.error(f"  加载失败 {file_path}: {e}")
            continue
    
    if not dataframes:
        logger.warning("没有成功加载任何文件")
        return pd.DataFrame()
    
    # 合并所有 DataFrame
    logger.info("合并所有数据...")
    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"合并完成，总计 {len(combined_df)} 行")
    
    return combined_df


def split_train_test(
    df: pd.DataFrame,
    test_size: int,
    seed: int = None
) -> tuple:
    """将 DataFrame 分割为训练集和测试集。
    
    Args:
        df: 要分割的 DataFrame
        test_size: 测试集样本数量
        seed: 随机种子
        
    Returns:
        (train_df, test_df) 元组
    """
    if len(df) == 0:
        logger.warning("DataFrame 为空，无法分割")
        return pd.DataFrame(), pd.DataFrame()
    
    if test_size <= 0:
        raise ValueError(f"test_size 必须大于 0，当前值: {test_size}")
    
    total_rows = len(df)
    
    if test_size >= total_rows:
        logger.warning(
            f"测试集大小 ({test_size}) 大于等于总行数 ({total_rows})，"
            f"将使用全部数据作为测试集，训练集为空"
        )
        return pd.DataFrame(), df.copy()
    
    # 使用 pandas 的 sample 方法进行随机采样，确保可复现性
    if seed is not None:
        logger.info(f"使用随机种子: {seed}")
    
    logger.info(f"从 {total_rows} 行中随机选择 {test_size} 行作为测试集...")
    
    # 使用 pandas sample 方法随机选择测试集
    test_df = df.sample(n=test_size, random_state=seed).copy()
    # 获取测试集的索引
    test_indices = set(test_df.index)
    # 剩余数据作为训练集
    train_df = df.drop(test_indices).copy()
    
    logger.info(f"分割完成: 训练集 {len(train_df)} 行, 测试集 {len(test_df)} 行")
    
    return train_df, test_df


def shuffle_dataframe(df: pd.DataFrame, seed: int = None) -> pd.DataFrame:
    """对 DataFrame 进行 shuffle。
    
    Args:
        df: 要 shuffle 的 DataFrame
        seed: 随机种子（用于可复现性）
        
    Returns:
        shuffle 后的 DataFrame
    """
    if len(df) == 0:
        return df.copy()
    
    # 使用 sample 方法进行 shuffle（frac=1 表示全部采样，即打乱）
    # random_state 参数确保可复现性
    shuffled_df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return shuffled_df


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(
        description='从多个 parquet 文件中随机选择 N 条样本作为测试集，剩余数据作为训练集'
    )
    parser.add_argument(
        '--input_files',
        type=str,
        nargs='+',
        required=True,
        help='输入 parquet 文件路径列表（可以指定多个文件）'
    )
    parser.add_argument(
        '--test_size',
        type=int,
        required=True,
        help='测试集样本数量'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='输出目录路径'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='随机种子（可选，用于可复现性）'
    )
    parser.add_argument(
        '--engine',
        choices=['pyarrow', 'fastparquet'],
        default='pyarrow',
        help='Parquet 处理引擎（默认: pyarrow）'
    )
    parser.add_argument(
        '--test_filename',
        type=str,
        default='test.parquet',
        help='测试集输出文件名（默认: test.parquet）'
    )
    parser.add_argument(
        '--train_filename',
        type=str,
        default='train.parquet',
        help='训练集输出文件名（默认: train.parquet）'
    )
    
    args = parser.parse_args()
    
    # 验证参数
    if args.test_size <= 0:
        logger.error(f"test_size 必须大于 0，当前值: {args.test_size}")
        sys.exit(1)
    
    # 验证输入文件是否存在
    input_files = []
    for file_path in args.input_files:
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"文件不存在，跳过: {file_path}")
            continue
        if not path.is_file():
            logger.warning(f"路径不是文件，跳过: {file_path}")
            continue
        if path.suffix.lower() != '.parquet':
            logger.warning(f"不是 parquet 文件，跳过: {file_path}")
            continue
        input_files.append(str(path))
    
    if not input_files:
        logger.error("没有找到任何有效的 parquet 文件")
        sys.exit(1)
    
    try:
        # 1. 加载所有 parquet 文件
        logger.info("=" * 60)
        logger.info("步骤 1: 加载 parquet 文件...")
        combined_df = load_all_parquet_files(input_files, engine=args.engine)
        
        if len(combined_df) == 0:
            logger.error("没有加载到任何数据")
            sys.exit(1)
        
        # 2. 分割训练集和测试集
        logger.info("=" * 60)
        logger.info("步骤 2: 分割训练集和测试集...")
        train_df, test_df = split_train_test(
            combined_df,
            test_size=args.test_size,
            seed=args.seed
        )
        
        if len(test_df) == 0:
            logger.error("测试集为空，无法继续")
            sys.exit(1)
        
        # 3. Shuffle 数据
        logger.info("=" * 60)
        logger.info("步骤 3: Shuffle 数据...")
        
        # 为训练集和测试集使用不同的种子偏移，确保 shuffle 结果不同
        # 如果提供了 seed，使用不同的偏移量；否则都使用 None（完全随机）
        train_seed = (args.seed + 1000) if args.seed is not None else None
        test_seed = (args.seed + 2000) if args.seed is not None else None
        
        logger.info("Shuffle 训练集...")
        train_df = shuffle_dataframe(train_df, seed=train_seed)
        
        logger.info("Shuffle 测试集...")
        test_df = shuffle_dataframe(test_df, seed=test_seed)
        
        # 4. 保存结果
        logger.info("=" * 60)
        logger.info("步骤 4: 保存结果...")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        test_path = output_dir / args.test_filename
        train_path = output_dir / args.train_filename
        
        logger.info(f"保存测试集到: {test_path}")
        test_df.to_parquet(
            test_path,
            engine='pyarrow',
            index=False,
            compression='snappy'
        )
        
        if len(train_df) > 0:
            logger.info(f"保存训练集到: {train_path}")
            train_df.to_parquet(
                train_path,
                engine='pyarrow',
                index=False,
                compression='snappy'
            )
        else:
            logger.warning("训练集为空，跳过保存")
        
        # 5. 输出统计信息
        logger.info("=" * 60)
        logger.info("处理完成！")
        logger.info(f"输入文件数: {len(input_files)}")
        logger.info(f"原始数据行数: {len(combined_df)}")
        logger.info(f"训练集行数: {len(train_df)}")
        logger.info(f"测试集行数: {len(test_df)}")
        logger.info(f"输出目录: {output_dir}")
        logger.info(f"训练集文件: {train_path}")
        logger.info(f"测试集文件: {test_path}")
        if args.seed is not None:
            logger.info(f"随机种子: {args.seed}")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("\n操作被用户取消")
        sys.exit(1)
    except Exception as e:
        logger.error(f"程序执行失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

