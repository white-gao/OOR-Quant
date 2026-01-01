#!/usr/bin/env python3
"""数据采样脚本

从一个或多个路径（目录或文件）中的 parquet 文件采样指定数量的样本，保存为一个 parquet 文件。
"""

import argparse
import logging
import random
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


def find_parquet_files(directory: str, recursive: bool = True) -> List[str]:
    """查找目录下所有 parquet 文件。
    
    Args:
        directory: 目录路径
        recursive: 是否递归查找子目录
        
    Returns:
        parquet 文件路径列表
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")
    
    if not dir_path.is_dir():
        raise ValueError(f"路径不是目录: {directory}")
    
    pattern = "**/*.parquet" if recursive else "*.parquet"
    parquet_files = [str(p) for p in dir_path.glob(pattern) if p.is_file()]
    
    return sorted(parquet_files)


def collect_parquet_files(input_paths: List[str], recursive: bool = True) -> List[str]:
    """收集所有 parquet 文件路径。
    
    Args:
        input_paths: 输入路径列表（可以是文件或目录）
        recursive: 是否递归查找子目录
        
    Returns:
        parquet 文件路径列表
    """
    all_files = []
    
    for input_path in input_paths:
        path = Path(input_path)
        
        if not path.exists():
            logger.warning(f"路径不存在，跳过: {input_path}")
            continue
        
        if path.is_file():
            if path.suffix.lower() == '.parquet':
                all_files.append(str(path))
            else:
                logger.warning(f"不是 parquet 文件，跳过: {input_path}")
        elif path.is_dir():
            files = find_parquet_files(str(path), recursive=recursive)
            all_files.extend(files)
        else:
            logger.warning(f"未知路径类型，跳过: {input_path}")
    
    return sorted(list(set(all_files)))  # 去重并排序


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


def sample_dataframe(df: pd.DataFrame, num_samples: int, seed: int = None) -> pd.DataFrame:
    """从 DataFrame 中采样指定数量的样本。
    
    Args:
        df: 要采样的 DataFrame
        num_samples: 采样数量
        seed: 随机种子
        
    Returns:
        采样后的 DataFrame
    """
    if len(df) == 0:
        logger.warning("DataFrame 为空，无法采样")
        return pd.DataFrame()
    
    if num_samples <= 0:
        raise ValueError(f"num_samples 必须大于 0，当前值: {num_samples}")
    
    total_rows = len(df)
    
    if num_samples >= total_rows:
        logger.warning(f"采样数量 ({num_samples}) 大于等于总行数 ({total_rows})，返回全部数据")
        return df.copy()
    
    # 设置随机种子
    if seed is not None:
        random.seed(seed)
        logger.info(f"使用随机种子: {seed}")
    
    # 随机采样
    logger.info(f"从 {total_rows} 行中采样 {num_samples} 行...")
    sampled_indices = random.sample(range(total_rows), num_samples)
    sampled_df = df.iloc[sampled_indices].copy()
    
    logger.info(f"采样完成，共 {len(sampled_df)} 行")
    
    return sampled_df


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(
        description='从一个或多个路径中的 parquet 文件采样指定数量的样本，保存为一个 parquet 文件'
    )
    parser.add_argument(
        '--input',
        type=str,
        nargs='+',
        required=True,
        help='输入路径（可以是文件或目录），可以指定多个'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出 parquet 文件路径'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        required=True,
        help='采样数量'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='随机种子（可选）'
    )
    parser.add_argument(
        '--engine',
        choices=['pyarrow', 'fastparquet'],
        default='pyarrow',
        help='Parquet 处理引擎（默认: pyarrow）'
    )
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='不递归查找子目录中的文件'
    )
    
    args = parser.parse_args()
    
    # 验证参数
    if args.num_samples <= 0:
        logger.error(f"num_samples 必须大于 0，当前值: {args.num_samples}")
        sys.exit(1)
    
    try:
        # 1. 收集所有 parquet 文件
        logger.info("=" * 60)
        logger.info("步骤 1: 收集 parquet 文件...")
        parquet_files = collect_parquet_files(
            args.input,
            recursive=not args.no_recursive
        )
        
        if not parquet_files:
            logger.error("没有找到任何 parquet 文件")
            sys.exit(1)
        
        logger.info(f"找到 {len(parquet_files)} 个 parquet 文件")
        
        # 2. 加载所有文件
        logger.info("=" * 60)
        logger.info("步骤 2: 加载 parquet 文件...")
        combined_df = load_all_parquet_files(parquet_files, engine=args.engine)
        
        if len(combined_df) == 0:
            logger.error("没有加载到任何数据")
            sys.exit(1)
        
        # 3. 采样数据
        logger.info("=" * 60)
        logger.info("步骤 3: 采样数据...")
        sampled_df = sample_dataframe(
            combined_df,
            num_samples=args.num_samples,
            seed=args.seed
        )
        
        if len(sampled_df) == 0:
            logger.error("采样后数据为空")
            sys.exit(1)
        
        # 4. 保存结果
        logger.info("=" * 60)
        logger.info("步骤 4: 保存结果...")
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        sampled_df.to_parquet(
            output_path,
            engine='pyarrow',
            index=False,
            compression='snappy'
        )
        
        logger.info(f"结果已保存到: {output_path}")
        
        # 5. 输出统计信息
        logger.info("=" * 60)
        logger.info("处理完成！")
        logger.info(f"输入文件数: {len(parquet_files)}")
        logger.info(f"原始数据行数: {len(combined_df)}")
        logger.info(f"采样后行数: {len(sampled_df)}")
        logger.info(f"输出文件: {output_path}")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("\n操作被用户取消")
        sys.exit(1)
    except Exception as e:
        logger.error(f"程序执行失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

