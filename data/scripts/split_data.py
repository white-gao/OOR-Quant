#!/usr/bin/env python3
"""数据切割脚本

将 general text 数据和推荐数据合并后，按照每 1000 条样本切割成多个文件。
"""

import argparse
import json
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


def split_dataframe(df: pd.DataFrame, max_rows: int, output_dir: str, prefix: str = "part") -> List[str]:
    """将 DataFrame 按照固定行数切割成多个文件。
    
    Args:
        df: 要切割的 DataFrame
        max_rows: 每个文件的最大行数
        output_dir: 输出目录
        prefix: 输出文件前缀
        
    Returns:
        输出文件路径列表
    """
    if len(df) == 0:
        logger.warning("DataFrame 为空，无需切割")
        return []
    
    if max_rows <= 0:
        raise ValueError(f"max_rows 必须大于 0，当前值: {max_rows}")
    
    # 创建输出目录
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # 计算需要切割的文件数
    total_rows = len(df)
    num_chunks = (total_rows + max_rows - 1) // max_rows  # 向上取整
    logger.info(f"将数据切割成 {num_chunks} 个文件（每个文件最多 {max_rows} 行）")
    
    # 使用固定的 5 位数字格式，确保文件名格式统一
    # 格式：part-00000-of-00010.parquet
    num_digits = 5
    
    # 切割并保存
    output_files = []
    for chunk_idx in tqdm(range(num_chunks), desc="切割文件"):
        start_idx = chunk_idx * max_rows
        end_idx = min(start_idx + max_rows, total_rows)
        
        # 提取数据块
        chunk_df = df.iloc[start_idx:end_idx]
        
        # 生成输出文件名，格式：part-00000-of-00010.parquet
        output_filename = f"{prefix}-{chunk_idx:0{num_digits}d}-of-{num_chunks:0{num_digits}d}.parquet"
        output_path = output_dir_path / output_filename
        
        # 保存文件
        chunk_df.to_parquet(
            output_path,
            engine='pyarrow',
            index=False,
            compression='snappy'
        )
        
        output_files.append(str(output_path))
        logger.debug(f"  保存文件 {chunk_idx + 1}/{num_chunks}: {output_path} (行 {start_idx} 到 {end_idx - 1})")
    
    logger.info(f"成功切割成 {len(output_files)} 个文件")
    return output_files


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(
        description='将 general text 数据和推荐数据合并后，按照每 1000 条样本切割成多个文件'
    )
    parser.add_argument(
        '--general_text_path',
        type=str,
        required=True,
        help='General text 数据路径（目录或文件）'
    )
    parser.add_argument(
        '--rec_data_path',
        type=str,
        required=True,
        help='推荐数据路径（目录或文件）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='输出目录路径'
    )
    parser.add_argument(
        '--max_rows',
        type=int,
        default=1000,
        help='每个文件的最大行数（默认: 1000）'
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
    if args.max_rows <= 0:
        logger.error(f"max_rows 必须大于 0，当前值: {args.max_rows}")
        sys.exit(1)
    
    try:
        # 1. 查找所有 parquet 文件
        logger.info("=" * 60)
        logger.info("步骤 1: 查找 general text 数据文件...")
        general_text_path = Path(args.general_text_path)
        if general_text_path.is_file():
            general_text_files = [str(general_text_path)]
        else:
            general_text_files = find_parquet_files(
                args.general_text_path,
                recursive=not args.no_recursive
            )
        logger.info(f"找到 {len(general_text_files)} 个 general text 文件")
        
        logger.info("步骤 2: 查找推荐数据文件...")
        rec_data_path = Path(args.rec_data_path)
        if rec_data_path.is_file():
            rec_data_files = [str(rec_data_path)]
        else:
            rec_data_files = find_parquet_files(
                args.rec_data_path,
                recursive=not args.no_recursive
            )
        logger.info(f"找到 {len(rec_data_files)} 个推荐数据文件")
        
        # 2. 加载所有文件
        logger.info("=" * 60)
        logger.info("步骤 3: 加载 general text 数据...")
        general_text_df = load_all_parquet_files(general_text_files, engine=args.engine)
        
        logger.info("步骤 4: 加载推荐数据...")
        rec_data_df = load_all_parquet_files(rec_data_files, engine=args.engine)
        
        # 3. 合并数据
        logger.info("=" * 60)
        logger.info("步骤 5: 合并数据...")
        if len(general_text_df) == 0 and len(rec_data_df) == 0:
            logger.error("没有加载到任何数据")
            sys.exit(1)
        
        if len(general_text_df) == 0:
            combined_df = rec_data_df
            logger.info("只使用推荐数据")
        elif len(rec_data_df) == 0:
            combined_df = general_text_df
            logger.info("只使用 general text 数据")
        else:
            combined_df = pd.concat([general_text_df, rec_data_df], ignore_index=True)
            logger.info(f"合并完成: general text {len(general_text_df)} 行 + 推荐数据 {len(rec_data_df)} 行 = 总计 {len(combined_df)} 行")
        
        # 4. 切割数据
        logger.info("=" * 60)
        logger.info("步骤 6: 切割数据...")
        output_files = split_dataframe(
            combined_df,
            max_rows=args.max_rows,
            output_dir=args.output_dir,
            prefix="part"
        )
        
        # 5. 生成文件列表 JSON
        logger.info("=" * 60)
        logger.info("步骤 7: 生成文件列表 JSON...")
        output_dir_path = Path(args.output_dir)
        json_file_path = output_dir_path / "file_list.json"
        
        # 将文件路径转换为相对路径或绝对路径（使用绝对路径更可靠）
        file_list = [str(Path(f).absolute()) for f in output_files]
        
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(file_list, f, indent=2, ensure_ascii=False)
        
        logger.info(f"文件列表已保存到: {json_file_path} ({len(file_list)} 个文件)")
        
        # 6. 输出统计信息
        logger.info("=" * 60)
        logger.info("处理完成！")
        logger.info(f"输入文件数: general text {len(general_text_files)} 个, 推荐数据 {len(rec_data_files)} 个")
        logger.info(f"总数据行数: {len(combined_df)}")
        logger.info(f"输出文件数: {len(output_files)}")
        logger.info(f"输出目录: {args.output_dir}")
        logger.info(f"文件列表 JSON: {json_file_path}")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("\n操作被用户取消")
        sys.exit(1)
    except Exception as e:
        logger.error(f"程序执行失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

