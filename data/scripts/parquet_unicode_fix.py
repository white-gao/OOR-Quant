#!/usr/bin/env python3
"""Parquet Unicode 修复脚本

修复 parquet 文件中 messages 和 segments 字段的 unicode 中文乱码问题。
支持单文件或批量处理目录。
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def decode_unicode_json(json_str: Optional[Union[str, bytes]]) -> Optional[str]:
    """解码 JSON 字符串中的 unicode 字符。
    
    Args:
        json_str: 可能包含 unicode 编码的 JSON 字符串
        
    Returns:
        解码后的 JSON 字符串
    """
    if json_str is None or pd.isna(json_str):
        return json_str
    
    # 处理 bytes 类型
    if isinstance(json_str, bytes):
        json_str = json_str.decode('utf-8', errors='ignore')
    
    # 如果已经是字符串且不包含 unicode 转义序列，直接返回
    if isinstance(json_str, str) and '\\u' not in json_str:
        return json_str
    
    try:
        # JSON 加载（自动解码 unicode）
        json_obj = json.loads(json_str)
        
        # JSON dump 时关闭 ensure_ascii（保留中文）
        decoded_str = json.dumps(
            json_obj,
            ensure_ascii=False,  # 关键：不将中文转为 unicode
            indent=None,         # 保持原始紧凑格式
            separators=(',', ':')  # 保持原始分隔符格式
        )
        return decoded_str
    
    except json.JSONDecodeError:
        # JSON 解析失败时返回原始字符串
        return json_str
    except Exception as e:
        logger.debug(f"处理 JSON 字符串时发生错误: {e}")
        return json_str

def find_parquet_files(directory: str, recursive: bool = True) -> List[str]:
    """
    查找目录下的所有parquet文件
    
    Args:
        directory: 目录路径
        recursive: 是否递归查找子目录，默认True
    
    Returns:
        parquet文件路径列表
    """
    parquet_files = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")
    
    if not directory_path.is_dir():
        raise ValueError(f"路径不是目录: {directory}")
    
    pattern = "**/*.parquet" if recursive else "*.parquet"
    parquet_files = [str(p) for p in directory_path.glob(pattern) if p.is_file()]
    
    logger.info(f"在目录 {directory} 中找到 {len(parquet_files)} 个parquet文件")
    return sorted(parquet_files)

def get_output_path(input_path: str, output_base: str, input_base: Optional[str] = None) -> str:
    """
    根据输入路径和输出基础路径生成输出路径
    
    Args:
        input_path: 输入文件路径
        output_base: 输出基础路径（文件或目录）
        input_base: 输入基础路径（用于保持相对路径结构），如果为None则使用输入文件的目录
    
    Returns:
        输出文件路径
    """
    input_path_obj = Path(input_path)
    output_base_obj = Path(output_base)
    
    # 如果输出基础路径是文件，直接返回
    if output_base_obj.is_file() or (not output_base_obj.exists() and not output_base_obj.suffix == ''):
        return str(output_base_obj)
    
    # 如果输出基础路径是目录
    if input_base:
        # 保持相对路径结构
        input_base_obj = Path(input_base)
        try:
            relative_path = input_path_obj.relative_to(input_base_obj)
            output_path = output_base_obj / relative_path
        except ValueError:
            # 如果无法计算相对路径，使用文件名
            output_path = output_base_obj / input_path_obj.name
    else:
        # 使用输入文件的目录作为基础
        output_path = output_base_obj / input_path_obj.name
    
    return str(output_path)

def process_parquet_file(
    input_path: str,
    output_path: str,
    engine: str = 'pyarrow',
    fields: Optional[List[str]] = None
) -> None:
    """处理 parquet 文件，修复指定字段的 unicode 中文乱码。
    
    Args:
        input_path: 输入 parquet 文件路径
        output_path: 输出 parquet 文件路径
        engine: 读取/写入 parquet 的引擎，可选 'pyarrow' 或 'fastparquet'
        fields: 要处理的字段列表，默认为 ['messages', 'segments']
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    if fields is None:
        fields = ['messages', 'segments']
    
    # 读取 parquet 文件
    logger.info(f"读取文件: {input_path}")
    df = pd.read_parquet(input_path, engine=engine)
    logger.info(f"数据总行数: {len(df)}")
    
    # 检查并处理字段
    processed_fields = []
    for field in fields:
        if field in df.columns:
            logger.debug(f"处理字段: {field}")
            df[field] = df[field].apply(decode_unicode_json)
            processed_fields.append(field)
        else:
            logger.debug(f"字段不存在，跳过: {field}")
    
    if not processed_fields:
        logger.warning(f"没有找到任何需要处理的字段: {fields}")
        # 如果没有需要处理的字段，直接复制文件
        if input_path != output_path:
            import shutil
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, output_path)
            logger.info(f"文件已复制到: {output_path}")
        return
    
    logger.info(f"已处理字段: {processed_fields}")
    
    # 保存处理后的文件
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(
        output_path,
        engine=engine,
        index=False,
        compression='snappy'
    )
    logger.info(f"文件保存成功: {output_path}")

def process_directory(input_dir: str, output_dir: str, engine: str = 'pyarrow', recursive: bool = True, overwrite: bool = False) -> None:
    """
    批量处理目录下的所有parquet文件
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        engine: parquet处理引擎
        recursive: 是否递归处理子目录
        overwrite: 是否覆盖原文件（如果为True，output_dir将被忽略，直接覆盖输入文件）
    """
    # 查找所有parquet文件
    parquet_files = find_parquet_files(input_dir, recursive=recursive)
    
    if not parquet_files:
        logger.warning(f"在目录 {input_dir} 中未找到任何parquet文件")
        return
    
    # 创建输出目录（如果需要）
    if not overwrite:
        output_path_obj = Path(output_dir)
        output_path_obj.mkdir(parents=True, exist_ok=True)
        logger.info(f"输出目录: {output_dir}")
    
    # 处理每个文件
    total_files = len(parquet_files)
    success_count = 0
    fail_count = 0
    
    for input_file in tqdm(parquet_files, desc="处理文件"):
        try:
            if overwrite:
                # 覆盖原文件
                output_file = input_file
            else:
                # 生成输出路径，保持目录结构
                output_file = get_output_path(input_file, output_dir, input_dir)
                # 确保输出目录存在
                Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            
            process_parquet_file(input_file, output_file, engine)
            success_count += 1
            
        except Exception as e:
            fail_count += 1
            logger.error(f"文件处理失败: {input_file}, 错误: {e}", exc_info=True)
            continue
    
    # 输出统计信息
    logger.info(f"\n{'='*60}")
    logger.info(f"批量处理完成！")
    logger.info(f"总文件数: {total_files}")
    logger.info(f"成功: {success_count}")
    logger.info(f"失败: {fail_count}")
    logger.info(f"{'='*60}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='处理parquet文件中messages和segments字段的unicode中文乱码（支持单文件或批量处理目录）'
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='输入parquet文件路径或目录路径（必填）'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='输出parquet文件路径或目录路径（必填）'
    )
    parser.add_argument(
        '-e', '--engine',
        choices=['pyarrow', 'fastparquet'],
        default='pyarrow',
        help='parquet处理引擎，默认使用pyarrow'
    )
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='处理目录时，不递归处理子目录（仅处理当前目录下的文件）'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='覆盖原文件（仅当输入是目录时有效，会忽略输出路径）'
    )
    
    args = parser.parse_args()
    
    # 执行处理
    try:
        input_path = Path(args.input)
        
        if not input_path.exists():
            logger.error(f"输入路径不存在: {args.input}")
            exit(1)
        
        # 判断输入是文件还是目录
        if input_path.is_file():
            # 单文件处理模式
            logger.info("单文件处理模式")
            if Path(args.output).is_dir():
                # 如果输出是目录，在目录中创建同名文件
                output_file = Path(args.output) / input_path.name
            else:
                output_file = args.output
            
            process_parquet_file(
                input_path=str(input_path),
                output_path=str(output_file),
                engine=args.engine
            )
            logger.info("所有操作完成！")
            
        elif input_path.is_dir():
            # 目录批量处理模式
            logger.info("目录批量处理模式")
            if args.overwrite:
                logger.info("将覆盖原文件")
                process_directory(
                    input_dir=str(input_path),
                    output_dir="",  # 不会被使用
                    engine=args.engine,
                    recursive=not args.no_recursive,
                    overwrite=True
                )
            else:
                output_path = Path(args.output)
                if output_path.exists() and output_path.is_file():
                    logger.error(f"输入是目录时，输出也应该是目录，但输出路径是文件: {args.output}")
                    exit(1)
                process_directory(
                    input_dir=str(input_path),
                    output_dir=str(output_path),
                    engine=args.engine,
                    recursive=not args.no_recursive,
                    overwrite=False
                )
            logger.info("所有操作完成！")
        else:
            logger.error(f"输入路径既不是文件也不是目录: {args.input}")
            exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n操作被用户取消")
        sys.exit(1)
    except Exception as e:
        logger.error(f"程序执行失败: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
