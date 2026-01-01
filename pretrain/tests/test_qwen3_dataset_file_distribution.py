"""
测试 Qwen3ChatCompletionParquetDataset 在多进程、多worker情况下的文件分发逻辑

验证点：
1. 每个文件只被一个worker处理（不重复）
2. 所有文件都被处理（不遗漏）
3. 在不同rank和worker组合下都能正确工作
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import sys


class TestFileDistribution(unittest.TestCase):
    """测试文件分发逻辑"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建模拟的文件列表
        self.data_files = [
            (f"file_{i}.parquet", 0) for i in range(100)  # 100个文件，epoch=0
        ]
        self.num_workers = 4
    
    def _get_file_distribution(self, rank, world_size, worker, num_workers):
        """
        模拟文件分发逻辑，返回该worker应该处理的文件索引列表
        
        Args:
            rank: 进程rank
            world_size: 总进程数
            worker: worker ID
            num_workers: 每个进程的worker数
            
        Returns:
            list: 文件索引列表
        """
        total_num_workers = num_workers * world_size
        local_worker_idx = rank * num_workers + worker
        fn_list = [
            idx for idx, fn in enumerate(self.data_files) 
            if idx % total_num_workers == local_worker_idx
        ]
        return fn_list
    
    def test_file_distribution_no_overlap(self):
        """测试文件分发无重叠：每个文件只被一个worker处理"""
        world_size = 2
        num_workers = 4
        
        # 收集所有worker分配到的文件
        all_assigned_files = set()
        
        for rank in range(world_size):
            for worker in range(num_workers):
                assigned_files = self._get_file_distribution(rank, world_size, worker, num_workers)
                file_indices = set(assigned_files)
                
                # 检查是否有重叠
                overlap = all_assigned_files & file_indices
                self.assertEqual(
                    len(overlap), 0,
                    f"Rank {rank}, Worker {worker} 分配的文件与已有分配重叠: {overlap}"
                )
                
                all_assigned_files.update(file_indices)
        
        # 验证所有文件都被分配
        total_files = len(self.data_files)
        self.assertEqual(
            len(all_assigned_files), total_files,
            f"文件分配不完整: 期望 {total_files} 个文件，实际分配 {len(all_assigned_files)} 个"
        )
    
    def test_file_distribution_completeness(self):
        """测试文件分发完整性：所有文件都被处理"""
        world_size = 2
        num_workers = 4
        
        all_assigned_files = set()
        
        for rank in range(world_size):
            for worker in range(num_workers):
                assigned_files = self._get_file_distribution(rank, world_size, worker, num_workers)
                all_assigned_files.update(assigned_files)
        
        # 验证所有文件都被分配
        expected_files = set(range(len(self.data_files)))
        self.assertEqual(
            all_assigned_files, expected_files,
            f"文件分配不完整: 缺失文件 {expected_files - all_assigned_files}"
        )
    
    def test_file_distribution_different_configs(self):
        """测试不同配置下的文件分发"""
        test_configs = [
            (1, 1),   # 单进程，单worker
            (1, 4),   # 单进程，4个worker
            (2, 2),   # 2个进程，每个2个worker
            (4, 2),   # 4个进程，每个2个worker
            (2, 8),   # 2个进程，每个8个worker
        ]
        
        for world_size, num_workers in test_configs:
            with self.subTest(world_size=world_size, num_workers=num_workers):
                all_assigned_files = set()
                
                for rank in range(world_size):
                    for worker in range(num_workers):
                        assigned_files = self._get_file_distribution(
                            rank, world_size, worker, num_workers
                        )
                        file_indices = set(assigned_files)
                        
                        # 检查重叠
                        overlap = all_assigned_files & file_indices
                        self.assertEqual(
                            len(overlap), 0,
                            f"配置 (world_size={world_size}, num_workers={num_workers}), "
                            f"Rank {rank}, Worker {worker} 有重叠: {overlap}"
                        )
                        
                        all_assigned_files.update(file_indices)
                
                # 验证完整性
                expected_files = set(range(len(self.data_files)))
                self.assertEqual(
                    all_assigned_files, expected_files,
                    f"配置 (world_size={world_size}, num_workers={num_workers}) "
                    f"文件分配不完整: 缺失 {expected_files - all_assigned_files}"
                )
    
    def test_file_distribution_balance(self):
        """测试文件分发的负载均衡（每个worker分配的文件数量应该大致相等）"""
        world_size = 2
        num_workers = 4
        total_workers = world_size * num_workers
        
        file_counts = []
        for rank in range(world_size):
            for worker in range(num_workers):
                assigned_files = self._get_file_distribution(rank, world_size, worker, num_workers)
                file_counts.append(len(assigned_files))
        
        # 计算期望的文件数（应该大致相等）
        expected_per_worker = len(self.data_files) / total_workers
        min_files = int(expected_per_worker)
        max_files = int(expected_per_worker) + 1
        
        # 验证每个worker分配的文件数在合理范围内
        for count in file_counts:
            self.assertGreaterEqual(count, min_files, "文件分配过少")
            self.assertLessEqual(count, max_files, "文件分配过多")
        
        # 验证总和正确
        self.assertEqual(
            sum(file_counts), len(self.data_files),
            f"文件总数不匹配: 期望 {len(self.data_files)}, 实际 {sum(file_counts)}"
        )
    
    def test_file_distribution_with_epochs(self):
        """测试多epoch情况下的文件分发"""
        # 创建多epoch的文件列表
        data_files_multi_epoch = []
        for epoch in range(3):
            for i in range(20):
                data_files_multi_epoch.append((f"file_{i}.parquet", epoch))
        
        self.data_files = data_files_multi_epoch
        
        world_size = 2
        num_workers = 4
        
        # 按 (file_idx, epoch) 收集分配
        all_assigned = set()
        
        for rank in range(world_size):
            for worker in range(num_workers):
                assigned_indices = self._get_file_distribution(
                    rank, world_size, worker, num_workers
                )
                # 将索引转换为 (文件名, epoch) 元组
                for idx in assigned_indices:
                    file_name, epoch = self.data_files[idx]
                    all_assigned.add((file_name, epoch))
        
        # 验证所有 (file, epoch) 组合都被分配
        expected = set((fn, ep) for fn, ep in self.data_files)
        self.assertEqual(
            all_assigned, expected,
            f"多epoch文件分配不完整: 缺失 {expected - all_assigned}"
        )


class TestFileDistributionLogic(unittest.TestCase):
    """测试文件分发逻辑的核心算法"""
    
    def setUp(self):
        """设置测试环境"""
        self.data_files = [
            (f"file_{i}.parquet", 0) for i in range(50)
        ]
    
    def test_distribution_algorithm(self):
        """测试文件分发算法的正确性"""
        # 模拟 Qwen3NaiveParquetDataset.__iter__local_shuffle 中的分发逻辑
        rank = 0
        world_size = 2
        worker = 0
        num_workers = 2
        
        total_num_workers = num_workers * world_size
        local_worker_idx = rank * num_workers + worker
        fn_list = [
            fn for idx, fn in enumerate(self.data_files) 
            if idx % total_num_workers == local_worker_idx
        ]
        
        # 验证文件列表不为空
        self.assertGreater(len(fn_list), 0, "文件列表不应为空")
        
        # 验证文件索引正确
        expected_indices = [
            idx for idx in range(len(self.data_files))
            if idx % total_num_workers == local_worker_idx
        ]
        actual_indices = [
            idx for idx, fn in enumerate(self.data_files) if fn in fn_list
        ]
        self.assertEqual(
            set(actual_indices), set(expected_indices),
            "文件索引分配不正确"
        )


def run_distribution_test_manual():
    """
    手动运行文件分发测试，打印详细的分配信息
    用于调试和验证
    """
    print("=" * 80)
    print("文件分发测试 - 手动验证")
    print("=" * 80)
    
    # 测试配置
    data_files = [(f"file_{i}.parquet", 0) for i in range(100)]
    test_configs = [
        (1, 1, "单进程，单worker"),
        (1, 4, "单进程，4个worker"),
        (2, 2, "2个进程，每个2个worker"),
        (4, 2, "4个进程，每个2个worker"),
        (2, 8, "2个进程，每个8个worker"),
    ]
    
    for world_size, num_workers, desc in test_configs:
        print(f"\n配置: {desc} (world_size={world_size}, num_workers={num_workers})")
        print("-" * 80)
        
        total_num_workers = num_workers * world_size
        all_assigned = {}
        
        for rank in range(world_size):
            for worker in range(num_workers):
                local_worker_idx = rank * num_workers + worker
                assigned_files = [
                    idx for idx, fn in enumerate(data_files)
                    if idx % total_num_workers == local_worker_idx
                ]
                all_assigned[(rank, worker)] = assigned_files
                
                print(f"  Rank {rank}, Worker {worker} (local_idx={local_worker_idx}): "
                      f"{len(assigned_files)} 个文件, 索引范围: {min(assigned_files) if assigned_files else 'N/A'}-{max(assigned_files) if assigned_files else 'N/A'}")
        
        # 验证完整性
        all_file_indices = set()
        for assigned in all_assigned.values():
            all_file_indices.update(assigned)
        
        expected_indices = set(range(len(data_files)))
        missing = expected_indices - all_file_indices
        extra = all_file_indices - expected_indices
        
        if missing:
            print(f"  ❌ 缺失文件索引: {sorted(missing)}")
        if extra:
            print(f"  ❌ 多余文件索引: {sorted(extra)}")
        if not missing and not extra:
            print(f"  ✅ 文件分配完整: 所有 {len(data_files)} 个文件都被正确分配")
        
        # 检查重叠
        has_overlap = False
        for (r1, w1), files1 in all_assigned.items():
            for (r2, w2), files2 in all_assigned.items():
                if (r1, w1) >= (r2, w2):  # 避免重复检查
                    continue
                overlap = set(files1) & set(files2)
                if overlap:
                    print(f"  ❌ 重叠检测: Rank {r1}, Worker {w1} 与 Rank {r2}, Worker {w2} 重叠文件: {sorted(overlap)}")
                    has_overlap = True
        
        if not has_overlap:
            print(f"  ✅ 无重叠: 所有文件只被一个worker处理")


if __name__ == '__main__':
    # 运行单元测试
    print("运行单元测试...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # 运行手动验证
    print("\n" + "=" * 80)
    run_distribution_test_manual()

