"""
Task table and utility functions for Benchmark
"""

from typing import List, Optional, Tuple
from benchmark.tasks.v1_0.registry import TaskTable as TaskTable_v1_0

LATEST_BENCHMARK_VERSION = "v1.0"


BenchmarkTable = {
    "v1.0": TaskTable_v1_0,
}


def get_available_benchmark_versions() -> List[str]:
    """Get all available benchmark versions"""
    return sorted(list(BenchmarkTable.keys()))


def get_available_task_types(benchmark_version: str = LATEST_BENCHMARK_VERSION) -> List[str]:
    """Get all task types for the specified version"""
    task_table = BenchmarkTable[benchmark_version]
    return sorted(list(task_table.keys()))


def get_available_domains(benchmark_version: str = LATEST_BENCHMARK_VERSION) -> List[str]:
    """Get all domains for the specified version"""
    domains = set()
    for task_table in BenchmarkTable[benchmark_version].values():
        for domain in task_table.keys():
            domains.add(domain)
    return sorted(list(domains))


def get_available_languages(benchmark_version: str = LATEST_BENCHMARK_VERSION) -> List[str]:
    """Get all languages for the specified version"""
    languages = set()
    for task_table in BenchmarkTable[benchmark_version].values():
        for task in task_table.values():
            for lang in task.keys():
                languages.add(lang)
    return sorted(list(languages))


def check_benchmark_version(benchmark_version: Optional[str]) -> str:
    """
    Validate if benchmark version is valid
    
    Args:
        benchmark_version: Version to validate, returns latest version if None
        
    Returns:
        str: Valid benchmark version
        
    Raises:
        ValueError: If version is invalid
    """
    if benchmark_version is None:
        benchmark_version = LATEST_BENCHMARK_VERSION
    else:
        available_benchmark_versions = get_available_benchmark_versions()

        if benchmark_version not in available_benchmark_versions:
            raise ValueError(
                f"Invalid benchmark version: {benchmark_version}. Available versions: {', '.join(available_benchmark_versions)}"
            )

    return benchmark_version


def check_task_types(
    task_types: Optional[List[str]],
    benchmark_version: str = LATEST_BENCHMARK_VERSION,
) -> List[str]:
    """
    Validate if task types are valid
    
    Args:
        task_types: List of task types to validate, returns all task types if None
        benchmark_version: Benchmark version
        
    Returns:
        List[str]: Valid task types list
        
    Raises:
        ValueError: If task type is invalid
    """
    available_task_types = get_available_task_types(benchmark_version)
    if task_types is None:
        task_types = available_task_types
    else:
        if isinstance(task_types, str):
            task_types = [task_types]
        task_types = sorted(list(set(task_types)))
        task_types = [task_type.lower() for task_type in task_types]
        for task_type in task_types:
            if task_type not in available_task_types:
                raise ValueError(
                    f"{benchmark_version} | Invalid task type: {task_type}. Available task types: {', '.join(available_task_types)}"
                )
    return task_types


def check_domains(
    domains: Optional[List[str]],
    benchmark_version: str = LATEST_BENCHMARK_VERSION,
) -> List[str]:
    """
    Validate if domains are valid
    
    Args:
        domains: List of domains to validate, returns all domains if None
        benchmark_version: Benchmark version
        
    Returns:
        List[str]: Valid domains list
        
    Raises:
        ValueError: If domain is invalid
    """
    available_domains = get_available_domains(benchmark_version)
    if domains is None:
        domains = available_domains
    else:
        if isinstance(domains, str):
            domains = [domains]
        domains = sorted(list(set(domains)))
        domains = [domain.lower() for domain in domains]
        for domain in domains:
            if domain not in available_domains:
                raise ValueError(
                    f"{benchmark_version} | Invalid domain: {domain}. Available domains: {', '.join(available_domains)}"
                )
    return domains


def check_languages(
    languages: Optional[List[str]],
    benchmark_version: str = LATEST_BENCHMARK_VERSION,
) -> List[str]:
    """
    Validate if languages are valid
    
    Args:
        languages: List of languages to validate, returns all languages if None
        benchmark_version: Benchmark version
        
    Returns:
        List[str]: Valid languages list
        
    Raises:
        ValueError: If language is invalid
    """
    available_languages = get_available_languages(benchmark_version)
    if languages is None:
        languages = available_languages
    else:
        if isinstance(languages, str):
            languages = [languages]
        languages = sorted(list(set(languages)))
        languages = [language.lower() for language in languages]
        for language in languages:
            if language not in available_languages:
                raise ValueError(
                    f"{benchmark_version} | Invalid language: {language}. Available languages: {', '.join(available_languages)}"
                )
    return languages


def check_splits(
    splits: Optional[List[str]],
    benchmark_version: str = LATEST_BENCHMARK_VERSION,
) -> List[str]:
    """
    Validate if dataset splits are valid
    
    Args:
        splits: List of splits to validate, returns all splits if None
        benchmark_version: Benchmark version
        
    Returns:
        List[str]: Valid splits list
        
    Raises:
        ValueError: If split is invalid
    """
    # Only allow test split
    available_splits = ["test"]
    
    if splits is None:
        splits = available_splits
    else:
        if isinstance(splits, str):
            splits = [splits]
        splits = sorted(list(set(splits)))
        splits = [split.lower() for split in splits]
    
    for split in splits:
        if split not in available_splits:
            raise ValueError(
                f"{benchmark_version} | Invalid split: {split}. Available splits: {', '.join(available_splits)}"
            )
    return splits


def get_dataset_name_list(
    benchmark_version: str, task_type: str, domain: str, language: str
) -> Tuple[bool, List[str]]:
    """
    Get dataset name list for specified task
    
    Args:
        benchmark_version: Benchmark version
        task_type: Task type
        domain: Domain
        language: Language
        
    Returns:
        Tuple[bool, List[str]]: (found or not, dataset name list)
    """
    if benchmark_version not in BenchmarkTable:
        return False, None
    task_table = BenchmarkTable[benchmark_version]
    if task_type not in task_table:
        return False, None
    if domain not in task_table[task_type]:
        return False, None
    if language not in task_table[task_type][domain]:
        return False, None
    return True, list(task_table[task_type][domain][language].keys())


def get_task_splits(
    benchmark_version: str, task_type: str, domain: str, language: str, dataset_name: str
) -> List[str]:
    """
    Get available splits for specified task and dataset
    
    Args:
        benchmark_version: Benchmark version
        task_type: Task type
        domain: Domain
        language: Language
        dataset_name: Dataset name
        
    Returns:
        List[str]: Available splits list
    """
    if benchmark_version not in BenchmarkTable:
        return []
    task_table = BenchmarkTable[benchmark_version]
    if task_type not in task_table:
        return []
    if domain not in task_table[task_type]:
        return []
    if language not in task_table[task_type][domain]:
        return []
    if dataset_name not in task_table[task_type][domain][language]:
        return []
    return task_table[task_type][domain][language][dataset_name]['splits']