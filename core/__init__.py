"""核心模块 - 缓存、工具类、配置、监控和 PatchChain"""

from .cache import TTLCache, MemoryUtils
from .utils import ModuleStats, rate, ChatVersionTracker
from .patch_chain import PatchChain, get_patch_chain
from .config import (
    ConfigFieldType,
    ConfigConstraint,
    ExtendedConfigField,
    ConfigVersion,
    ConfigMigrator,
    ConfigManager,
    get_config_manager,
    get_config,
    set_config,
)
from .monitor import (
    MemorySnapshot,
    PerformanceMetrics,
    MemoryMonitor,
    StatsReporter,
    PerformanceCollector,
    get_memory_monitor,
    get_stats_reporter,
    get_perf_collector,
)
from .module_config import (
    ModuleConfigMapper,
    ModuleEnabler,
    get_module_mapper,
    get_module_enabler,
)

__all__ = [
    # 缓存
    "TTLCache",
    "MemoryUtils",
    # 工具
    "ModuleStats",
    "rate",
    "ChatVersionTracker",
    # PatchChain
    "PatchChain",
    "get_patch_chain",
    # 配置
    "ConfigFieldType",
    "ConfigConstraint",
    "ExtendedConfigField",
    "ConfigVersion",
    "ConfigMigrator",
    "ConfigManager",
    "get_config_manager",
    "get_config",
    "set_config",
    # 监控
    "MemorySnapshot",
    "PerformanceMetrics",
    "MemoryMonitor",
    "StatsReporter",
    "PerformanceCollector",
    "get_memory_monitor",
    "get_stats_reporter",
    "get_perf_collector",
    # 模块配置
    "ModuleConfigMapper",
    "ModuleEnabler",
    "get_module_mapper",
    "get_module_enabler",
]
