"""核心模块 - 缓存、工具类、配置、监控、PatchChain、过期管理、通知和日志处理器"""

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
from .expiration import (
    RefreshDecision,
    ExpirationConfig,
    ExpirationState,
    ExpirationManager,
    DatabaseExpirationManager,
    FileExpirationManager,
    CompositeExpirationManager,
    create_default_expiration_config,
    create_expiration_manager,
)
from .notification import (
    NotificationConfig,
    NotificationManager,
    NotificationRecord,
    NOTIFICATION_TEMPLATES,
    get_notification_manager,
    init_notification_manager,
)
from .logging_handler import (
    ErrorLogConfig,
    NotificationLogHandler,
    get_log_handler,
    init_log_handler,
    shutdown_log_handler,
)
from .compat import (
    load_core_module,
    validate_file_path,
    validate_json_schema,
    safe_load_json_file,
    PARAGRAPH_HASH_SCHEMA,
    ENTITY_COUNT_SCHEMA,
    CoreModuleLoadError,
    PathTraversalError,
    JsonValidationError,
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
    # 过期管理
    "RefreshDecision",
    "ExpirationConfig",
    "ExpirationState",
    "ExpirationManager",
    "DatabaseExpirationManager",
    "FileExpirationManager",
    "CompositeExpirationManager",
    "create_default_expiration_config",
    "create_expiration_manager",
    # 通知
    "NotificationConfig",
    "NotificationManager",
    "NotificationRecord",
    "NOTIFICATION_TEMPLATES",
    "get_notification_manager",
    "init_notification_manager",
    # 日志处理器
    "ErrorLogConfig",
    "NotificationLogHandler",
    "get_log_handler",
    "init_log_handler",
    "shutdown_log_handler",
    # 兼容性模块
    "load_core_module",
    "validate_file_path",
    "validate_json_schema",
    "safe_load_json_file",
    "PARAGRAPH_HASH_SCHEMA",
    "ENTITY_COUNT_SCHEMA",
    "CoreModuleLoadError",
    "PathTraversalError",
    "JsonValidationError",
]
