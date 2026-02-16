"""
核心配置模块 - 配置类型、验证器和加载器

提供完整的配置系统支持：
- 扩展 ConfigField 支持 float、嵌套对象等类型
- 配置验证机制（数值范围、依赖关系）
- 配置热更新支持
- 向后兼容性和配置迁移
"""

from __future__ import annotations

import copy
import json
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        tomllib = None  # type: ignore

try:
    from src.common.logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger("CM_perf_opt")

T = TypeVar("T")


class ConfigFieldType(Enum):
    """配置字段类型枚举"""

    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    STR = "str"
    LIST = "list"
    DICT = "dict"
    NESTED = "nested"  # 嵌套对象


@dataclass
class ConfigConstraint:
    """配置约束条件"""

    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    choices: Optional[List[Any]] = None
    pattern: Optional[str] = None  # 正则表达式模式
    required: bool = False
    depends_on: Optional[str] = None  # 依赖的其他配置项
    depends_value: Optional[Any] = None  # 依赖项需要的值
    validator: Optional[Callable[[Any], Tuple[bool, str]]] = None  # 自定义验证器


@dataclass
class ExtendedConfigField:
    """扩展的配置字段定义

    支持更多类型和验证功能：
    - float 类型
    - 嵌套对象
    - 数值范围约束
    - 依赖关系
    - 热更新标记
    """

    field_type: ConfigFieldType
    default: Any
    description: str = ""
    constraint: Optional[ConfigConstraint] = None
    hot_reload: bool = False  # 是否支持热更新
    section: str = "plugin"  # 所属配置节
    order: int = 0  # 显示顺序
    hidden: bool = False  # 是否隐藏（高级配置）
    deprecated: bool = False  # 是否已弃用
    deprecated_message: str = ""  # 弃用提示信息
    nested_schema: Optional[Dict[str, "ExtendedConfigField"]] = None  # 嵌套对象的 schema

    def validate(self, value: Any, full_config: Optional[Dict] = None) -> Tuple[bool, str]:
        """验证配置值

        Args:
            value: 要验证的值
            full_config: 完整配置（用于依赖检查）

        Returns:
            Tuple[bool, str]: (是否有效, 错误信息)
        """
        # 类型检查
        type_valid, type_error = self._validate_type(value)
        if not type_valid:
            return False, type_error

        # 约束检查
        if self.constraint:
            constraint_valid, constraint_error = self._validate_constraint(value, full_config)
            if not constraint_valid:
                return False, constraint_error

        # 嵌套对象验证
        if self.field_type == ConfigFieldType.NESTED and self.nested_schema:
            nested_valid, nested_error = self._validate_nested(value)
            if not nested_valid:
                return False, nested_error

        return True, ""

    def _validate_type(self, value: Any) -> Tuple[bool, str]:
        """验证类型"""
        expected_types = {
            ConfigFieldType.BOOL: bool,
            ConfigFieldType.INT: int,
            ConfigFieldType.FLOAT: (int, float),
            ConfigFieldType.STR: str,
            ConfigFieldType.LIST: list,
            ConfigFieldType.DICT: dict,
            ConfigFieldType.NESTED: dict,
        }

        expected = expected_types.get(self.field_type)
        if expected is None:
            return True, ""

        if not isinstance(value, expected):
            return False, f"期望类型 {self.field_type.value}，实际类型 {type(value).__name__}"

        return True, ""

    def _validate_constraint(
        self, value: Any, full_config: Optional[Dict] = None
    ) -> Tuple[bool, str]:
        """验证约束条件"""
        c = self.constraint
        if c is None:
            return True, ""

        # 依赖检查
        if c.depends_on and full_config:
            depends_value = self._get_nested_value(full_config, c.depends_on)
            if c.depends_value is not None and depends_value != c.depends_value:
                return True, ""  # 依赖条件不满足时跳过验证

        # 数值范围
        if c.min_value is not None and value < c.min_value:
            return False, f"值 {value} 小于最小值 {c.min_value}"
        if c.max_value is not None and value > c.max_value:
            return False, f"值 {value} 大于最大值 {c.max_value}"

        # 选项列表
        if c.choices is not None and value not in c.choices:
            return False, f"值 {value} 不在允许的选项 {c.choices} 中"

        # 正则模式
        if c.pattern and isinstance(value, str):
            import re

            if not re.match(c.pattern, value):
                return False, f"值 {value} 不匹配模式 {c.pattern}"

        # 自定义验证器
        if c.validator:
            return c.validator(value)

        return True, ""

    def _validate_nested(self, value: Dict) -> Tuple[bool, str]:
        """验证嵌套对象"""
        if not self.nested_schema:
            return True, ""

        for key, field_def in self.nested_schema.items():
            if key in value:
                valid, error = field_def.validate(value[key])
                if not valid:
                    return False, f"{key}: {error}"
            elif field_def.constraint and field_def.constraint.required:
                return False, f"缺少必需字段 {key}"

        return True, ""

    @staticmethod
    def _get_nested_value(config: Dict, path: str) -> Any:
        """获取嵌套配置值"""
        parts = path.split(".")
        current = config
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current


class ConfigVersion:
    """配置版本管理"""

    CURRENT_VERSION = "5.2.0"
    MIN_COMPATIBLE_VERSION = "1.0.0"

    @staticmethod
    def parse_version(version_str: str) -> Tuple[int, int, int]:
        """解析版本号"""
        try:
            parts = version_str.split(".")
            return (int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0)
        except (ValueError, IndexError):
            return (1, 0, 0)

    @staticmethod
    def compare_versions(v1: str, v2: str) -> int:
        """比较版本号 (-1: v1 < v2, 0: v1 == v2, 1: v1 > v2)"""
        p1 = ConfigVersion.parse_version(v1)
        p2 = ConfigVersion.parse_version(v2)
        if p1 < p2:
            return -1
        if p1 > p2:
            return 1
        return 0

    @staticmethod
    def is_compatible(version_str: str) -> bool:
        """检查版本是否兼容"""
        return ConfigVersion.compare_versions(version_str, ConfigVersion.MIN_COMPATIBLE_VERSION) >= 0


class ConfigMigrator:
    """配置迁移器 - 处理旧版本配置的升级"""

    def __init__(self):
        self._migrations: Dict[str, Callable[[Dict], Dict]] = {}
        self._register_migrations()

    def _register_migrations(self):
        """注册迁移函数"""
        # 从 1.0.0 迁移到 2.0.0
        self._migrations["1.0.0->2.0.0"] = self._migrate_1_0_to_2_0

    def _migrate_1_0_to_2_0(self, config: Dict) -> Dict:
        """从 1.0.0 迁移到 2.0.0"""
        new_config = copy.deepcopy(config)

        # 迁移旧的缓存大小配置到模块配置
        if "cache" in new_config:
            cache = new_config["cache"]

            # 创建模块配置节（如果不存在）
            if "modules" not in new_config:
                new_config["modules"] = {}

            modules = new_config["modules"]

            # message_cache 迁移
            if "message_cache_size" in cache:
                if "message_cache" not in modules:
                    modules["message_cache"] = {}
                modules["message_cache"]["max_size"] = cache.pop("message_cache_size")

            # person_cache 迁移
            if "person_cache_size" in cache:
                if "person_cache" not in modules:
                    modules["person_cache"] = {}
                modules["person_cache"]["max_size"] = cache.pop("person_cache_size")

            # expression_cache 迁移
            if "expression_cache_size" in cache:
                if "expression_cache" not in modules:
                    modules["expression_cache"] = {}
                modules["expression_cache"]["batch_size"] = cache.pop("expression_cache_size")

            # jargon_cache 迁移
            if "jargon_cache_size" in cache:
                if "jargon_cache" not in modules:
                    modules["jargon_cache"] = {}
                modules["jargon_cache"]["batch_size"] = cache.pop("jargon_cache_size")

            # kg_cache 迁移
            if "kg_cache_size" in cache:
                if "kg_cache" not in modules:
                    modules["kg_cache"] = {}
                modules["kg_cache"]["batch_size"] = cache.pop("kg_cache_size")

            # cache_ttl 迁移到各模块
            if "cache_ttl" in cache:
                ttl = cache.pop("cache_ttl")
                for module_name in ["message_cache", "person_cache"]:
                    if module_name not in modules:
                        modules[module_name] = {}
                    modules[module_name]["ttl"] = ttl

        # 更新版本号
        if "plugin" not in new_config:
            new_config["plugin"] = {}
        new_config["plugin"]["config_version"] = "2.0.0"

        return new_config

    def migrate(self, config: Dict) -> Dict:
        """执行配置迁移"""
        current_version = config.get("plugin", {}).get("config_version", "1.0.0")

        if ConfigVersion.compare_versions(current_version, ConfigVersion.CURRENT_VERSION) >= 0:
            return config  # 无需迁移

        result = copy.deepcopy(config)

        # 链式迁移
        version_chain = ["1.0.0", "2.0.0"]
        start_idx = -1

        for i, v in enumerate(version_chain):
            if ConfigVersion.compare_versions(current_version, v) <= 0:
                start_idx = i
                break

        if start_idx < 0:
            return result

        for i in range(start_idx, len(version_chain) - 1):
            migration_key = f"{version_chain[i]}->{version_chain[i + 1]}"
            if migration_key in self._migrations:
                logger.info(f"[Config] 执行配置迁移: {migration_key}")
                result = self._migrations[migration_key](result)

        return result


class ConfigManager:
    """配置管理器 - 核心配置系统

    功能：
    - 配置加载和保存
    - 配置验证
    - 热更新支持
    - 向后兼容性
    - 配置变更通知
    """

    _instance: Optional["ConfigManager"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, plugin_dir: Optional[Path] = None):
        if self._initialized:
            return
        self._initialized = True

        self._plugin_dir = plugin_dir or Path(__file__).parent.parent
        self._config_path = self._plugin_dir / "config.toml"
        self._config: Dict[str, Any] = {}
        self._schema: Dict[str, Dict[str, ExtendedConfigField]] = {}
        self._listeners: Dict[str, List[Callable[[str, Any, Any], None]]] = {}
        self._migrator = ConfigMigrator()
        self._config_lock = threading.RLock()

        self._build_schema()

    def _build_schema(self):
        """构建完整的配置 schema - 与config.toml结构匹配"""
        self._schema = {
            "plugin": self._build_plugin_schema(),
            "modules": self._build_modules_schema(),
            "message_cache": self._build_message_cache_schema(),
            "person_cache": self._build_person_cache_schema(),
            "expression_cache": self._build_expression_cache_schema(),
            "jargon_cache": self._build_jargon_cache_schema(),
            "kg_cache": self._build_kg_cache_schema(),
            "db_tuning": self._build_db_tuning_schema(),
            "lightweight_profiler": self._build_lightweight_profiler_schema(),
            "advanced": self._build_advanced_schema(),
            "monitoring": self._build_monitoring_schema(),
        }

    def _build_plugin_schema(self) -> Dict[str, ExtendedConfigField]:
        """插件基础配置 schema"""
        return {
            "enabled": ExtendedConfigField(
                field_type=ConfigFieldType.BOOL,
                default=True,
                description="是否启用插件",
                section="plugin",
                order=0,
            ),
            "config_version": ExtendedConfigField(
                field_type=ConfigFieldType.STR,
                default="2.0.0",
                description="配置文件版本",
                section="plugin",
                order=1,
                hidden=True,
            ),
            "log_level": ExtendedConfigField(
                field_type=ConfigFieldType.STR,
                default="INFO",
                description="日志级别",
                constraint=ConfigConstraint(
                    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                ),
                section="plugin",
                order=2,
                hot_reload=True,
            ),
            "degraded_mode": ExtendedConfigField(
                field_type=ConfigFieldType.BOOL,
                default=False,
                description="降级模式（仅在出错时自动启用）",
                section="plugin",
                order=3,
                hidden=True,
            ),
        }

    def _build_modules_schema(self) -> Dict[str, ExtendedConfigField]:
        """模块开关配置 schema - 只包含扁平化开关"""
        return {
            "message_cache_enabled": ExtendedConfigField(
                field_type=ConfigFieldType.BOOL,
                default=True,
                description="是否启用消息缓存",
                section="modules",
                order=0,
            ),
            "message_repository_fastpath_enabled": ExtendedConfigField(
                field_type=ConfigFieldType.BOOL,
                default=True,
                description="是否启用message_repository快速路径",
                section="modules",
                order=1,
            ),
            "person_cache_enabled": ExtendedConfigField(
                field_type=ConfigFieldType.BOOL,
                default=True,
                description="是否启用人物信息缓存",
                section="modules",
                order=2,
            ),
            "expression_cache_enabled": ExtendedConfigField(
                field_type=ConfigFieldType.BOOL,
                default=True,
                description="是否启用表达式缓存",
                section="modules",
                order=3,
            ),
            "jargon_cache_enabled": ExtendedConfigField(
                field_type=ConfigFieldType.BOOL,
                default=True,
                description="是否启用黑话缓存",
                section="modules",
                order=4,
            ),
            "jargon_matcher_automaton_enabled": ExtendedConfigField(
                field_type=ConfigFieldType.BOOL,
                default=True,
                description="是否启用黑话匹配自动机加速",
                section="modules",
                order=5,
            ),
            "kg_cache_enabled": ExtendedConfigField(
                field_type=ConfigFieldType.BOOL,
                default=True,
                description="是否启用知识图谱缓存",
                section="modules",
                order=6,
            ),
            "levenshtein_fast_enabled": ExtendedConfigField(
                field_type=ConfigFieldType.BOOL,
                default=True,
                description="是否启用Levenshtein距离加速",
                section="modules",
                order=7,
            ),
            "image_desc_bulk_lookup_enabled": ExtendedConfigField(
                field_type=ConfigFieldType.BOOL,
                default=True,
                description="是否启用图片描述批量替换",
                section="modules",
                order=8,
            ),
            "user_reference_batch_resolve_enabled": ExtendedConfigField(
                field_type=ConfigFieldType.BOOL,
                default=True,
                description="是否启用用户引用批量解析",
                section="modules",
                order=9,
            ),
            "regex_precompile_enabled": ExtendedConfigField(
                field_type=ConfigFieldType.BOOL,
                default=True,
                description="是否启用正则预编译",
                section="modules",
                order=10,
            ),
            "typo_generator_cache_enabled": ExtendedConfigField(
                field_type=ConfigFieldType.BOOL,
                default=True,
                description="是否启用typo_generator缓存",
                section="modules",
                order=11,
            ),
            "db_tuning_enabled": ExtendedConfigField(
                field_type=ConfigFieldType.BOOL,
                default=True,
                description="是否启用SQLite数据库调优",
                section="modules",
                order=12,
            ),
            "lightweight_profiler_enabled": ExtendedConfigField(
                field_type=ConfigFieldType.BOOL,
                default=False,
                description="是否启用轻量性能剖析",
                section="modules",
                order=13,
            ),
            "asyncio_loop_pool_enabled": ExtendedConfigField(
                field_type=ConfigFieldType.BOOL,
                default=True,
                description="是否启用asyncio_loop_pool",
                section="modules",
                order=14,
            ),
        }

    def _build_message_cache_schema(self) -> Dict[str, ExtendedConfigField]:
        """消息缓存详细配置"""
        return {
            "per_chat_limit": ExtendedConfigField(
                field_type=ConfigFieldType.INT,
                default=200,
                description="每个聊天的缓存消息数量",
                constraint=ConfigConstraint(min_value=50, max_value=1000),
                hot_reload=True,
            ),
            "ttl": ExtendedConfigField(
                field_type=ConfigFieldType.INT,
                default=300,
                description="缓存过期时间（秒）",
                constraint=ConfigConstraint(min_value=60, max_value=3600),
                hot_reload=True,
            ),
            "max_chats": ExtendedConfigField(
                field_type=ConfigFieldType.INT,
                default=500,
                description="最大缓存聊天数",
                constraint=ConfigConstraint(min_value=100, max_value=2000),
                hot_reload=True,
            ),
            "ignore_time_limit_when_active": ExtendedConfigField(
                field_type=ConfigFieldType.BOOL,
                default=True,
                description="活跃聊天流忽略TTL限制",
                hot_reload=True,
            ),
            "active_time_window": ExtendedConfigField(
                field_type=ConfigFieldType.INT,
                default=300,
                description="活跃时间窗口（秒）",
                constraint=ConfigConstraint(min_value=60, max_value=1800),
                hot_reload=True,
            ),
            "normalize_lt_window_seconds": ExtendedConfigField(
                field_type=ConfigFieldType.FLOAT,
                default=300.0,
                description="时间戳归一化窗口（秒）",
                constraint=ConfigConstraint(min_value=60.0, max_value=600.0),
            ),
            "bucket_enabled": ExtendedConfigField(
                field_type=ConfigFieldType.BOOL,
                default=False,
                description="滑动窗口分桶功能（预留）",
            ),
            "bucket_seconds": ExtendedConfigField(
                field_type=ConfigFieldType.INT,
                default=5,
                description="分桶时间间隔（秒）",
            ),
            "mode": ExtendedConfigField(
                field_type=ConfigFieldType.STR,
                default="query",
                description="缓存模式: query 或 full",
                constraint=ConfigConstraint(choices=["query", "full"]),
            ),
        }

    def _build_person_cache_schema(self) -> Dict[str, ExtendedConfigField]:
        """人物信息缓存详细配置"""
        return {
            "max_size": ExtendedConfigField(
                field_type=ConfigFieldType.INT,
                default=3000,
                description="最大缓存条目数",
                constraint=ConfigConstraint(min_value=500, max_value=10000),
                hot_reload=True,
            ),
            "ttl": ExtendedConfigField(
                field_type=ConfigFieldType.INT,
                default=1800,
                description="缓存过期时间（秒）",
                constraint=ConfigConstraint(min_value=300, max_value=7200),
                hot_reload=True,
            ),
            "warmup_enabled": ExtendedConfigField(
                field_type=ConfigFieldType.BOOL,
                default=True,
                description="启用预热功能",
                hot_reload=True,
            ),
            "warmup_per_chat_sample": ExtendedConfigField(
                field_type=ConfigFieldType.INT,
                default=30,
                description="预热时每聊天采样消息数",
                constraint=ConfigConstraint(min_value=10, max_value=100),
                hot_reload=True,
            ),
            "warmup_max_persons": ExtendedConfigField(
                field_type=ConfigFieldType.INT,
                default=20,
                description="每聊天最多预热人数",
                constraint=ConfigConstraint(min_value=5, max_value=50),
                hot_reload=True,
            ),
            "warmup_ttl": ExtendedConfigField(
                field_type=ConfigFieldType.INT,
                default=120,
                description="预热记录过期时间（秒）",
                constraint=ConfigConstraint(min_value=30, max_value=600),
                hot_reload=True,
            ),
            "warmup_debounce_seconds": ExtendedConfigField(
                field_type=ConfigFieldType.FLOAT,
                default=3.0,
                description="预热防抖时间（秒）",
                constraint=ConfigConstraint(min_value=0.5, max_value=10.0),
                hot_reload=True,
            ),
        }

    def _build_expression_cache_schema(self) -> Dict[str, ExtendedConfigField]:
        """表达式缓存详细配置"""
        return {
            "batch_size": ExtendedConfigField(
                field_type=ConfigFieldType.INT,
                default=100,
                description="每批加载条数",
                constraint=ConfigConstraint(min_value=50, max_value=500),
            ),
            "batch_delay": ExtendedConfigField(
                field_type=ConfigFieldType.FLOAT,
                default=0.05,
                description="批次间延迟（秒）",
                constraint=ConfigConstraint(min_value=0.01, max_value=0.5),
            ),
            "refresh_interval": ExtendedConfigField(
                field_type=ConfigFieldType.INT,
                default=3600,
                description="自动刷新间隔（秒）",
                constraint=ConfigConstraint(min_value=600, max_value=86400),
                hot_reload=True,
            ),
            "incremental_refresh_interval": ExtendedConfigField(
                field_type=ConfigFieldType.INT,
                default=600,
                description="增量刷新间隔（秒）",
                constraint=ConfigConstraint(min_value=60, max_value=3600),
                hot_reload=True,
            ),
            "incremental_threshold_ratio": ExtendedConfigField(
                field_type=ConfigFieldType.FLOAT,
                default=0.1,
                description="触发全量重建的增量比例阈值",
                constraint=ConfigConstraint(min_value=0.01, max_value=1.0),
            ),
            "full_rebuild_interval": ExtendedConfigField(
                field_type=ConfigFieldType.INT,
                default=86400,
                description="全量重建间隔（秒）",
                constraint=ConfigConstraint(min_value=3600, max_value=604800),
            ),
            "deletion_check_interval": ExtendedConfigField(
                field_type=ConfigFieldType.INT,
                default=10,
                description="删除检测间隔（每N次增量刷新）",
                constraint=ConfigConstraint(min_value=1, max_value=100),
            ),
        }

    def _build_jargon_cache_schema(self) -> Dict[str, ExtendedConfigField]:
        """黑话缓存详细配置"""
        return {
            "batch_size": ExtendedConfigField(
                field_type=ConfigFieldType.INT,
                default=100,
                description="每批加载条数",
                constraint=ConfigConstraint(min_value=50, max_value=500),
            ),
            "batch_delay": ExtendedConfigField(
                field_type=ConfigFieldType.FLOAT,
                default=0.05,
                description="批次间延迟（秒）",
                constraint=ConfigConstraint(min_value=0.01, max_value=0.5),
            ),
            "refresh_interval": ExtendedConfigField(
                field_type=ConfigFieldType.INT,
                default=3600,
                description="自动刷新间隔（秒）",
                constraint=ConfigConstraint(min_value=600, max_value=86400),
                hot_reload=True,
            ),
            "enable_content_index": ExtendedConfigField(
                field_type=ConfigFieldType.BOOL,
                default=True,
                description="启用内容索引",
            ),
            "incremental_refresh_interval": ExtendedConfigField(
                field_type=ConfigFieldType.INT,
                default=600,
                description="增量刷新间隔（秒）",
                constraint=ConfigConstraint(min_value=60, max_value=3600),
                hot_reload=True,
            ),
            "incremental_threshold_ratio": ExtendedConfigField(
                field_type=ConfigFieldType.FLOAT,
                default=0.1,
                description="触发全量重建的增量比例阈值",
                constraint=ConfigConstraint(min_value=0.01, max_value=1.0),
            ),
            "full_rebuild_interval": ExtendedConfigField(
                field_type=ConfigFieldType.INT,
                default=86400,
                description="全量重建间隔（秒）",
                constraint=ConfigConstraint(min_value=3600, max_value=604800),
            ),
            "deletion_check_interval": ExtendedConfigField(
                field_type=ConfigFieldType.INT,
                default=10,
                description="删除检测间隔（每N次增量刷新）",
                constraint=ConfigConstraint(min_value=1, max_value=100),
            ),
        }

    def _build_kg_cache_schema(self) -> Dict[str, ExtendedConfigField]:
        """知识图谱缓存详细配置"""
        return {
            "batch_size": ExtendedConfigField(
                field_type=ConfigFieldType.INT,
                default=100,
                description="每批加载条数",
                constraint=ConfigConstraint(min_value=50, max_value=500),
            ),
            "batch_delay": ExtendedConfigField(
                field_type=ConfigFieldType.FLOAT,
                default=0.05,
                description="批次间延迟（秒）",
                constraint=ConfigConstraint(min_value=0.01, max_value=0.5),
            ),
            "refresh_interval": ExtendedConfigField(
                field_type=ConfigFieldType.INT,
                default=3600,
                description="自动刷新间隔（秒）",
                constraint=ConfigConstraint(min_value=600, max_value=86400),
                hot_reload=True,
            ),
            "incremental_refresh_interval": ExtendedConfigField(
                field_type=ConfigFieldType.INT,
                default=600,
                description="增量刷新间隔（秒）",
                constraint=ConfigConstraint(min_value=60, max_value=3600),
                hot_reload=True,
            ),
            "incremental_threshold_ratio": ExtendedConfigField(
                field_type=ConfigFieldType.FLOAT,
                default=0.1,
                description="触发全量重建的增量比例阈值",
                constraint=ConfigConstraint(min_value=0.01, max_value=1.0),
            ),
            "full_rebuild_interval": ExtendedConfigField(
                field_type=ConfigFieldType.INT,
                default=86400,
                description="全量重建间隔（秒）",
                constraint=ConfigConstraint(min_value=3600, max_value=604800),
            ),
            "deletion_check_interval": ExtendedConfigField(
                field_type=ConfigFieldType.INT,
                default=10,
                description="删除检测间隔（每N次增量刷新）",
                constraint=ConfigConstraint(min_value=1, max_value=100),
            ),
            "use_parquet": ExtendedConfigField(
                field_type=ConfigFieldType.BOOL,
                default=True,
                description="启用 Parquet 格式存储",
            ),
        }

    def _build_db_tuning_schema(self) -> Dict[str, ExtendedConfigField]:
        """数据库调优配置 schema"""
        return {
            "mmap_size": ExtendedConfigField(
                field_type=ConfigFieldType.INT,
                default=268435456,
                description="SQLite mmap_size(字节,0=禁用)",
                constraint=ConfigConstraint(min_value=0, max_value=2147483647),
                hot_reload=True,
            ),
            "wal_checkpoint_interval": ExtendedConfigField(
                field_type=ConfigFieldType.INT,
                default=300,
                description="WAL checkpoint周期(秒,0=禁用)",
                constraint=ConfigConstraint(min_value=0, max_value=86400),
                hot_reload=True,
            ),
        }

    def _build_lightweight_profiler_schema(self) -> Dict[str, ExtendedConfigField]:
        """轻量性能剖析配置 schema"""
        return {
            "sample_rate": ExtendedConfigField(
                field_type=ConfigFieldType.FLOAT,
                default=0.1,
                description="采样率(0-1)",
                constraint=ConfigConstraint(min_value=0.0, max_value=1.0),
                hot_reload=True,
            ),
        }

    def _build_advanced_schema(self) -> Dict[str, ExtendedConfigField]:
        """高级配置 schema"""
        return {
            "enable_async_io": ExtendedConfigField(
                field_type=ConfigFieldType.BOOL,
                default=True,
                description="启用异步IO优化（需要 aiofiles）",
                section="advanced",
                order=0,
            ),
            "enable_orjson": ExtendedConfigField(
                field_type=ConfigFieldType.BOOL,
                default=True,
                description="启用 orjson 加速（需要 orjson）",
                section="advanced",
                order=1,
            ),
            "thread_pool_size": ExtendedConfigField(
                field_type=ConfigFieldType.INT,
                default=4,
                description="线程池大小",
                constraint=ConfigConstraint(min_value=1, max_value=32),
                section="advanced",
                order=2,
            ),
            "gc_interval": ExtendedConfigField(
                field_type=ConfigFieldType.INT,
                default=300,
                description="垃圾回收间隔（秒）",
                constraint=ConfigConstraint(min_value=60, max_value=3600),
                section="advanced",
                order=3,
                hot_reload=True,
            ),
            "enable_hot_reload": ExtendedConfigField(
                field_type=ConfigFieldType.BOOL,
                default=True,
                description="启用配置热更新",
                section="advanced",
                order=4,
            ),
            "strict_validation": ExtendedConfigField(
                field_type=ConfigFieldType.BOOL,
                default=False,
                description="配置验证严格模式",
                section="advanced",
                order=5,
            ),
            "enable_change_notifications": ExtendedConfigField(
                field_type=ConfigFieldType.BOOL,
                default=True,
                description="启用配置变更通知",
                section="advanced",
                order=6,
            ),
        }

    def _build_monitoring_schema(self) -> Dict[str, ExtendedConfigField]:
        """监控配置 schema"""
        return {
            "enable_stats": ExtendedConfigField(
                field_type=ConfigFieldType.BOOL,
                default=True,
                description="启用统计功能",
                section="monitoring",
                order=0,
            ),
            "stats_interval": ExtendedConfigField(
                field_type=ConfigFieldType.INT,
                default=60,
                description="统计报告间隔（秒）",
                constraint=ConfigConstraint(min_value=10, max_value=3600),
                section="monitoring",
                order=1,
                hot_reload=True,
            ),
            "enable_memory_monitor": ExtendedConfigField(
                field_type=ConfigFieldType.BOOL,
                default=True,
                description="启用内存监控",
                section="monitoring",
                order=2,
            ),
            "memory_warning_threshold": ExtendedConfigField(
                field_type=ConfigFieldType.FLOAT,
                default=0.8,
                description="内存警告阈值（0-1）",
                constraint=ConfigConstraint(min_value=0.1, max_value=1.0),
                section="monitoring",
                order=3,
                hot_reload=True,
            ),
            "memory_critical_threshold": ExtendedConfigField(
                field_type=ConfigFieldType.FLOAT,
                default=0.9,
                description="内存临界阈值（0-1）",
                constraint=ConfigConstraint(min_value=0.1, max_value=1.0),
                section="monitoring",
                order=4,
                hot_reload=True,
            ),
            "enable_health_check": ExtendedConfigField(
                field_type=ConfigFieldType.BOOL,
                default=True,
                description="启用健康检查",
                section="monitoring",
                order=5,
            ),
            "health_check_interval": ExtendedConfigField(
                field_type=ConfigFieldType.INT,
                default=30,
                description="健康检查间隔（秒）",
                constraint=ConfigConstraint(min_value=10, max_value=300),
                section="monitoring",
                order=6,
            ),
        }

    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        config: Dict[str, Any] = {}

        for section_name, section_schema in self._schema.items():
            config[section_name] = {}
            for field_name, field_def in section_schema.items():
                if field_def.field_type == ConfigFieldType.NESTED and field_def.nested_schema:
                    config[section_name][field_name] = {}
                    for nested_name, nested_def in field_def.nested_schema.items():
                        config[section_name][field_name][nested_name] = nested_def.default
                else:
                    config[section_name][field_name] = field_def.default

        return config

    def load(self) -> Dict[str, Any]:
        """加载配置文件"""
        with self._config_lock:
            default_config = self.get_default_config()

            if not self._config_path.exists():
                logger.info("[Config] 配置文件不存在，使用默认配置")
                self._config = default_config
                return copy.deepcopy(self._config)

            try:
                if tomllib is not None:
                    with open(self._config_path, "rb") as f:
                        loaded_config = tomllib.load(f)
                else:
                    # Fallback to JSON
                    json_path = self._config_path.with_suffix(".json")
                    if json_path.exists():
                        with open(json_path, "r", encoding="utf-8") as f:
                            loaded_config = json.load(f)
                    else:
                        logger.warning("[Config] 无法加载 TOML（缺少 tomllib），使用默认配置")
                        self._config = default_config
                        return copy.deepcopy(self._config)

                # 检查版本并迁移
                loaded_version = loaded_config.get("plugin", {}).get("config_version", "1.0.0")
                if not ConfigVersion.is_compatible(loaded_version):
                    logger.warning(
                        f"[Config] 配置版本 {loaded_version} 不兼容，使用默认配置"
                    )
                    self._config = default_config
                    return copy.deepcopy(self._config)

                # 执行迁移
                migrated_config = self._migrator.migrate(loaded_config)

                # 合并配置（保留默认值）
                self._config = self._merge_config(default_config, migrated_config)

                # 验证配置
                validation_errors = self.validate()
                if validation_errors:
                    logger.warning(f"[Config] 配置验证警告: {validation_errors}")

                logger.info(f"[Config] 配置加载完成，版本 {self._config.get('plugin', {}).get('config_version', 'unknown')}")

            except Exception as e:
                logger.error(f"[Config] 配置加载失败: {e}，使用默认配置")
                self._config = default_config

            return copy.deepcopy(self._config)

    def _merge_config(self, default: Dict, loaded: Dict) -> Dict:
        """递归合并配置"""
        result = copy.deepcopy(default)

        for key, value in loaded.items():
            if key in result:
                if isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = self._merge_config(result[key], value)
                else:
                    result[key] = value
            else:
                result[key] = value

        return result

    def validate(self) -> List[str]:
        """验证当前配置"""
        errors = []

        for section_name, section_schema in self._schema.items():
            section_config = self._config.get(section_name, {})

            for field_name, field_def in section_schema.items():
                if field_name not in section_config:
                    continue

                value = section_config[field_name]
                valid, error = field_def.validate(value, self._config)
                if not valid:
                    errors.append(f"{section_name}.{field_name}: {error}")

        return errors

    def get(self, path: str, default: Any = None) -> Any:
        """获取配置值

        Args:
            path: 配置路径，如 "modules.message_cache_enabled" 或 "modules.message_cache.ttl"
            default: 默认值

        Returns:
            配置值或默认值
        """
        with self._config_lock:
            parts = path.split(".")
            current = self._config
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return default
            return current

    def set(self, path: str, value: Any, notify: bool = True) -> bool:
        """设置配置值

        Args:
            path: 配置路径
            value: 新值
            notify: 是否通知监听器

        Returns:
            是否设置成功
        """
        with self._config_lock:
            parts = path.split(".")
            if len(parts) < 2:
                return False

            # 验证新值
            section_name = parts[0]
            if section_name not in self._schema:
                return False

            # 获取字段定义
            field_def = self._get_field_def(path)
            if field_def is None:
                return False

            # 验证
            valid, error = field_def.validate(value, self._config)
            if not valid:
                logger.warning(f"[Config] 配置验证失败 {path}: {error}")
                return False

            # 检查是否支持热更新
            if not field_def.hot_reload:
                logger.warning(f"[Config] 配置项 {path} 不支持热更新，需要重启生效")

            # 设置值
            old_value = self.get(path)
            current = self._config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value

            # 通知监听器
            if notify and old_value != value:
                self._notify_listeners(path, old_value, value)

            return True

    def _get_field_def(self, path: str) -> Optional[ExtendedConfigField]:
        """获取字段定义"""
        parts = path.split(".")
        if len(parts) < 2:
            return None

        section_name = parts[0]
        if section_name not in self._schema:
            return None

        section_schema = self._schema[section_name]
        field_name = parts[1]

        if field_name not in section_schema:
            return None

        field_def = section_schema[field_name]

        # 处理嵌套字段
        if len(parts) > 2 and field_def.nested_schema:
            nested_name = parts[2]
            if nested_name in field_def.nested_schema:
                return field_def.nested_schema[nested_name]
            return None

        return field_def

    def add_listener(self, path: str, callback: Callable[[str, Any, Any], None]):
        """添加配置变更监听器

        Args:
            path: 监听的配置路径（支持通配符 "*"）
            callback: 回调函数 (path, old_value, new_value)
        """
        if path not in self._listeners:
            self._listeners[path] = []
        self._listeners[path].append(callback)

    def remove_listener(self, path: str, callback: Callable[[str, Any, Any], None]):
        """移除配置变更监听器"""
        if path in self._listeners and callback in self._listeners[path]:
            self._listeners[path].remove(callback)

    def _notify_listeners(self, path: str, old_value: Any, new_value: Any):
        """通知监听器"""
        # 精确匹配
        if path in self._listeners:
            for callback in self._listeners[path]:
                try:
                    callback(path, old_value, new_value)
                except Exception as e:
                    logger.error(f"[Config] 监听器回调失败: {e}")

        # 通配符匹配
        for pattern, callbacks in self._listeners.items():
            if pattern == "*" or (pattern.endswith(".*") and path.startswith(pattern[:-2])):
                for callback in callbacks:
                    try:
                        callback(path, old_value, new_value)
                    except Exception as e:
                        logger.error(f"[Config] 监听器回调失败: {e}")

    def get_hot_reload_fields(self) -> List[str]:
        """获取所有支持热更新的配置项路径"""
        fields = []

        for section_name, section_schema in self._schema.items():
            for field_name, field_def in section_schema.items():
                if field_def.hot_reload:
                    fields.append(f"{section_name}.{field_name}")

                if field_def.nested_schema:
                    for nested_name, nested_def in field_def.nested_schema.items():
                        if nested_def.hot_reload:
                            fields.append(f"{section_name}.{field_name}.{nested_name}")

        return fields

    def get_schema(self) -> Dict[str, Dict[str, ExtendedConfigField]]:
        """获取完整 schema"""
        return copy.deepcopy(self._schema)

    def to_dict(self) -> Dict[str, Any]:
        """导出配置为字典"""
        with self._config_lock:
            return copy.deepcopy(self._config)

    def reset_to_default(self, path: Optional[str] = None):
        """重置配置为默认值

        Args:
            path: 指定路径重置，None 表示全部重置
        """
        with self._config_lock:
            if path is None:
                old_config = copy.deepcopy(self._config)
                self._config = self.get_default_config()
                self._notify_listeners("*", old_config, self._config)
            else:
                field_def = self._get_field_def(path)
                if field_def:
                    self.set(path, field_def.default)


# 便捷函数
_config_manager: Optional[ConfigManager] = None


def get_config_manager(plugin_dir: Optional[Path] = None) -> ConfigManager:
    """获取配置管理器单例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(plugin_dir)
    return _config_manager


def get_config(path: str, default: Any = None) -> Any:
    """便捷函数：获取配置值"""
    return get_config_manager().get(path, default)


def set_config(path: str, value: Any) -> bool:
    """便捷函数：设置配置值"""
    return get_config_manager().set(path, value)
