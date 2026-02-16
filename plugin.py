"""
CM 性能优化插件

功能模块：
1. 消息缓存 (message_cache) - 缓存 find_messages 查询结果
2. 人物信息缓存 (person_cache) - 缓存人物信息查询
3. 表达式缓存 (expression_cache) - 双缓冲+缓慢加载+原子切换
4. 黑话缓存 (jargon_cache) - 双缓冲+缓慢加载+原子切换+内容索引
5. 知识库图谱缓存 (kg_cache) - 双缓冲+缓慢加载+原子切换

配置系统：
- 完整的配置验证和热更新支持
- 向后兼容旧版本配置
- 详细的模块配置选项

通知系统：
- QQ消息通知渠道
- 控制台通知渠道
- 错误日志通知
- 性能警告通知

安装：将目录放入 MaiBot/plugins/ 下，重启 MaiBot
依赖：无额外依赖（可选：aiofiles, orjson, psutil）

版本信息请参阅 version.py
"""

from __future__ import annotations

import sys
import asyncio
import time
import threading
import importlib.util
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union

# 统一版本管理
from version import PLUGIN_VERSION, CONFIG_VERSION, MAIBOT_MIN_VERSION

# 异步优化相关导入
try:
    import aiofiles

    AIOFILES_AVAILABLE = True
except ImportError:
    aiofiles = None
    AIOFILES_AVAILABLE = False

try:
    import orjson

    ORJSON_AVAILABLE = True
except ImportError:
    orjson = None
    ORJSON_AVAILABLE = False

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

try:
    from src.plugin_system.apis.plugin_register_api import register_plugin
    from src.plugin_system.base.base_plugin import BasePlugin
    from src.plugin_system.base.base_events_handler import BaseEventHandler
    from src.plugin_system.base.component_types import EventType, PythonDependency
    from src.plugin_system.base.config_types import ConfigField, ConfigSection, ConfigLayout, ConfigTab
    from src.common.logger import get_logger
except ImportError:
    # 让本文件可被"独立 import"用于静态检查/离线测试
    class _FallbackEventType:
        ON_STOP = "on_stop"

    EventType = _FallbackEventType  # type: ignore

    class BasePlugin:
        def __init__(self, plugin_dir=None):
            pass

    class BaseEventHandler:
        def __init__(self, plugin_dir=None):
            pass

    class PythonDependency:
        def __init__(self, *a, **kw):
            pass

    class ConfigField:
        def __init__(self, **kw):
            pass

    class ConfigSection:
        def __init__(self, **kw):
            pass

    class ConfigLayout:
        def __init__(self, **kw):
            pass

    class ConfigTab:
        def __init__(self, **kw):
            pass

    def register_plugin(cls):
        return cls

    def get_logger(name):
        import logging

        return logging.getLogger(name)

logger = get_logger("CM_perf_opt")

PLUGIN_NAME = "CM-performance-optimizer"
# PLUGIN_VERSION 已从 version.py 统一导入

# 全局变量，用于存储动态加载的模块
_global_modules: Dict[str, Any] = {}

# 全局变量，存储插件实例（供事件处理器使用）
_plugin_instance: Optional["CMPerformanceOptimizerPlugin"] = None


def _load_local_module(module_filename: str, module_name: str):
    """Load a sibling .py module by file path.

    MaiBot loads external plugins via spec_from_file_location with a non-package module name
    (and directories may contain '-'), so relative imports (from .xxx import yyy) are unreliable.
    """
    if module_name in sys.modules:
        return sys.modules[module_name]
    plugin_dir = Path(__file__).parent
    module_path = plugin_dir / module_filename
    if not module_path.exists():
        raise FileNotFoundError(f"Module file not found: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# 缓存管理器（单例模式）
class _CacheManager:
    """管理所有缓存实例"""

    _instance: Optional["_CacheManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.caches: Dict[str, Any] = {}
        self.logger = logger

    def register_cache(self, name: str, cache_instance: Any) -> None:
        """注册缓存实例。
        
        Args:
            name: 缓存名称，用于后续检索
            cache_instance: 缓存实例对象
        """
        self.caches[name] = cache_instance
        self.logger.debug(f"[CacheManager] 注册缓存: {name}")

    def get_cache(self, name: str) -> Optional[Any]:
        """获取缓存实例。
        
        Args:
            name: 缓存名称
            
        Returns:
            缓存实例，若不存在则返回 None
        """
        return self.caches.get(name)

    def clear_all(self) -> None:
        """清除所有缓存。
        
        遍历所有已注册的缓存实例，调用其 clear() 方法清空缓存数据，
        最后清空缓存注册表。
        """
        for name, cache in self.caches.items():
            try:
                if hasattr(cache, "clear"):
                    cache.clear()
                self.logger.debug(f"[CacheManager] 清除缓存: {name}")
            except Exception as e:
                self.logger.error(f"[CacheManager] 清除缓存 {name} 失败: {e}")
        self.caches.clear()

    def stop_all(self) -> None:
        """停止所有缓存。
        
        遍历所有已注册的缓存实例，调用其 stop() 方法停止后台任务，
        释放相关资源。
        """
        for name, cache in self.caches.items():
            try:
                if hasattr(cache, "stop"):
                    cache.stop()
                self.logger.debug(f"[CacheManager] 停止缓存: {name}")
            except Exception as e:
                self.logger.error(f"[CacheManager] 停止缓存 {name} 失败: {e}")

    def get_memory_usage(self) -> Dict[str, int]:
        """获取所有缓存的内存使用"""
        result = {}
        for name, cache in self.caches.items():
            try:
                if hasattr(cache, "get_memory_usage"):
                    result[name] = cache.get_memory_usage()
            except Exception:
                result[name] = 0
        return result


def _try_early_preload_kg_cache() -> None:
    """尽可能提前启动 kg_cache 预加载，并提前 patch KGManager.load_from_file。

    背景：[`lpmm_start_up()`](../src/chat/knowledge/__init__.py:38) 启动阶段会同步调用
    `KGManager.load_from_file()`。

    若插件仅在 ON_START 才应用补丁，通常已经错过唯一调用点，导致：
    - kg_cache 统计长期 0 命中
    - 无法降低主线程启动卡顿

    因此这里在插件模块导入阶段就尝试启动 kg_cache（失败不影响启动）。
    """

    try:
        plugin_dir = Path(__file__).parent
        cfg_path = plugin_dir / "config.toml"

        # 默认“尝试启用”，若配置显式关闭则跳过
        allow = True
        try:
            import tomllib  # py311+

            if cfg_path.exists():
                with open(cfg_path, "rb") as f:
                    cfg = tomllib.load(f)
                if isinstance(cfg, dict):
                    if cfg.get("plugin", {}).get("enabled") is False:
                        allow = False
                    if cfg.get("modules", {}).get("kg_cache_enabled") is False:
                        allow = False
        except Exception:
            # 解析失败时保持 allow=True（以便尽量提前预热）
            pass

        if not allow:
            return

        kg_cache_module = _load_local_module(
            "components/modules/kg_cache.py",
            "CM_perf_opt_kg_cache",
        )
        apply_kg_cache = getattr(kg_cache_module, "apply_kg_cache", None)
        if callable(apply_kg_cache):
            apply_kg_cache(_CacheManager())
            logger.info("[PerfOpt] ✓ kg_cache 已提前预热/patch（import-time）")

    except Exception as e:
        # 仅 debug：不影响插件加载
        try:
            logger.debug(f"[PerfOpt] kg_cache 提前预热失败（忽略）: {e}")
        except Exception:
            pass


# 尽早执行一次预热（失败不影响启动）
_try_early_preload_kg_cache()


# 性能优化器（单例模式）
class _PerformanceOptimizer:
    """性能优化器，管理所有优化模块"""

    _instance: Optional["_PerformanceOptimizer"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.cache_manager = _CacheManager()
        self.patches_applied = False
        self.started = False
        self.logger = logger
        self.plugin_instance: Optional["CMPerformanceOptimizerPlugin"] = None

        # 配置和监控组件（延迟初始化）
        self._config_manager = None
        self._memory_monitor = None
        self._stats_reporter = None
        self._module_enabler = None

        # 通知系统组件（延迟初始化）
        self._notification_manager = None
        self._log_handler = None

    def set_plugin_instance(self, plugin_instance: "CMPerformanceOptimizerPlugin"):
        """设置插件实例引用"""
        self.plugin_instance = plugin_instance

    def _init_config_system(self):
        """初始化配置系统"""
        try:
            from .core import (
                get_config_manager,
                get_memory_monitor,
                get_stats_reporter,
                get_module_enabler,
            )

            plugin_dir = Path(__file__).parent
            self._config_manager = get_config_manager(plugin_dir)
            self._config_manager.load()

            self._memory_monitor = get_memory_monitor()
            self._stats_reporter = get_stats_reporter()
            self._module_enabler = get_module_enabler()

            self.logger.info("[PerfOpt] ✓ 配置系统初始化完成")
        except Exception as e:
            self.logger.warning(f"[PerfOpt] 配置系统初始化失败，使用默认配置: {e}")

    def _init_notification_system(self):
        """初始化通知系统（简化版：只在内存占用过高时通知）"""
        try:
            from .core import (
                NotificationConfig,
                get_notification_manager,
                init_notification_manager,
            )

            # 读取通知配置（简化版）
            notification_enabled = True
            admin_qq = ""

            if self._config_manager:
                notification_enabled = self._config_manager.get("notification.enabled", True)
                admin_qq = self._config_manager.get("notification.admin_qq", "")

            # 解析QQ号（支持字符串或整数）
            qq_target = 0
            if admin_qq:
                try:
                    qq_target = int(str(admin_qq).strip())
                except (ValueError, TypeError):
                    qq_target = 0

            # 创建通知配置（简化版）
            notification_config = NotificationConfig(
                enabled=notification_enabled,
                mode="qq" if qq_target > 0 else "console",
                qq_target=qq_target,
                qq_level="warning",
                qq_cooldown_seconds=300.0,
                qq_daily_limit=50,
                performance_warning_enabled=True,
                memory_warning_enabled=True,
                memory_critical_enabled=True,
            )

            # 初始化通知管理器
            self._notification_manager = init_notification_manager(notification_config)

            # 尝试设置 Bot 实例
            self._try_set_bot_instance()

            if qq_target > 0:
                self.logger.info(f"[PerfOpt] ✓ 通知系统初始化完成，QQ: {qq_target}")
            else:
                self.logger.info("[PerfOpt] ✓ 通知系统初始化完成（仅控制台模式）")

        except Exception as e:
            self.logger.warning(f"[PerfOpt] 通知系统初始化失败: {e}")

    def _try_set_bot_instance(self):
        """尝试设置 Bot 实例到通知管理器"""
        if not self._notification_manager:
            return

        try:
            # 尝试从不同来源获取 Bot 实例
            bot_instance = None

            # 方式1: 从全局变量获取
            try:
                from src.core.bot import bot
                bot_instance = bot
            except ImportError:
                pass

            # 方式2: 从插件实例获取
            if bot_instance is None and self.plugin_instance:
                try:
                    bot_instance = getattr(self.plugin_instance, "_bot", None)
                except Exception:
                    pass

            if bot_instance:
                self._notification_manager.set_bot_instance(bot_instance)
                self.logger.info("[PerfOpt] ✓ Bot 实例已设置到通知管理器")

        except Exception as e:
            self.logger.debug(f"[PerfOpt] 设置 Bot 实例失败（稍后重试）: {e}")

    def _reload_notification_config(self):
        """重载通知系统配置（简化版）"""
        if not self._config_manager or not self._notification_manager:
            return

        try:
            from .core import NotificationConfig

            # 读取简化配置
            notification_enabled = self._config_manager.get("notification.enabled", True)
            admin_qq = self._config_manager.get("notification.admin_qq", "")

            # 解析QQ号
            qq_target = 0
            if admin_qq:
                try:
                    qq_target = int(str(admin_qq).strip())
                except (ValueError, TypeError):
                    qq_target = 0

            # 创建通知配置
            notification_config = NotificationConfig(
                enabled=notification_enabled,
                mode="qq" if qq_target > 0 else "console",
                qq_target=qq_target,
                qq_level="warning",
                qq_cooldown_seconds=300.0,
                qq_daily_limit=50,
                performance_warning_enabled=True,
                memory_warning_enabled=True,
                memory_critical_enabled=True,
            )

            self._notification_manager.set_config(notification_config)
            self.logger.info("[PerfOpt] ✓ 通知系统配置已重载")

        except Exception as e:
            self.logger.warning(f"[PerfOpt] 通知系统配置重载失败: {e}")

    def _register_cache_memory_callbacks(self):
        """注册缓存内存监控回调"""
        if not self._memory_monitor:
            return

        for name, cache in self.cache_manager.caches.items():
            if hasattr(cache, "get_memory_usage"):
                self._memory_monitor.register_cache_memory_callback(
                    name, cache.get_memory_usage
                )

    def _register_stats_callbacks(self):
        """注册统计回调

        约定：
        - 回调返回的是一个 dict，将被 StatsReporter 直接格式化。
        - 为支持“间隔命中次数”等字段，这里会在每次采集时调用 stats.reset_interval()。
          因此 i_* 表示“自上次采集以来”的统计。
        """
        if not self._stats_reporter:
            return

        for name, cache in self.cache_manager.caches.items():
            if not hasattr(cache, "stats"):
                continue

            def make_callback(c):
                def _cb() -> Dict[str, Any]:
                    s = getattr(c, "stats", None)
                    if s is None:
                        return {}

                    out: Dict[str, Any] = {}

                    # 1) 累计统计
                    try:
                        if hasattr(s, "total"):
                            total = s.total()
                            if isinstance(total, dict):
                                out.update(total)
                    except Exception as e:
                        return {"error": f"total() 失败: {e}"}

                    # 2) 间隔统计（采集后清零）
                    try:
                        if hasattr(s, "reset_interval"):
                            interval = s.reset_interval()
                            if isinstance(interval, dict):
                                out.update(interval)
                    except Exception:
                        # 间隔统计缺失不影响累计统计
                        pass

                    return out

                return _cb

            self._stats_reporter.register_stats_callback(name, make_callback(cache))

    def apply_patches(self):
        """应用性能优化补丁"""
        if self.patches_applied:
            self.logger.debug("[PerfOpt] 补丁已应用，跳过")
            return

        try:
            self.logger.info("[PerfOpt] 开始应用性能优化补丁...")


            # 初始化 PatchChain（链式 patch 管理器）
            from .core.patch_chain import get_patch_chain
            self._patch_chain = get_patch_chain()
            # 初始化配置系统
            self._init_config_system()

            # 获取模块启用状态
            enable_lightweight_profiler = False
            profiler_sample_rate = 0.1

            enable_message_cache = True
            enable_message_repository_fastpath = True
            enable_person_cache = True
            enable_regex_precompile = True
            enable_typo_generator_cache = True
            enable_user_reference_batch_resolve = True
            enable_expression_cache = True
            enable_jargon_cache = True
            enable_jargon_matcher_automaton = True
            enable_kg_cache = True
            enable_levenshtein_fast = True
            enable_image_desc_bulk_lookup = True
            enable_db_tuning = True
            db_mmap_size = 268435456
            db_wal_checkpoint_interval = 300
            enable_asyncio_loop_pool = True
 
            if self._config_manager:
                enable_lightweight_profiler = self._config_manager.get(
                    "modules.lightweight_profiler_enabled", False
                )
                profiler_sample_rate = float(
                    self._config_manager.get("modules.lightweight_profiler.sample_rate", 0.1)
                )
 
                enable_message_cache = self._config_manager.get(
                    "modules.message_cache_enabled", True
                )
                enable_message_repository_fastpath = self._config_manager.get(
                    "modules.message_repository_fastpath_enabled", True
                )
                enable_person_cache = self._config_manager.get(
                    "modules.person_cache_enabled", True
                )
                enable_regex_precompile = self._config_manager.get(
                    "modules.regex_precompile_enabled", True
                )
                enable_typo_generator_cache = self._config_manager.get(
                    "modules.typo_generator_cache_enabled", True
                )
                enable_user_reference_batch_resolve = self._config_manager.get(
                    "modules.user_reference_batch_resolve_enabled", True
                )
                enable_expression_cache = self._config_manager.get(
                    "modules.expression_cache_enabled", True
                )
                enable_jargon_cache = self._config_manager.get(
                    "modules.jargon_cache_enabled", True
                )
                enable_jargon_matcher_automaton = self._config_manager.get(
                    "modules.jargon_matcher_automaton_enabled", True
                )
                enable_kg_cache = self._config_manager.get(
                    "modules.kg_cache_enabled", True
                )
                enable_levenshtein_fast = self._config_manager.get(
                    "modules.levenshtein_fast_enabled", True
                )
                enable_image_desc_bulk_lookup = self._config_manager.get(
                    "modules.image_desc_bulk_lookup_enabled", True
                )
                enable_db_tuning = self._config_manager.get(
                    "modules.db_tuning_enabled", True
                )
                enable_asyncio_loop_pool = self._config_manager.get(
                    "modules.asyncio_loop_pool_enabled", True
                )
                db_mmap_size = int(self._config_manager.get("modules.db_tuning.mmap_size", 268435456))
                db_wal_checkpoint_interval = int(
                    self._config_manager.get("modules.db_tuning.wal_checkpoint_interval", 300)
                )

            # 动态加载并应用各个缓存模块
            # DB tuning（PRAGMA + 索引自检 + 可选 checkpoint）
            if enable_db_tuning:
                try:
                    db_tuning_module = _load_local_module(
                        "components/modules/db_tuning.py",
                        "CM_perf_opt_db_tuning",
                    )
                    apply_db_tuning = getattr(db_tuning_module, "apply_db_tuning", None)
                    if apply_db_tuning:
                        mod = apply_db_tuning(self.cache_manager)
                        if mod is not None:
                            try:
                                mod.mmap_size = int(db_mmap_size)
                                mod.checkpoint_interval = int(db_wal_checkpoint_interval)
                                # 二次 apply，确保使用最新配置值（模块内部幂等）
                                if hasattr(mod, "apply_patch"):
                                    mod.apply_patch()
                            except Exception:
                                pass
                        self.logger.info("[PerfOpt] ✓ DB tuning 已启用")
                except Exception as e:
                    self.logger.error(f"[PerfOpt] DB tuning 启用失败: {e}")
            else:
                self.logger.info("[PerfOpt] DB tuning 已禁用")

            # Lightweight Profiler（纯观测层，默认关闭）
            if enable_lightweight_profiler:
                try:
                    profiler_module = _load_local_module(
                        "components/modules/lightweight_profiler.py",
                        "CM_perf_opt_lightweight_profiler",
                    )
                    apply_lightweight_profiler = getattr(
                        profiler_module, "apply_lightweight_profiler", None
                    )
                    if apply_lightweight_profiler:
                        mod = apply_lightweight_profiler(self.cache_manager)
                        if mod is not None:
                            try:
                                mod.sample_rate = float(profiler_sample_rate)
                            except Exception:
                                pass
                        self.logger.info(
                            "[PerfOpt] ✓ Lightweight profiler 已启用（观测层）"
                        )
                except Exception as e:
                    self.logger.error(f"[PerfOpt] Lightweight profiler 启用失败: {e}")
            else:
                self.logger.info("[PerfOpt] Lightweight profiler 已禁用")

            # 消息缓存
            if enable_message_cache:
                try:
                    message_cache_module = _load_local_module(
                        "components/modules/message_cache.py", "CM_perf_opt_message_cache"
                    )
                    apply_message_cache = getattr(
                        message_cache_module, "apply_message_cache", None
                    )
                    if apply_message_cache:
                        apply_message_cache(self.cache_manager)
                        self.logger.info("[PerfOpt] ✓ 消息缓存补丁已应用")
                except Exception as e:
                    self.logger.error(f"[PerfOpt] 消息缓存补丁失败: {e}")
            else:
                self.logger.info("[PerfOpt] 消息缓存已禁用")

            # message_repository count 快速路径（仅 patch count_messages）
            if enable_message_repository_fastpath:
                try:
                    mr_fast_module = _load_local_module(
                        "components/modules/message_repository_fastpath.py",
                        "CM_perf_opt_message_repository_fastpath",
                    )
                    apply_message_repository_fastpath = getattr(
                        mr_fast_module, "apply_message_repository_fastpath", None
                    )
                    if apply_message_repository_fastpath:
                        apply_message_repository_fastpath(self.cache_manager)
                        self.logger.info(
                            "[PerfOpt] ✓ message_repository_fastpath 已启用（仅 count_messages）"
                        )
                except Exception as e:
                    self.logger.error(
                        f"[PerfOpt] message_repository_fastpath 启用失败: {e}"
                    )
            else:
                self.logger.info("[PerfOpt] message_repository_fastpath 已禁用")

            # 人物信息缓存
            if enable_person_cache:
                try:
                    person_cache_module = _load_local_module(
                        "components/modules/person_cache.py", "CM_perf_opt_person_cache"
                    )
                    apply_person_cache = getattr(
                        person_cache_module, "apply_person_cache", None
                    )
                    if apply_person_cache:
                        apply_person_cache(self.cache_manager)
                        self.logger.info("[PerfOpt] ✓ 人物信息缓存补丁已应用")
                except Exception as e:
                    self.logger.error(f"[PerfOpt] 人物信息缓存补丁失败: {e}")
            else:
                self.logger.info("[PerfOpt] 人物信息缓存已禁用")

            # 正则预编译优化（regex_precompile）
            # 注意：建议在 user_reference_batch_resolve 之前应用，以便后者包装预编译实现。
            if enable_regex_precompile:
                try:
                    regex_module = _load_local_module(
                        "components/modules/regex_precompile.py",
                        "CM_perf_opt_regex_precompile",
                    )
                    apply_regex_precompile = getattr(
                        regex_module, "apply_regex_precompile", None
                    )
                    if apply_regex_precompile:
                        apply_regex_precompile(self.cache_manager)
                        self.logger.info(
                            "[PerfOpt] ✓ regex_precompile 已启用（预编译高频正则）"
                        )
                except Exception as e:
                    self.logger.error(f"[PerfOpt] regex_precompile 启用失败: {e}")
            else:
                self.logger.info("[PerfOpt] regex_precompile 已禁用")

            # 用户引用批量解析缓存（user_reference_batch_resolve）
            if enable_user_reference_batch_resolve:
                try:
                    user_ref_module = _load_local_module(
                        "components/modules/user_reference_batch_resolve.py",
                        "CM_perf_opt_user_reference_batch_resolve",
                    )
                    apply_user_reference_batch_resolve = getattr(
                        user_ref_module, "apply_user_reference_batch_resolve", None
                    )
                    if apply_user_reference_batch_resolve:
                        apply_user_reference_batch_resolve(self.cache_manager)
                        self.logger.info(
                            "[PerfOpt] ✓ 用户引用批量解析缓存补丁已应用"
                        )
                except Exception as e:
                    self.logger.error(
                        f"[PerfOpt] 用户引用批量解析缓存补丁失败: {e}"
                    )
            else:
                self.logger.info("[PerfOpt] 用户引用批量解析缓存已禁用")

            # 表达式缓存
            if enable_expression_cache:
                try:
                    expression_cache_module = _load_local_module(
                        "components/modules/expression_cache.py",
                        "CM_perf_opt_expression_cache",
                    )
                    apply_expression_cache = getattr(
                        expression_cache_module, "apply_expression_cache", None
                    )
                    if apply_expression_cache:
                        apply_expression_cache(self.cache_manager)
                        self.logger.info("[PerfOpt] ✓ 表达式缓存补丁已应用")
                except Exception as e:
                    self.logger.error(f"[PerfOpt] 表达式缓存补丁失败: {e}")
            else:
                self.logger.info("[PerfOpt] 表达式缓存已禁用")

            # 黑话缓存
            if enable_jargon_cache:
                try:
                    jargon_cache_module = _load_local_module(
                        "components/modules/jargon_cache.py", "CM_perf_opt_jargon_cache"
                    )
                    apply_jargon_cache = getattr(
                        jargon_cache_module, "apply_jargon_cache", None
                    )
                    if apply_jargon_cache:
                        apply_jargon_cache(self.cache_manager)
                        self.logger.info("[PerfOpt] ✓ 黑话缓存补丁已应用")
                except Exception as e:
                    self.logger.error(f"[PerfOpt] 黑话缓存补丁失败: {e}")
            else:
                self.logger.info("[PerfOpt] 黑话缓存已禁用")

            # 黑话匹配自动机（Aho-Corasick）
            if enable_jargon_matcher_automaton:
                try:
                    jm_auto_module = _load_local_module(
                        "components/modules/jargon_matcher_automaton.py",
                        "CM_perf_opt_jargon_matcher_automaton",
                    )
                    apply_jargon_matcher_automaton = getattr(
                        jm_auto_module, "apply_jargon_matcher_automaton", None
                    )
                    if apply_jargon_matcher_automaton:
                        apply_jargon_matcher_automaton(self.cache_manager)
                        self.logger.info(
                            "[PerfOpt] ✓ 黑话匹配自动机补丁已应用（Aho-Corasick）"
                        )
                except Exception as e:
                    self.logger.error(
                        f"[PerfOpt] 黑话匹配自动机补丁失败: {e}"
                    )
            else:
                self.logger.info("[PerfOpt] 黑话匹配自动机已禁用")

            # 知识库图谱缓存
            if enable_kg_cache:
                try:
                    kg_cache_module = _load_local_module(
                        "components/modules/kg_cache.py", "CM_perf_opt_kg_cache"
                    )
                    apply_kg_cache = getattr(kg_cache_module, "apply_kg_cache", None)
                    if apply_kg_cache:
                        apply_kg_cache(self.cache_manager)
                        self.logger.info("[PerfOpt] ✓ 知识库图谱缓存补丁已应用")
                except Exception as e:
                    self.logger.error(f"[PerfOpt] 知识库图谱缓存补丁失败: {e}")
            else:
                self.logger.info("[PerfOpt] 知识库图谱缓存已禁用")
    
            # Levenshtein 距离加速（rapidfuzz）
            if enable_levenshtein_fast:
                try:
                    levenshtein_fast_module = _load_local_module(
                        "components/modules/levenshtein_fast.py",
                        "CM_perf_opt_levenshtein_fast",
                    )
                    apply_levenshtein_fast = getattr(
                        levenshtein_fast_module, "apply_levenshtein_fast", None
                    )
                    if apply_levenshtein_fast:
                        apply_levenshtein_fast(self.cache_manager)
                        self.logger.info("[PerfOpt] ✓ Levenshtein 加速补丁已应用")
                except Exception as e:
                    self.logger.error(f"[PerfOpt] Levenshtein 加速补丁失败: {e}")
            else:
                self.logger.info("[PerfOpt] Levenshtein 加速补丁已禁用")

            # 图片描述批量查询替换（image_desc_bulk_lookup）
            if enable_image_desc_bulk_lookup:
                try:
                    img_desc_module = _load_local_module(
                        "components/modules/image_desc_bulk_lookup.py",
                        "CM_perf_opt_image_desc_bulk_lookup",
                    )
                    apply_image_desc_bulk_lookup = getattr(
                        img_desc_module, "apply_image_desc_bulk_lookup", None
                    )
                    if apply_image_desc_bulk_lookup:
                        apply_image_desc_bulk_lookup(self.cache_manager)
                        self.logger.info("[PerfOpt] ✓ 图片描述批量替换补丁已应用")
                except Exception as e:
                    self.logger.error(f"[PerfOpt] 图片描述批量替换补丁失败: {e}")
            else:
                self.logger.info("[PerfOpt] 图片描述批量替换补丁已禁用")

            # typo_generator_cache（错别字生成器：pinyin_dict 持久化缓存 + jieba valid_words 内存缓存）
            # 注意：该模块不依赖其他缓存模块，但建议在基础模块之后加载。
            if enable_typo_generator_cache:
                try:
                    typo_cache_module = _load_local_module(
                        "components/modules/typo_generator_cache.py",
                        "CM_perf_opt_typo_generator_cache",
                    )
                    apply_typo_generator_cache = getattr(
                        typo_cache_module, "apply_typo_generator_cache", None
                    )
                    if apply_typo_generator_cache:
                        apply_typo_generator_cache(self.cache_manager)
                        self.logger.info("[PerfOpt] ✓ 错别字生成器缓存补丁已应用")
                except Exception as e:
                    self.logger.error(f"[PerfOpt] 错别字生成器缓存补丁失败: {e}")
            else:
                self.logger.info("[PerfOpt] 错别字生成器缓存补丁已禁用")
 
            # asyncio 事件循环池（thread-local，默认关闭，高风险）
            if enable_asyncio_loop_pool:
                try:
                    loop_pool_module = _load_local_module(
                        "components/modules/asyncio_loop_pool.py",
                        "CM_perf_opt_asyncio_loop_pool",
                    )
                    apply_asyncio_loop_pool = getattr(
                        loop_pool_module, "apply_asyncio_loop_pool", None
                    )
                    if apply_asyncio_loop_pool:
                        apply_asyncio_loop_pool(self.cache_manager)
                        self.logger.info(
                            "[PerfOpt] ✓ asyncio_loop_pool 已启用（thread-local loop）"
                        )
                except Exception as e:
                    self.logger.error(f"[PerfOpt] asyncio_loop_pool 启用失败: {e}")
            else:
                self.logger.info("[PerfOpt] asyncio_loop_pool 已禁用（默认关闭）")
 
            # PatchChain 摘要日志（展示冲突链）
            if hasattr(self, "_patch_chain") and self._patch_chain is not None:
                try:
                    summary = self._patch_chain.summary()
                    if summary:
                        self.logger.info(
                            "[PerfOpt] PatchChain 摘要: %s",
                            {k: v for k, v in summary.items()},
                        )
                        for func_id, modules in summary.items():
                            if len(modules) > 1:
                                self.logger.warning(
                                    "[PerfOpt] ⚠️ 链式 patch: %s <- %s",
                                    func_id,
                                    " -> ".join(modules),
                                )
                except Exception:
                    pass

            self.patches_applied = True
            self.logger.info("[PerfOpt] ✓ 所有性能优化补丁应用完成")

        except Exception as e:
            self.logger.error(f"[PerfOpt] 补丁应用失败: {e}")
            raise

    def start(self):
        """启动优化器"""
        if self.started:
            self.logger.debug("[PerfOpt] 优化器已启动，跳过")
            return

        try:
            self.logger.info("[PerfOpt] 启动性能优化器...")

            # 启动所有缓存
            for name, cache in self.cache_manager.caches.items():
                try:
                    if hasattr(cache, "start"):
                        cache.start()
                        self.logger.debug(f"[PerfOpt] 缓存 {name} 已启动")
                except Exception as e:
                    self.logger.error(f"[PerfOpt] 启动缓存 {name} 失败: {e}")

            # 注册内存监控回调
            self._register_cache_memory_callbacks()

            # 注册统计回调
            self._register_stats_callbacks()

            # 启动监控
            enable_memory_monitor = True
            enable_stats = True

            if self._config_manager:
                enable_memory_monitor = self._config_manager.get(
                    "monitoring.enable_memory_monitor", True
                )
                enable_stats = self._config_manager.get("monitoring.enable_stats", True)

            if enable_memory_monitor and self._memory_monitor:
                try:
                    self._memory_monitor.start()
                    self.logger.info("[PerfOpt] ✓ 内存监控已启动")
                except Exception as e:
                    self.logger.warning(f"[PerfOpt] 内存监控启动失败: {e}")

            if enable_stats and self._stats_reporter:
                try:
                    self._stats_reporter.start()
                    self.logger.info("[PerfOpt] ✓ 统计报告已启动")
                except Exception as e:
                    self.logger.warning(f"[PerfOpt] 统计报告启动失败: {e}")

            # 初始化通知系统
            self._init_notification_system()

            self.started = True
            self._log_startup_info()
            self.logger.info("[PerfOpt] ✓ 性能优化器启动完成")

        except Exception as e:
            self.logger.error(f"[PerfOpt] 启动失败: {e}")
            raise

    def _log_startup_info(self):
        """记录启动信息"""
        info_lines = [
            f"[PerfOpt] 插件版本: {PLUGIN_VERSION}",
            f"[PerfOpt] aiofiles: {'可用' if AIOFILES_AVAILABLE else '不可用'}",
            f"[PerfOpt] orjson: {'可用' if ORJSON_AVAILABLE else '不可用'}",
            f"[PerfOpt] psutil: {'可用' if PSUTIL_AVAILABLE else '不可用'}",
            f"[PerfOpt] 已加载缓存模块: {list(self.cache_manager.caches.keys())}",
        ]
        for line in info_lines:
            self.logger.info(line)

    def stop(self):
        """停止优化器"""
        if not self.started:
            self.logger.debug("[PerfOpt] 优化器未启动，跳过停止")
            return

        try:
            self.logger.info("[PerfOpt] 停止性能优化器...")

            # 停止监控
            if self._memory_monitor:
                self._memory_monitor.stop()
            if self._stats_reporter:
                self._stats_reporter.stop()

            # 关闭日志处理器
            if self._log_handler:
                try:
                    from .core import shutdown_log_handler
                    shutdown_log_handler()
                    self.logger.debug("[PerfOpt] 日志处理器已关闭")
                except Exception as e:
                    self.logger.warning(f"[PerfOpt] 关闭日志处理器失败: {e}")

            # 停止所有缓存
            self.cache_manager.stop_all()

            # BUG FIX: 统一回滚所有模块的 monkey-patch
            # 遍历所有缓存模块，调用 remove_patch() 方法
            for name, cache in list(self.cache_manager.caches.items()):
                try:
                    if hasattr(cache, "remove_patch") and callable(getattr(cache, "remove_patch")):
                        cache.remove_patch()
                        self.logger.debug(f"[PerfOpt] 已回滚 {name} 的补丁")
                except Exception as e:
                    self.logger.warning(f"[PerfOpt] 回滚 {name} 补丁失败: {e}")

            # 清除所有缓存
            self.cache_manager.clear_all()

            self.started = False
            self.patches_applied = False
            self.logger.info("[PerfOpt] ✓ 性能优化器已停止")

        except Exception as e:
            self.logger.error(f"[PerfOpt] 停止失败: {e}")

    def cleanup(self):
        """清理资源"""
        try:
            self.logger.info("[PerfOpt] 清理性能优化器资源...")
            self.cache_manager.clear_all()
            self.logger.info("[PerfOpt] ✓ 资源清理完成")
        except Exception as e:
            self.logger.error(f"[PerfOpt] 资源清理失败: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if self._stats_reporter:
            return self._stats_reporter.collect_stats()
        return {}

    def get_memory_usage(self) -> Dict[str, int]:
        """获取内存使用信息"""
        return self.cache_manager.get_memory_usage()


def _cleanup_optimizer(opt: _PerformanceOptimizer, reason: str):
    """安全清理优化器"""
    try:
        logger.warning(f"[PerfOpt] 由于 {reason} 进行清理...")
        opt.stop()
        opt.cleanup()
    except Exception as e:
        logger.error(f"[PerfOpt] 清理失败: {e}")


# 插件主类
@register_plugin
class CMPerformanceOptimizerPlugin(BasePlugin):
    """CM 性能优化插件

    提供多种缓存机制提升 MaiBot 性能，包括：
    - 消息缓存（热集 + 查询缓存）
    - 人物信息缓存
    - 表达式缓存（双缓冲）
    - 黑话缓存（双缓冲 + 内容索引）
    - 知识图谱缓存（双缓冲）

    配置系统特性：
    - 支持热更新
    - 向后兼容
    - 详细验证
    """

    plugin_name: str = "CM-performance-optimizer"  # type: ignore[assignment]
    plugin_description = "CM 性能优化插件，提供多种缓存机制提升 MaiBot 性能"
    plugin_version = PLUGIN_VERSION
    plugin_author = "城陌"
    plugin_type = "performance"

    # PluginBase 抽象属性 - 直接定义为类属性
    enable_plugin: bool = True  # type: ignore[assignment]
    dependencies: List[str] = []  # type: ignore[assignment]
    config_file_name: str = "config.toml"  # type: ignore[assignment]

    # 配置节描述 - 每个模块独立section
    config_section_descriptions = {  # type: ignore[assignment]
        "plugin": ConfigSection(
            title="插件设置",
            description="插件的基础配置",
            icon="🔧",
            collapsed=False,
            order=0,
        ),
        "modules": ConfigSection(
            title="功能模块开关",
            description="选择要启用的性能优化功能模块",
            icon="⚡",
            collapsed=False,
            order=1,
        ),
        "message_cache": ConfigSection(
            title="消息缓存配置",
            description="消息热集缓存，加速消息查询",
            icon="💬",
            collapsed=True,
            order=2,
        ),
        "person_cache": ConfigSection(
            title="人物缓存配置",
            description="人物信息缓存，减少数据库查询",
            icon="👤",
            collapsed=True,
            order=3,
        ),
        "expression_cache": ConfigSection(
            title="表达式缓存配置",
            description="表达式缓存，加速表达式匹配",
            icon="📝",
            collapsed=True,
            order=4,
        ),
        "jargon_cache": ConfigSection(
            title="黑话缓存配置",
            description="黑话缓存，加速黑话解析",
            icon="📖",
            collapsed=True,
            order=5,
        ),
        "kg_cache": ConfigSection(
            title="知识图谱缓存配置",
            description="知识图谱缓存，加速知识检索",
            icon="🧠",
            collapsed=True,
            order=6,
        ),
        "db_tuning": ConfigSection(
            title="数据库调优配置",
            description="SQLite数据库性能优化参数",
            icon="🗄️",
            collapsed=True,
            order=7,
        ),
        "lightweight_profiler": ConfigSection(
            title="性能剖析配置",
            description="轻量性能剖析器设置",
            icon="🔬",
            collapsed=True,
            order=8,
        ),
        "advanced": ConfigSection(
            title="高级设置",
            description="异步IO、JSON加速等高级选项",
            icon="⚙️",
            collapsed=True,
            order=9,
        ),
        "monitoring": ConfigSection(
            title="监控设置",
            description="统计报告和内存监控配置",
            icon="📊",
            collapsed=True,
            order=10,
        ),
        "notification": ConfigSection(
            title="通知设置",
            description="QQ通知和控制台通知配置",
            icon="🔔",
            collapsed=True,
            order=11,
        ),
    }

    # 布局配置 - 使用标签页布局
    config_layout = ConfigLayout(  # type: ignore[assignment]
        type="tabs",
        tabs=[
            ConfigTab(
                id="plugin",
                title="插件",
                icon="🔧",
                sections=["plugin"],
                order=0,
            ),
            ConfigTab(
                id="modules",
                title="模块开关",
                icon="⚡",
                sections=["modules"],
                order=1,
            ),
            ConfigTab(
                id="message_cache",
                title="消息缓存",
                icon="💬",
                sections=["message_cache"],
                order=2,
            ),
            ConfigTab(
                id="person_cache",
                title="人物缓存",
                icon="👤",
                sections=["person_cache"],
                order=3,
            ),
            ConfigTab(
                id="expression_cache",
                title="表达式缓存",
                icon="📝",
                sections=["expression_cache"],
                order=4,
            ),
            ConfigTab(
                id="jargon_cache",
                title="黑话缓存",
                icon="📖",
                sections=["jargon_cache"],
                order=5,
            ),
            ConfigTab(
                id="kg_cache",
                title="知识图谱缓存",
                icon="🧠",
                sections=["kg_cache"],
                order=6,
            ),
            ConfigTab(
                id="db_tuning",
                title="数据库调优",
                icon="🗄️",
                sections=["db_tuning"],
                order=7,
            ),
            ConfigTab(
                id="lightweight_profiler",
                title="性能剖析",
                icon="🔬",
                sections=["lightweight_profiler"],
                order=8,
            ),
            ConfigTab(
                id="advanced",
                title="高级",
                icon="⚙️",
                sections=["advanced"],
                order=9,
            ),
            ConfigTab(
                id="monitoring",
                title="监控",
                icon="📊",
                sections=["monitoring"],
                order=10,
            ),
            ConfigTab(
                id="notification",
                title="通知",
                icon="🔔",
                sections=["notification"],
                order=11,
            ),
        ],
    )

    # 配置Schema定义 - 与config.toml结构匹配
    config_schema = {  # type: ignore[assignment]
        "plugin": {
            "enabled": ConfigField(
                type=bool, default=True, description="是否启用插件"
            ),
            "config_version": ConfigField(
                type=str, default=CONFIG_VERSION, description="配置文件版本"
            ),
            "log_level": ConfigField(
                type=str, default="INFO", description="日志级别"
            ),
        },
        "modules": {
            "message_cache_enabled": ConfigField(
                type=bool, default=True, description="是否启用消息缓存"
            ),
            "message_repository_fastpath_enabled": ConfigField(
                type=bool,
                default=True,
                description="是否启用消息仓库快速路径",
            ),
            "person_cache_enabled": ConfigField(
                type=bool, default=True, description="是否启用人物信息缓存"
            ),
            "expression_cache_enabled": ConfigField(
                type=bool, default=True, description="是否启用表达式缓存"
            ),
            "jargon_cache_enabled": ConfigField(
                type=bool, default=True, description="是否启用黑话缓存"
            ),
            "jargon_matcher_automaton_enabled": ConfigField(
                type=bool,
                default=True,
                description="是否启用黑话匹配自动机加速",
            ),
            "kg_cache_enabled": ConfigField(
                type=bool, default=True, description="是否启用知识图谱缓存"
            ),
            "levenshtein_fast_enabled": ConfigField(
                type=bool, default=True, description="是否启用Levenshtein距离加速"
            ),
            "image_desc_bulk_lookup_enabled": ConfigField(
                type=bool, default=True, description="是否启用图片描述批量替换"
            ),
            "user_reference_batch_resolve_enabled": ConfigField(
                type=bool, default=True, description="是否启用用户引用批量解析"
            ),
            "regex_precompile_enabled": ConfigField(
                type=bool, default=True, description="是否启用正则预编译"
            ),
            "typo_generator_cache_enabled": ConfigField(
                type=bool, default=True, description="是否启用typo_generator缓存"
            ),
            "db_tuning_enabled": ConfigField(
                type=bool,
                default=True,
                description="是否启用SQLite数据库调优",
            ),
            "lightweight_profiler_enabled": ConfigField(
                type=bool, default=False, description="是否启用轻量性能剖析"
            ),
            "asyncio_loop_pool_enabled": ConfigField(
                type=bool, default=True, description="是否启用asyncio_loop_pool"
            ),
        },
        "message_cache": {
            "per_chat_limit": ConfigField(
                type=int, default=200, description="每个聊天的缓存消息数量 (50-1000)"
            ),
            "ttl": ConfigField(
                type=int, default=300, description="缓存过期时间(秒) (60-3600)"
            ),
            "max_chats": ConfigField(
                type=int, default=500, description="最大缓存聊天数 (100-2000)"
            ),
            "mode": ConfigField(
                type=str, default="query", description="缓存模式: query或full"
            ),
            "ignore_time_limit_when_active": ConfigField(
                type=bool, default=True, description="活跃聊天流是否忽略TTL限制"
            ),
            "active_time_window": ConfigField(
                type=int, default=300, description="活跃时间窗口(秒) (60-1800)"
            ),
            "bucket_enabled": ConfigField(
                type=bool, default=False, description="滑动窗口分桶功能(预留)"
            ),
            "bucket_seconds": ConfigField(
                type=int, default=5, description="分桶时间间隔(秒)"
            ),
        },
        "person_cache": {
            "max_size": ConfigField(
                type=int, default=3000, description="最大缓存条目数 (500-10000)"
            ),
            "ttl": ConfigField(
                type=int, default=1800, description="缓存过期时间(秒) (300-7200)"
            ),
            "warmup_enabled": ConfigField(
                type=bool, default=True, description="是否启用预热功能"
            ),
            "warmup_per_chat_sample": ConfigField(
                type=int, default=30, description="预热时每聊天采样消息数 (10-100)"
            ),
            "warmup_max_persons": ConfigField(
                type=int, default=20, description="每聊天最多预热人数 (5-50)"
            ),
            "warmup_ttl": ConfigField(
                type=int, default=120, description="预热记录过期时间(秒) (60-300)"
            ),
            "warmup_debounce_seconds": ConfigField(
                type=float, default=3.0, description="预热防抖时间(秒) (1.0-10.0)"
            ),
        },
        "expression_cache": {
            "batch_size": ConfigField(
                type=int, default=100, description="批量处理大小 (10-500)"
            ),
            "batch_delay": ConfigField(
                type=float, default=0.05, description="批量处理延迟(秒) (0.01-1.0)"
            ),
            "refresh_interval": ConfigField(
                type=int, default=3600, description="刷新间隔(秒) (600-86400)"
            ),
            "incremental_refresh_interval": ConfigField(
                type=int, default=600, description="增量刷新间隔(秒) (60-3600)"
            ),
            "incremental_threshold_ratio": ConfigField(
                type=float, default=0.1, description="增量刷新阈值比例 (0.05-0.5)"
            ),
            "full_rebuild_interval": ConfigField(
                type=int, default=86400, description="完全重建间隔(秒) (3600-604800)"
            ),
            "deletion_check_interval": ConfigField(
                type=int, default=10, description="删除检查间隔(秒) (5-100)"
            ),
        },
        "jargon_cache": {
            "batch_size": ConfigField(
                type=int, default=100, description="批量处理大小 (10-500)"
            ),
            "batch_delay": ConfigField(
                type=float, default=0.05, description="批量处理延迟(秒) (0.01-1.0)"
            ),
            "refresh_interval": ConfigField(
                type=int, default=3600, description="刷新间隔(秒) (600-86400)"
            ),
            "enable_content_index": ConfigField(
                type=bool, default=True, description="是否启用内容索引"
            ),
            "incremental_refresh_interval": ConfigField(
                type=int, default=600, description="增量刷新间隔(秒) (60-3600)"
            ),
            "incremental_threshold_ratio": ConfigField(
                type=float, default=0.1, description="增量刷新阈值比例 (0.05-0.5)"
            ),
            "full_rebuild_interval": ConfigField(
                type=int, default=86400, description="完全重建间隔(秒) (3600-604800)"
            ),
            "deletion_check_interval": ConfigField(
                type=int, default=10, description="删除检查间隔(秒) (5-100)"
            ),
        },
        "kg_cache": {
            "batch_size": ConfigField(
                type=int, default=100, description="批量处理大小 (10-500)"
            ),
            "batch_delay": ConfigField(
                type=float, default=0.05, description="批量处理延迟(秒) (0.01-1.0)"
            ),
            "refresh_interval": ConfigField(
                type=int, default=3600, description="刷新间隔(秒) (600-86400)"
            ),
            "incremental_refresh_interval": ConfigField(
                type=int, default=600, description="增量刷新间隔(秒) (60-3600)"
            ),
            "incremental_threshold_ratio": ConfigField(
                type=float, default=0.1, description="增量刷新阈值比例 (0.05-0.5)"
            ),
            "full_rebuild_interval": ConfigField(
                type=int, default=86400, description="完全重建间隔(秒) (3600-604800)"
            ),
            "deletion_check_interval": ConfigField(
                type=int, default=10, description="删除检查间隔(秒) (5-100)"
            ),
            "use_parquet": ConfigField(
                type=bool, default=True, description="是否使用Parquet格式"
            ),
        },
        "db_tuning": {
            "mmap_size": ConfigField(
                type=int,
                default=268435456,
                description="SQLite mmap_size(字节,0=禁用)",
            ),
            "wal_checkpoint_interval": ConfigField(
                type=int,
                default=300,
                description="WAL checkpoint周期(秒,0=禁用)",
            ),
        },
        "lightweight_profiler": {
            "sample_rate": ConfigField(
                type=float, default=0.1, description="采样率(0-1)"
            ),
        },
        "advanced": {
            "enable_async_io": ConfigField(
                type=bool, default=True, description="是否启用异步IO"
            ),
            "enable_orjson": ConfigField(
                type=bool, default=True, description="是否启用orjson"
            ),
            "enable_hot_reload": ConfigField(
                type=bool, default=True, description="是否启用配置热重载"
            ),
            "strict_validation": ConfigField(
                type=bool, default=False, description="是否启用严格验证"
            ),
            "enable_change_notifications": ConfigField(
                type=bool, default=True, description="是否启用配置变更通知"
            ),
        },
        "monitoring": {
            "enable_stats": ConfigField(
                type=bool, default=True, description="是否启用统计"
            ),
            "stats_interval": ConfigField(
                type=int, default=60, description="统计间隔(秒) (10-3600)"
            ),
            "enable_memory_monitor": ConfigField(
                type=bool, default=True, description="是否启用内存监控"
            ),
            "memory_warning_threshold": ConfigField(
                type=float, default=0.8, description="内存警告阈值(0-1)"
            ),
            "memory_critical_threshold": ConfigField(
                type=float, default=0.9, description="内存严重阈值(0-1)"
            ),
            "enable_health_check": ConfigField(
                type=bool, default=True, description="是否启用健康检查"
            ),
            "health_check_interval": ConfigField(
                type=int, default=30, description="健康检查间隔(秒) (10-300)"
            ),
        },
        "notification": {
            "enabled": ConfigField(
                type=bool, default=True, description="启用通知功能"
            ),
            "admin_qq": ConfigField(
                type=str, default="", description="接收通知的QQ号（留空则不发送QQ通知）"
            ),
        },
    }

    # 依赖检查
    python_dependencies = [  # type: ignore[assignment]
        PythonDependency(
            package_name="aiofiles",
            version=">=0.8.0",
            optional=True,
            description="异步文件操作",
        ),
        PythonDependency(
            package_name="orjson",
            version=">=3.8.0",
            optional=True,
            description="高性能 JSON 处理",
        ),
        PythonDependency(
            package_name="psutil",
            version=">=5.9.0",
            optional=True,
            description="系统资源监控",
        ),
    ]

    def __init__(self, plugin_dir=None):
        super().__init__(plugin_dir)
        self.log_prefix = "[PerfOpt]"
        self._opt: Optional[_PerformanceOptimizer] = None
        self._started = False
        self._degraded = False
        self._degraded_reason: Optional[str] = None
        # 设置全局插件实例，供事件处理器使用
        global _plugin_instance
        _plugin_instance = self

    async def activate(self, ctx):
        """激活插件"""
        logger.info("[PerfOpt] 插件激活中...")
        # 插件激活时不执行任何操作，等待 ON_START 事件
        logger.info("[PerfOpt] ✓ 插件已激活，等待启动事件")

    async def deactivate(self, ctx):
        """停用插件"""
        logger.info("[PerfOpt] 插件停用中...")
        if self._opt:
            _cleanup_optimizer(self._opt, "plugin deactivate")
            self._opt = None
        self._started = False
        logger.info("[PerfOpt] ✓ 插件已停用")

    async def _apply_patches_and_start(self):
        """应用补丁并启动优化器"""
        if self._started:
            logger.debug("[PerfOpt] 优化器已启动，跳过")
            return

        try:
            # 创建或获取优化器单例
            self._opt = _PerformanceOptimizer()
            self._opt.set_plugin_instance(self)

            # 应用补丁
            # 全量模式：应用所有补丁
            if not self._degraded:
                try:
                    self._opt.apply_patches()
                except Exception as e:
                    logger.error(f"[PerfOpt] 全量模式补丁失败: {e}")
                    logger.warning("[PerfOpt] 全量模式补丁失败，插件将以降级模式运行")
                    # 设置降级标志
                    self._degraded = True
                    self._degraded_reason = "全量模式补丁失败"
                    # 清理已创建的优化器
                    if self._opt:
                        _cleanup_optimizer(self._opt, "full mode patch failure")
                        self._opt = None
                    return

            self._opt.start()
            self._started = True
            logger.info("[PerfOpt] ✓ 插件启动完成")
        except Exception as e:
            logger.error(f"[PerfOpt] 启动失败: {e}")
            # Best-effort rollback
            if self._opt:
                _cleanup_optimizer(self._opt, "startup failure")
                self._opt = None
            # 不阻止插件加载，记录错误并继续
            logger.warning("[PerfOpt] 插件将以降级模式运行")
            # 设置降级标志
            self._degraded = True
            self._degraded_reason = str(e)

    def get_plugin_components(self):
        """返回插件组件列表"""
        components = []

        # 动态加载启动事件处理器
        try:
            start_handler_module = _load_local_module(
                "components/handlers/start_handler.py", "CM_perf_opt_start_handler"
            )
            # 加载模块后立即注入实例
            start_handler_module._plugin_instance = self

            PerfOptStartHandler = getattr(
                start_handler_module, "PerfOptStartHandler", None
            )
            if PerfOptStartHandler:
                # 返回处理器类（不是实例），插件系统会自动实例化
                components.append(
                    (PerfOptStartHandler.get_handler_info(), PerfOptStartHandler)
                )
        except Exception as e:
            logger.error(f"[PerfOpt] 加载启动事件处理器失败: {e}")

        # 动态加载停止事件处理器（始终添加，确保插件停止时正确回滚）
        try:
            stop_handler_module = _load_local_module(
                "components/handlers/stop_handler.py", "CM_perf_opt_stop_handler"
            )
            # 加载模块后立即注入实例
            stop_handler_module._plugin_instance = self

            PerfOptStopHandler = getattr(
                stop_handler_module, "PerfOptStopHandler", None
            )
            if PerfOptStopHandler:
                # 返回处理器类（不是实例），插件系统会自动实例化
                components.append(
                    (PerfOptStopHandler.get_handler_info(), PerfOptStopHandler)
                )
        except Exception as e:
            logger.error(f"[PerfOpt] 加载停止事件处理器失败: {e}")

        return components

    # 公开 API
    def get_stats(self) -> Dict[str, Any]:
        """获取插件统计信息

        Returns:
            包含各模块统计信息的字典
        """
        if self._opt:
            return self._opt.get_stats()
        return {}

    def get_memory_usage(self) -> Dict[str, int]:
        """获取内存使用信息

        Returns:
            包含各缓存模块内存使用（字节）的字典
        """
        if self._opt:
            return self._opt.get_memory_usage()
        return {}

    def is_degraded(self) -> Tuple[bool, Optional[str]]:
        """检查是否处于降级模式

        Returns:
            Tuple[是否降级, ���级原因]
        """
        return self._degraded, self._degraded_reason

    async def reload_config(self) -> bool:
        """重新加载配置（需重启生效）

        注意：配置修改后需要重启 MaiBot 才能生效。
        此方法仅用于测试配置加载是否正常。

        Returns:
            是否重载成功
        """
        if self._opt and self._opt._config_manager:
            try:
                self._opt._config_manager.load()
                logger.warning("[PerfOpt] 配置已重新加载，重启后生效")
                return True
            except Exception as e:
                logger.error(f"[PerfOpt] 配置加载失败: {e}")
                return False
        return False

    def get_config(self, path: str, default: Any = None) -> Any:
        """获取配置值

        Args:
            path: 配置路径，如 "modules.message_cache_enabled"
            default: 默认值

        Returns:
            配置值
        """
        if self._opt and self._opt._config_manager:
            return self._opt._config_manager.get(path, default)
        return default

    def set_config(self, path: str, value: Any) -> bool:
        """设置配置值（支持热更新的配置项）

        Args:
            path: 配置路径
            value: 新值

        Returns:
            是否设置成功
        """
        if self._opt and self._opt._config_manager:
            return self._opt._config_manager.set(path, value)
        return False
