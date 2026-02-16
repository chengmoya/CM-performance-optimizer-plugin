"""
模块配置映射 - 配置到缓存模块的映射和热更新支持

功能：
- 配置项到模块参数的映射
- 热更新分发
- 模块配置应用
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ConfigManager

try:
    from src.common.logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger("CM_perf_opt")


class ModuleConfigMapper:
    """模块配置映射器

    负责将配置系统中的配置项映射到具体模块的参数，
    并支持热更新分发。
    """

    _instance: Optional["ModuleConfigMapper"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
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

        self._module_instances: Dict[str, Any] = {}
        self._update_callbacks: Dict[str, List[Callable[[str, Any], None]]] = {}
        self._mapping_lock = threading.RLock()

        # 配置路径到模块参数的映射定义
        self._config_mappings = self._build_mappings()

    def _build_mappings(self) -> Dict[str, Dict[str, str]]:
        """构建配置路径到模块属性的映射

        格式: {配置路径: {模块名: 属性名}}
        """
        return {
            # 消息缓存模块配置
            "modules.message_cache.enabled": {"message_hotset": "enabled"},
            "modules.message_cache.per_chat_limit": {"message_hotset": "per_chat_limit"},
            "modules.message_cache.ttl": {"message_hotset": "ttl"},
            "modules.message_cache.max_chats": {"message_hotset": "max_chats"},
            "modules.message_cache.ignore_time_limit_when_active": {
                "message_hotset": "ignore_time_limit_when_active"
            },
            "modules.message_cache.active_time_window": {"message_hotset": "active_time_window"},
            # 人物缓存模块配置
            "modules.person_cache.max_size": {"person_cache": "cache.max_size"},
            "modules.person_cache.ttl": {"person_cache": "cache.ttl"},
            "modules.person_cache.warmup_enabled": {"person_warmup": "enabled"},
            "modules.person_cache.warmup_per_chat_sample": {
                "person_warmup": "per_chat_message_sample"
            },
            "modules.person_cache.warmup_max_persons": {"person_warmup": "max_persons_per_chat"},
            "modules.person_cache.warmup_ttl": {"person_warmup": "ttl"},
            "modules.person_cache.warmup_debounce": {"person_warmup": "debounce_seconds"},
            # 表达式缓存模块配置
            "modules.expression_cache.refresh_interval": {"expression_cache": "refresh_interval"},
            # 黑话缓存模块配置
            "modules.jargon_cache.refresh_interval": {"jargon_cache": "refresh_interval"},
            # 知识图谱缓存模块配置
            "modules.kg_cache.refresh_interval": {"kg_cache": "refresh_interval"},
            # 监控配置
            "monitoring.stats_interval": {"stats_reporter": "report_interval"},
            "monitoring.memory_warning_threshold": {"memory_monitor": "warning_threshold"},
            "monitoring.memory_critical_threshold": {"memory_monitor": "critical_threshold"},
            # Lightweight profiler 配置
            "performance.profiler_sample_rate": {"lightweight_profiler": "sample_rate"},
        }

    def register_module(self, name: str, instance: Any):
        """注册模块实例

        Args:
            name: 模块名称
            instance: 模块实例
        """
        with self._mapping_lock:
            self._module_instances[name] = instance
            logger.debug(f"[ModuleConfigMapper] 注册模块: {name}")

    def unregister_module(self, name: str):
        """取消注册模块"""
        with self._mapping_lock:
            self._module_instances.pop(name, None)

    def register_update_callback(
        self, module_name: str, callback: Callable[[str, Any], None]
    ):
        """注册配置更新回调

        Args:
            module_name: 模块名称
            callback: 回调函数 (属性路径, 新值)
        """
        with self._mapping_lock:
            if module_name not in self._update_callbacks:
                self._update_callbacks[module_name] = []
            self._update_callbacks[module_name].append(callback)

    def apply_config(self, config_manager: "ConfigManager"):
        """应用完整配置到所有已注册模块

        Args:
            config_manager: 配置管理器实例
        """
        with self._mapping_lock:
            for config_path, module_mappings in self._config_mappings.items():
                value = config_manager.get(config_path)
                if value is None:
                    continue

                for module_name, attr_path in module_mappings.items():
                    self._apply_to_module(module_name, attr_path, value)

    def _apply_to_module(self, module_name: str, attr_path: str, value: Any):
        """应用配置到模块

        Args:
            module_name: 模块名称
            attr_path: 属性路径（支持点分隔的嵌套路径，如 "cache.max_size"）
            value: 配置值
        """
        instance = self._module_instances.get(module_name)
        if instance is None:
            return

        try:
            parts = attr_path.split(".")
            target = instance

            # 遍历到倒数第二级
            for part in parts[:-1]:
                target = getattr(target, part, None)
                if target is None:
                    logger.warning(
                        f"[ModuleConfigMapper] 无法访问属性路径: {module_name}.{attr_path}"
                    )
                    return

            # 设置最终属性
            final_attr = parts[-1]
            if hasattr(target, final_attr):
                setattr(target, final_attr, value)
                logger.debug(
                    f"[ModuleConfigMapper] 已应用配置: {module_name}.{attr_path} = {value}"
                )
            else:
                logger.warning(
                    f"[ModuleConfigMapper] 属性不存在: {module_name}.{attr_path}"
                )
        except Exception as e:
            logger.error(f"[ModuleConfigMapper] 应用配置失败: {module_name}.{attr_path}: {e}")

    def on_config_change(self, path: str, old_value: Any, new_value: Any):
        """配置变更回调

        用于热更新时分发配置变更

        Args:
            path: 配置路径
            old_value: 旧值
            new_value: 新值
        """
        with self._mapping_lock:
            if path not in self._config_mappings:
                return

            module_mappings = self._config_mappings[path]
            for module_name, attr_path in module_mappings.items():
                # 应用到模块
                self._apply_to_module(module_name, attr_path, new_value)

                # 调用回调
                callbacks = self._update_callbacks.get(module_name, [])
                for callback in callbacks:
                    try:
                        callback(attr_path, new_value)
                    except Exception as e:
                        logger.error(
                            f"[ModuleConfigMapper] 回调执行失败: {module_name}: {e}"
                        )

    def get_module_config(self, module_name: str) -> Dict[str, Any]:
        """获取模块的当前配置

        Args:
            module_name: 模块名称

        Returns:
            模块配置字典
        """
        config: Dict[str, Any] = {}

        instance = self._module_instances.get(module_name)
        if instance is None:
            return config

        # 找出所有映射到该模块的配置项
        for config_path, module_mappings in self._config_mappings.items():
            if module_name in module_mappings:
                attr_path = module_mappings[module_name]
                try:
                    parts = attr_path.split(".")
                    target = instance
                    for part in parts:
                        target = getattr(target, part, None)
                        if target is None:
                            break
                    if target is not None:
                        config[attr_path] = target
                except Exception:
                    pass

        return config


class ModuleEnabler:
    """模块启用控制器

    根据配置控制模块的启用和禁用
    """

    def __init__(self):
        self._module_states: Dict[str, bool] = {}
        self._enable_callbacks: Dict[str, Callable[[], None]] = {}
        self._disable_callbacks: Dict[str, Callable[[], None]] = {}
        self._lock = threading.Lock()

    def register_callbacks(
        self,
        module_name: str,
        enable_callback: Callable[[], None],
        disable_callback: Callable[[], None],
    ):
        """注册模块启用/禁用回调

        Args:
            module_name: 模块名称
            enable_callback: 启用回调
            disable_callback: 禁用回调
        """
        with self._lock:
            self._enable_callbacks[module_name] = enable_callback
            self._disable_callbacks[module_name] = disable_callback

    def set_enabled(self, module_name: str, enabled: bool):
        """设置模块启用状态

        Args:
            module_name: 模块名称
            enabled: 是否启用
        """
        with self._lock:
            current = self._module_states.get(module_name, True)
            if current == enabled:
                return

            self._module_states[module_name] = enabled

            try:
                if enabled:
                    callback = self._enable_callbacks.get(module_name)
                    if callback:
                        callback()
                        logger.info(f"[ModuleEnabler] 已启用模块: {module_name}")
                else:
                    callback = self._disable_callbacks.get(module_name)
                    if callback:
                        callback()
                        logger.info(f"[ModuleEnabler] 已禁用模块: {module_name}")
            except Exception as e:
                logger.error(f"[ModuleEnabler] 切换模块状态失败: {module_name}: {e}")

    def is_enabled(self, module_name: str) -> bool:
        """检查模块是否启用"""
        with self._lock:
            return self._module_states.get(module_name, True)

    def apply_from_config(self, config_manager: "ConfigManager"):
        """从配置应用模块状态

        Args:
            config_manager: 配置管理器
        """
        module_enable_paths = {
            "message_cache": "modules.message_cache_enabled",
            "message_repository_fastpath": "modules.message_repository_fastpath_enabled",
            "person_cache": "modules.person_cache_enabled",
            "expression_cache": "modules.expression_cache_enabled",
            "jargon_cache": "modules.jargon_cache_enabled",
            "jargon_matcher_automaton": "modules.jargon_matcher_automaton_enabled",
            "kg_cache": "modules.kg_cache_enabled",
            "levenshtein_fast": "modules.levenshtein_fast_enabled",
            "image_desc_bulk_lookup": "modules.image_desc_bulk_lookup_enabled",
            "user_reference_batch_resolve": "modules.user_reference_batch_resolve_enabled",
            "regex_precompile": "modules.regex_precompile_enabled",
            "typo_generator_cache": "modules.typo_generator_cache_enabled",
            "lightweight_profiler": "modules.lightweight_profiler_enabled",
            "asyncio_loop_pool": "modules.asyncio_loop_pool_enabled",
            "db_tuning": "modules.db_tuning_enabled",
        }

        for module_name, config_path in module_enable_paths.items():
            enabled = config_manager.get(config_path, True)
            self.set_enabled(module_name, enabled)


# 全局实例
_module_mapper: Optional[ModuleConfigMapper] = None
_module_enabler: Optional[ModuleEnabler] = None


def get_module_mapper() -> ModuleConfigMapper:
    """获取模块配置映射器单例"""
    global _module_mapper
    if _module_mapper is None:
        _module_mapper = ModuleConfigMapper()
    return _module_mapper


def get_module_enabler() -> ModuleEnabler:
    """获取模块启用控制器单例"""
    global _module_enabler
    if _module_enabler is None:
        _module_enabler = ModuleEnabler()
    return _module_enabler
