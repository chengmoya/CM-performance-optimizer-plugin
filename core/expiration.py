"""
缓存过期管理器 - ExpirationManager

提供统一的缓存过期策略管理：
- 增量更新策略
- 全量重建决策
- 删除检测周期
- 过期时间追踪

设计目标：
- 减少全量刷新开销
- 支持多种缓存类型
- 可配置的过期策略
- 线程安全操作
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

try:
    from src.common.logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger("CM_perf_opt")

T = TypeVar("T")


class RefreshDecision(Enum):
    """刷新决策枚举"""
    SKIP = "skip"  # 跳过刷新
    INCREMENTAL = "incremental"  # 增量刷新
    FULL_REBUILD = "full_rebuild"  # 全量重建
    DELETION_CHECK = "deletion_check"  # 删除检测


@dataclass
class ExpirationConfig:
    """过期配置
    
    Attributes:
        incremental_refresh_interval: 增量刷新间隔（秒），默认 600
        full_rebuild_interval: 全量重建间隔（秒），默认 86400
        incremental_threshold_ratio: 增量阈值比例，默认 0.1
        deletion_check_interval: 删除检测间隔（每N次增量），默认 10
    """
    incremental_refresh_interval: int = 600
    full_rebuild_interval: int = 86400
    incremental_threshold_ratio: float = 0.1
    deletion_check_interval: int = 10
    
    def __post_init__(self):
        """验证配置参数"""
        self.incremental_refresh_interval = max(60, int(self.incremental_refresh_interval))
        self.full_rebuild_interval = max(3600, int(self.full_rebuild_interval))
        self.incremental_threshold_ratio = max(0.01, min(1.0, float(self.incremental_threshold_ratio)))
        self.deletion_check_interval = max(1, int(self.deletion_check_interval))
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ExpirationConfig":
        """从字典创建配置
        
        Args:
            config: 配置字典
            
        Returns:
            ExpirationConfig 实例
        """
        return cls(
            incremental_refresh_interval=config.get("incremental_refresh_interval", 600),
            full_rebuild_interval=config.get("full_rebuild_interval", 86400),
            incremental_threshold_ratio=config.get("incremental_threshold_ratio", 0.1),
            deletion_check_interval=config.get("deletion_check_interval", 10),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            配置字典
        """
        return {
            "incremental_refresh_interval": self.incremental_refresh_interval,
            "full_rebuild_interval": self.full_rebuild_interval,
            "incremental_threshold_ratio": self.incremental_threshold_ratio,
            "deletion_check_interval": self.deletion_check_interval,
        }


@dataclass
class ExpirationState:
    """过期状态追踪
    
    Attributes:
        last_full_rebuild: 上次全量重建时间戳
        last_incremental_refresh: 上次增量刷新时间戳
        incremental_refresh_count: 增量刷新计数
        total_count: 缓存总条目数
        incremental_count: 累计增量条目数
        last_max_id: 上次加载的最大ID（用于增量查询）
    """
    last_full_rebuild: float = 0.0
    last_incremental_refresh: float = 0.0
    incremental_refresh_count: int = 0
    total_count: int = 0
    incremental_count: int = 0
    last_max_id: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            状态字典
        """
        return {
            "last_full_rebuild": self.last_full_rebuild,
            "last_incremental_refresh": self.last_incremental_refresh,
            "incremental_refresh_count": self.incremental_refresh_count,
            "total_count": self.total_count,
            "incremental_count": self.incremental_count,
            "last_max_id": self.last_max_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExpirationState":
        """从字典创建状态
        
        Args:
            data: 状态字典
            
        Returns:
            ExpirationState 实例
        """
        return cls(
            last_full_rebuild=data.get("last_full_rebuild", 0.0),
            last_incremental_refresh=data.get("last_incremental_refresh", 0.0),
            incremental_refresh_count=data.get("incremental_refresh_count", 0),
            total_count=data.get("total_count", 0),
            incremental_count=data.get("incremental_count", 0),
            last_max_id=data.get("last_max_id", 0),
        )


class ExpirationManager(ABC):
    """缓存过期管理器基类
    
    提供统一的缓存过期策略管理接口，支持：
    - 增量更新决策
    - 全量重建决策
    - 删除检测周期
    - 状态追踪
    
    使用方式：
        manager = DatabaseExpirationManager(config)
        
        # 判断刷新决策
        decision = manager.get_refresh_decision(is_first_load=False)
        
        # 更新状态
        manager.record_full_rebuild(total_count=1000)
        manager.record_incremental_refresh(new_count=50)
    """
    
    def __init__(
        self,
        config: ExpirationConfig,
        name: str = "cache",
    ):
        """初始化过期管理器
        
        Args:
            config: 过期配置
            name: 缓存名称（用于日志）
        """
        self._config = config
        self._name = name
        self._state = ExpirationState()
        self._lock = threading.RLock()
        self._is_first_load = True
    
    @property
    def config(self) -> ExpirationConfig:
        """获取当前配置"""
        return self._config
    
    @property
    def state(self) -> ExpirationState:
        """获取当前状态（只读）"""
        with self._lock:
            return ExpirationState(
                last_full_rebuild=self._state.last_full_rebuild,
                last_incremental_refresh=self._state.last_incremental_refresh,
                incremental_refresh_count=self._state.incremental_refresh_count,
                total_count=self._state.total_count,
                incremental_count=self._state.incremental_count,
                last_max_id=self._state.last_max_id,
            )
    
    def update_config(self, config: ExpirationConfig) -> None:
        """更新配置
        
        Args:
            config: 新的过期配置
        """
        with self._lock:
            self._config = config
            logger.debug(f"[{self._name}] 过期配置已更新: {config.to_dict()}")
    
    def get_refresh_decision(self, is_first_load: bool = False) -> RefreshDecision:
        """获取刷新决策
        
        根据当前状态和配置，决定应该执行哪种刷新操作：
        1. 首次加载 -> 全量重建
        2. 达到全量重建间隔 -> 全量重建
        3. 累计增量超过阈值 -> 全量重建
        4. 达到删除检测周期 -> 删除检测
        5. 达到增量刷新间隔 -> 增量刷新
        6. 否则 -> 跳过
        
        Args:
            is_first_load: 是否首次加载
            
        Returns:
            刷新决策
        """
        with self._lock:
            now = time.time()
            
            # 1. 首次加载
            if is_first_load or self._is_first_load:
                logger.info(f"[{self._name}] 首次加载，执行全量重建")
                return RefreshDecision.FULL_REBUILD
            
            # 2. 达到全量重建间隔
            time_since_full_rebuild = now - self._state.last_full_rebuild
            if time_since_full_rebuild >= self._config.full_rebuild_interval:
                logger.info(
                    f"[{self._name}] 达到全量重建间隔 "
                    f"({time_since_full_rebuild:.0f}s >= {self._config.full_rebuild_interval}s)，"
                    f"触发全量重建"
                )
                return RefreshDecision.FULL_REBUILD
            
            # 3. 累计增量超过阈值
            if self._state.total_count > 0:
                ratio = self._state.incremental_count / self._state.total_count
                if ratio > self._config.incremental_threshold_ratio:
                    logger.info(
                        f"[{self._name}] 增量比例 {ratio:.2%} "
                        f"超过阈值 {self._config.incremental_threshold_ratio:.2%}，"
                        f"触发全量重建"
                    )
                    return RefreshDecision.FULL_REBUILD
            
            # 4. 检查是否需要删除检测
            if self._should_check_deletion():
                logger.debug(f"[{self._name}] 达到删除检测周期")
                return RefreshDecision.DELETION_CHECK
            
            # 5. 检查是否需要增量刷新
            time_since_incremental = now - self._state.last_incremental_refresh
            if time_since_incremental >= self._config.incremental_refresh_interval:
                logger.debug(
                    f"[{self._name}] 达到增量刷新间隔 "
                    f"({time_since_incremental:.0f}s >= {self._config.incremental_refresh_interval}s)"
                )
                return RefreshDecision.INCREMENTAL
            
            # 6. 跳过刷新
            return RefreshDecision.SKIP
    
    def _should_check_deletion(self) -> bool:
        """判断是否需要执行删除检测
        
        Returns:
            True 表示需要删除检测
        """
        return (
            self._state.incremental_refresh_count > 0
            and self._state.incremental_refresh_count % self._config.deletion_check_interval == 0
        )
    
    def record_full_rebuild(
        self,
        total_count: int,
        max_id: int = 0,
    ) -> None:
        """记录全量重建完成
        
        Args:
            total_count: 缓存总条目数
            max_id: 最大ID（用于增量查询）
        """
        with self._lock:
            now = time.time()
            self._state.last_full_rebuild = now
            self._state.last_incremental_refresh = now
            self._state.total_count = total_count
            self._state.incremental_count = 0
            self._state.incremental_refresh_count = 0
            self._state.last_max_id = max_id
            self._is_first_load = False
            
            logger.info(
                f"[{self._name}] 全量重建完成: "
                f"total_count={total_count}, max_id={max_id}"
            )
    
    def record_incremental_refresh(
        self,
        new_count: int,
        new_max_id: Optional[int] = None,
    ) -> None:
        """记录增量刷新完成
        
        Args:
            new_count: 新增条目数
            new_max_id: 新的最大ID（可选）
        """
        with self._lock:
            now = time.time()
            self._state.last_incremental_refresh = now
            self._state.incremental_refresh_count += 1
            self._state.incremental_count += new_count
            self._state.total_count += new_count
            
            if new_max_id is not None:
                self._state.last_max_id = new_max_id
            
            logger.debug(
                f"[{self._name}] 增量刷新完成: "
                f"new_count={new_count}, "
                f"累计增量={self._state.incremental_count}, "
                f"增量次数={self._state.incremental_refresh_count}"
            )
    
    def record_deletion_check(
        self,
        deleted_count: int,
        actual_count: int,
    ) -> None:
        """记录删除检测完成
        
        Args:
            deleted_count: 删除的条目数
            actual_count: 实际缓存条目数
        """
        with self._lock:
            self._state.total_count = actual_count
            
            logger.info(
                f"[{self._name}] 删除检测完成: "
                f"deleted_count={deleted_count}, actual_count={actual_count}"
            )
    
    def reset(self) -> None:
        """重置状态"""
        with self._lock:
            self._state = ExpirationState()
            self._is_first_load = True
            logger.info(f"[{self._name}] 过期管理器状态已重置")
    
    def get_state_dict(self) -> Dict[str, Any]:
        """获取状态字典
        
        Returns:
            状态字典
        """
        with self._lock:
            return self._state.to_dict()
    
    def load_state_dict(self, data: Dict[str, Any]) -> None:
        """从字典加载状态
        
        Args:
            data: 状态字典
        """
        with self._lock:
            self._state = ExpirationState.from_dict(data)
            self._is_first_load = self._state.last_full_rebuild == 0.0
            logger.debug(f"[{self._name}] 已加载状态: {data}")


class DatabaseExpirationManager(ExpirationManager):
    """数据库缓存过期管理器
    
    适用于基于数据库的缓存（如 Jargon、Expression），
    支持基于 ID 的增量查询。
    """
    
    def __init__(
        self,
        config: ExpirationConfig,
        name: str = "db_cache",
    ):
        """初始化数据库缓存过期管理器
        
        Args:
            config: 过期配置
            name: 缓存名称
        """
        super().__init__(config, name)
    
    def get_last_max_id(self) -> int:
        """获取上次加载的最大ID
        
        Returns:
            最大ID
        """
        with self._lock:
            return self._state.last_max_id
    
    def should_skip_incremental(self, current_max_id: int) -> bool:
        """判断是否可以跳过增量刷新
        
        当数据库最大ID与缓存最大ID相同时，可以跳过。
        
        Args:
            current_max_id: 当前数据库最大ID
            
        Returns:
            True 表示可以跳过
        """
        with self._lock:
            return current_max_id <= self._state.last_max_id


class FileExpirationManager(ExpirationManager):
    """文件缓存过期管理器
    
    适用于基于文件的缓存（如 KG Cache），
    支持基于文件修改时间的增量检测。
    """
    
    def __init__(
        self,
        config: ExpirationConfig,
        name: str = "file_cache",
    ):
        """初始化文件缓存过期管理器
        
        Args:
            config: 过期配置
            name: 缓存名称
        """
        super().__init__(config, name)
        self._file_trackers: Dict[str, Dict[str, Any]] = {}
    
    def track_file(
        self,
        file_path: str,
        mtime: float,
        size: int,
    ) -> bool:
        """追踪文件状态
        
        Args:
            file_path: 文件路径
            mtime: 修改时间
            size: 文件大小
            
        Returns:
            True 表示文件有变化
        """
        with self._lock:
            old_tracker = self._file_trackers.get(file_path, {})
            has_change = (
                old_tracker.get("mtime", 0) != mtime
                or old_tracker.get("size", 0) != size
            )
            
            self._file_trackers[file_path] = {
                "mtime": mtime,
                "size": size,
            }
            
            return has_change
    
    def update_file_stats(
        self,
        file_path: str,
        mtime: Optional[float] = None,
        size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """更新并返回文件统计信息 (兼容API)
        
        支持两种调用方式：
        1. update_file_stats(file_path) - 自动获取文件状态
        2. update_file_stats(file_path, mtime, size) - 手动指定状态
        
        Args:
            file_path: 文件路径
            mtime: 修改时间（可选，不提供则自动获取）
            size: 文件大小（可选，不提供则自动获取）
            
        Returns:
            包含 mtime 和 size 的字典
        """
        import os
        
        if mtime is None or size is None:
            try:
                stat = os.stat(file_path)
                mtime = stat.st_mtime
                size = stat.st_size
            except OSError:
                mtime = mtime or 0.0
                size = size or 0
        
        self.track_file(file_path, mtime, size)
        return {"mtime": mtime, "size": size}
    
    def get_file_stats(self, file_path: str) -> Optional[Dict[str, Any]]:
        """获取文件统计信息 (兼容API)
        
        Args:
            file_path: 文件路径
            
        Returns:
            包含 mtime 和 size 的字典，如果文件未被追踪则返回 None
        """
        with self._lock:
            tracker = self._file_trackers.get(file_path)
            if tracker is None:
                return None
            return {
                "mtime": tracker.get("mtime", 0.0),
                "size": tracker.get("size", 0),
            }
    
    def has_file_changed(
        self,
        file_path: str,
        current_mtime: Optional[float] = None,
        current_size: Optional[int] = None,
    ) -> bool:
        """检查单个文件是否发生变化 (兼容API)
        
        支持两种调用方式：
        1. has_file_changed(file_path) - 自动获取文件当前状态
        2. has_file_changed(file_path, current_mtime, current_size) - 手动指定状态
        
        Args:
            file_path: 文件路径
            current_mtime: 当前修改时间（可选）
            current_size: 当前文件大小（可选）
            
        Returns:
            True 表示文件有变化或未被追踪
        """
        import os
        
        with self._lock:
            if file_path not in self._file_trackers:
                return True
            
            stats = self._file_trackers[file_path]
            
            if current_mtime is None or current_size is None:
                try:
                    stat = os.stat(file_path)
                    current_mtime = stat.st_mtime
                    current_size = stat.st_size
                except OSError:
                    return True
            
            return (
                stats.get("mtime", 0) != current_mtime
                or stats.get("size", 0) != current_size
            )
    
    def has_file_changes(self, file_paths: List[str]) -> bool:
        """检查文件是否有变化
        
        Args:
            file_paths: 要检查的文件路径列表
            
        Returns:
            True 表示有文件变化
        """
        import os
        
        with self._lock:
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    continue
                    
                try:
                    stat = os.stat(file_path)
                    old_tracker = self._file_trackers.get(file_path, {})
                    
                    if (
                        old_tracker.get("mtime", 0) != stat.st_mtime
                        or old_tracker.get("size", 0) != stat.st_size
                    ):
                        return True
                except OSError:
                    continue
            
            return False
    
    def get_file_tracker(self, file_path: str) -> Dict[str, Any]:
        """获取文件追踪信息
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件追踪信息字典
        """
        with self._lock:
            return self._file_trackers.get(file_path, {}).copy()
    
    def get_all_trackers(self) -> Dict[str, Dict[str, Any]]:
        """获取所有文件追踪信息
        
        Returns:
            所有文件追踪信息
        """
        with self._lock:
            return {k: v.copy() for k, v in self._file_trackers.items()}


class CompositeExpirationManager:
    """组合过期管理器
    
    管理多个缓存的过期策略，提供统一的访问接口。
    """
    
    def __init__(self):
        """初始化组合过期管理器"""
        self._managers: Dict[str, ExpirationManager] = {}
        self._lock = threading.RLock()
    
    def register(
        self,
        name: str,
        manager: ExpirationManager,
    ) -> None:
        """注册过期管理器
        
        Args:
            name: 管理器名称
            manager: 过期管理器实例
        """
        with self._lock:
            self._managers[name] = manager
            logger.debug(f"[CompositeExpiration] 已注册管理器: {name}")
    
    def unregister(self, name: str) -> Optional[ExpirationManager]:
        """注销过期管理器
        
        Args:
            name: 管理器名称
            
        Returns:
            被注销的管理器实例，如果不存在则返回 None
        """
        with self._lock:
            manager = self._managers.pop(name, None)
            if manager:
                logger.debug(f"[CompositeExpiration] 已注销管理器: {name}")
            return manager
    
    def get(self, name: str) -> Optional[ExpirationManager]:
        """获取过期管理器
        
        Args:
            name: 管理器名称
            
        Returns:
            过期管理器实例，如果不存在则返回 None
        """
        with self._lock:
            return self._managers.get(name)
    
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """获取所有管理器的状态
        
        Returns:
            管理器名称到状态的映射
        """
        with self._lock:
            return {
                name: manager.get_state_dict()
                for name, manager in self._managers.items()
            }
    
    def reset_all(self) -> None:
        """重置所有管理器状态"""
        with self._lock:
            for manager in self._managers.values():
                manager.reset()
            logger.info("[CompositeExpiration] 所有管理器状态已重置")


# 默认配置工厂
def create_default_expiration_config() -> ExpirationConfig:
    """创建默认过期配置
    
    Returns:
        默认配置实例
    """
    return ExpirationConfig(
        incremental_refresh_interval=600,  # 10分钟
        full_rebuild_interval=86400,  # 24小时
        incremental_threshold_ratio=0.1,  # 10%
        deletion_check_interval=10,  # 每10次增量
    )


def create_expiration_manager(
    cache_type: str,
    config: Optional[ExpirationConfig] = None,
    name: Optional[str] = None,
) -> ExpirationManager:
    """创建过期管理器工厂函数
    
    Args:
        cache_type: 缓存类型 ("database" 或 "file")
        config: 过期配置，如果为 None 则使用默认配置
        name: 缓存名称
        
    Returns:
        过期管理器实例
    """
    if config is None:
        config = create_default_expiration_config()
    
    if name is None:
        name = cache_type
    
    if cache_type == "database":
        return DatabaseExpirationManager(config, name)
    elif cache_type == "file":
        return FileExpirationManager(config, name)
    else:
        raise ValueError(f"未知的缓存类型: {cache_type}")
