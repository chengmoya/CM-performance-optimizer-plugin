"""
黑话缓存模块 - JargonCacheModule

采用动态导入 core 模块，避免相对导入问题。
支持内容索引加速精确匹配查询。
支持增量更新机制，减少全量刷新开销。
集成 ExpirationManager 统一过期策略管理。
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING, cast

try:
    from src.common.logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger("CM_perf_opt")


# 从公共模块导入动态加载函数
try:
    from core.compat import load_core_module, CoreModuleLoadError
except ImportError:
    # 回退定义
    import importlib.util
    
    def load_core_module(caller_path=None, module_name="CM_perf_opt_core", submodules=None):
        """Fallback load_core_module 实现"""
        module_name = "CM_perf_opt_core"
        if module_name in sys.modules:
            return sys.modules[module_name]
        
        current_dir = Path(__file__).parent
        plugin_dir = current_dir.parent.parent
        core_init = plugin_dir / "core" / "__init__.py"
        
        if not core_init.exists():
            raise ImportError(f"Core module not found at {core_init}")
        
        spec = importlib.util.spec_from_file_location(module_name, core_init)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load core module from {core_init}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    
    class CoreModuleLoadError(ImportError):
        """Core 模块加载失败异常"""
        pass


# 动态导入核心模块
try:
    # 优先尝试相对导入（在正确的包结构中）
    from ..core import ModuleStats, MemoryUtils
    from ..core import (
        ExpirationConfig,
        ExpirationManager,
        DatabaseExpirationManager,
        RefreshDecision,
    )
except ImportError:
    try:
        # 回退到动态加载
        _core = load_core_module(Path(__file__).parent)
        ModuleStats = _core.ModuleStats
        MemoryUtils = _core.MemoryUtils
        ExpirationConfig = _core.ExpirationConfig
        ExpirationManager = _core.ExpirationManager
        DatabaseExpirationManager = _core.DatabaseExpirationManager
        RefreshDecision = _core.RefreshDecision
    except (ImportError, CoreModuleLoadError) as e:
        logger.warning(f"[JargonCache] 无法加载核心模块，使用内置实现: {e}")
        
        # 内置 fallback 实现
        class ModuleStats:
            """Fallback ModuleStats 实现"""
            def __init__(self, name: str):
                self.name = name
                self._lock = threading.Lock()
                self.t_hit = self.t_miss = self.t_filtered = self.t_skipped = 0
                self.i_hit = self.i_miss = self.i_filtered = self.i_skipped = 0
            
            def hit(self):
                with self._lock:
                    self.t_hit += 1
                    self.i_hit += 1
            
            def miss(self, elapsed: float = 0.0):
                with self._lock:
                    self.t_miss += 1
                    self.i_miss += 1
            
            def skipped(self):
                with self._lock:
                    self.t_skipped += 1
                    self.i_skipped += 1
            
            def filtered(self):
                with self._lock:
                    self.t_filtered += 1
                    self.i_filtered += 1
            
            def total(self) -> Dict[str, Any]:
                with self._lock:
                    return {
                        "t_hit": self.t_hit,
                        "t_miss": self.t_miss,
                        "t_filtered": self.t_filtered,
                        "t_skipped": self.t_skipped,
                    }
            
            def reset_interval(self) -> Dict[str, Any]:
                with self._lock:
                    r = {
                        "i_hit": self.i_hit,
                        "i_miss": self.i_miss,
                        "i_filtered": self.i_filtered,
                        "i_skipped": self.i_skipped,
                    }
                    self.i_hit = self.i_miss = self.i_filtered = self.i_skipped = 0
                    return r
        
        class MemoryUtils:
            """Fallback MemoryUtils 实现"""
            @staticmethod
            def get_size(obj, seen=None) -> int:
                if seen is None:
                    seen = set()
                obj_id = id(obj)
                if obj_id in seen:
                    return 0
                seen.add(obj_id)
                size = sys.getsizeof(obj)
                if isinstance(obj, dict):
                    size += sum(MemoryUtils.get_size(k, seen) + MemoryUtils.get_size(v, seen) for k, v in obj.items())
                elif isinstance(obj, (list, tuple, set, frozenset)):
                    size += sum(MemoryUtils.get_size(i, seen) for i in obj)
                return size
        
        # Fallback ExpirationManager 实现
        from dataclasses import dataclass
        from enum import Enum
        
        class RefreshDecision(Enum):
            SKIP = "skip"
            INCREMENTAL = "incremental"
            FULL_REBUILD = "full_rebuild"
            DELETION_CHECK = "deletion_check"
        
        @dataclass
        class ExpirationConfig:
            incremental_refresh_interval: int = 600
            full_rebuild_interval: int = 86400
            incremental_threshold_ratio: float = 0.1
            deletion_check_interval: int = 10
        
        @dataclass
        class ExpirationState:
            last_full_rebuild: float = 0.0
            last_incremental_refresh: float = 0.0
            incremental_refresh_count: int = 0
            total_count: int = 0
            incremental_count: int = 0
            last_max_id: int = 0
        
        class ExpirationManager:
            def __init__(self, config, name="cache"):
                self._config = config
                self._name = name
                self._state = ExpirationState()
            
            @property
            def state(self):
                return self._state
            
            def get_refresh_decision(self, is_first_load=False):
                return RefreshDecision.FULL_REBUILD if is_first_load else RefreshDecision.INCREMENTAL
            
            def record_full_rebuild(self, total_count, max_id=0):
                self._state.last_full_rebuild = time.time()
                self._state.total_count = total_count
                self._state.incremental_count = 0
                self._state.last_max_id = max_id
            
            def record_incremental_refresh(self, new_count, new_max_id=None):
                self._state.incremental_count += new_count
                self._state.total_count += new_count
                self._state.incremental_refresh_count += 1
                if new_max_id is not None:
                    self._state.last_max_id = new_max_id
            
            def get_state_dict(self):
                return {
                    "last_full_rebuild": self._state.last_full_rebuild,
                    "total_count": self._state.total_count,
                    "incremental_count": self._state.incremental_count,
                    "last_max_id": self._state.last_max_id,
                    "incremental_refresh_count": self._state.incremental_refresh_count,
                }
        
        class DatabaseExpirationManager(ExpirationManager):
            def get_last_max_id(self):
                return self._state.last_max_id
            
            def should_skip_incremental(self, current_max_id):
                return current_max_id <= self._state.last_max_id


class JargonCacheModule:
    """黑话全量缓存 - 双缓冲 + 缓慢加载 + 原子切换 + 内容索引 + 增量更新

    功能：
    - 双缓冲设计：buffer_a 为当前使用，buffer_b 为后台加载
    - 缓慢加载：分批从数据库加载，避免 CPU 峰值
    - 原子切换：加载完成后原子交换缓冲区
    - 内容索引：支持按内容精确匹配快速查询
    - 定期刷新：支持自动和手动刷新
    - 增量更新：优先增量刷新，定期全量重建

    注意：数据库中允许同 content 存在多条记录（不同 chat_id / is_global 等）。
    因此 content_index_* 的值为列表而不是单个对象，以避免索引覆盖导致结果不完整。

    Attributes:
        buffer_a: 当前使用的缓存数据
        buffer_b: 后台加载的缓存数据
        content_index_a: 当前使用的内容索引（key=content.lower(), value=Jargon 列表）
        content_index_b: 后台加载的内容索引（key=content.lower(), value=Jargon 列表）
        batch_size: 每批加载的条目数
        batch_delay: 批次间的延迟（秒）
        refresh_interval: 自动刷新间隔（秒）
        enable_content_index: 是否启用内容索引
        stats: 统计信息
        _tracker: 增量更新追踪器
    """
    
    def __init__(
        self,
        batch_size: int = 100,
        batch_delay: float = 0.05,
        refresh_interval: int = 3600,
        enable_content_index: bool = True,
        # 新增增量更新配置参数
        incremental_refresh_interval: int = 600,
        incremental_threshold_ratio: float = 0.1,
        full_rebuild_interval: int = 86400,
        deletion_check_interval: int = 10,
        # 支持传入 ExpirationConfig
        expiration_config: Optional["ExpirationConfig"] = None,
    ):
        """初始化黑话缓存模块

        Args:
            batch_size: 每批加载的条目数，默认 100
            batch_delay: 批次间的延迟秒数，默认 0.05
            refresh_interval: 自动刷新间隔秒数，默认 3600（作为增量刷新间隔的备选）
            enable_content_index: 是否启用内容索引，默认 True
            incremental_refresh_interval: 增量刷新间隔秒数，默认 600（10分钟）
            incremental_threshold_ratio: 触发全量重建的增量比例阈值，默认 0.1（10%）
            full_rebuild_interval: 全量重建间隔秒数，默认 86400（24小时）
            deletion_check_interval: 删除检测间隔（每 N 次增量刷新），默认 10
            expiration_config: 过期配置对象，如果提供则覆盖单独的参数
        """
        # 双缓冲
        self.buffer_a: Optional[List[Any]] = None
        self.buffer_b: Optional[List[Any]] = None
        self.buffer_lock = threading.Lock()

        # 内容索引：content.lower() -> List[Jargon]
        self.content_index_a: Optional[Dict[str, List[Any]]] = None
        self.content_index_b: Optional[Dict[str, List[Any]]] = None
        self.enable_content_index = bool(enable_content_index)

        # 加载配置
        self.batch_size = max(1, int(batch_size))
        self.batch_delay = max(0.001, float(batch_delay))
        self.refresh_interval = max(60, int(refresh_interval))

        # 初始化过期管理器
        if expiration_config is not None:
            self._expiration_config = expiration_config
        else:
            self._expiration_config = ExpirationConfig(
                incremental_refresh_interval=incremental_refresh_interval,
                full_rebuild_interval=full_rebuild_interval,
                incremental_threshold_ratio=incremental_threshold_ratio,
                deletion_check_interval=deletion_check_interval,
            )
        
        # 创建数据库过期管理器
        self._expiration_manager = DatabaseExpirationManager(
            config=self._expiration_config,
            name="jargon_cache",
        )

        # 兼容旧接口：保留属性访问
        self._incremental_refresh_interval = self._expiration_config.incremental_refresh_interval
        self._incremental_threshold_ratio = self._expiration_config.incremental_threshold_ratio
        self._full_rebuild_interval = self._expiration_config.full_rebuild_interval
        self._deletion_check_interval = self._expiration_config.deletion_check_interval

        # 增量更新追踪器（保留用于向后兼容）
        self._tracker: Dict[str, Any] = {
            "last_max_id": 0,
            "total_count": 0,
            "incremental_count": 0,
            "last_full_rebuild": 0.0,
            "incremental_refresh_count": 0,
        }

        # 状态
        self.loading = False
        self.load_lock = asyncio.Lock()
        self.last_refresh: float = 0.0
        self.stats = ModuleStats("jargon_cache")

        # 刷新任务
        self._refresh_task: Optional[asyncio.Task] = None
        self._stopped = False

        # patch 状态（用于修复 from-import 名称绑定问题）
        self._patched = False
        self._patch_lock = threading.RLock()
        self._patched_module_attr = False
        self._original_search_jargon: Optional[Callable[..., Any]] = None
        self._patched_search_jargon: Optional[Callable[..., Any]] = None
        # (owner, globals_dict, old_value)
        self._patched_globals: List[Tuple[str, Dict[str, Any], Any]] = []

        # 启动时立即开始加载
        self._schedule_initial_load()
    
    def _schedule_initial_load(self) -> None:
        """调度初始加载任务"""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._load_to_buffer_b())
            logger.debug("[JargonCache] 已调度初始加载任务")
        except RuntimeError:
            logger.debug("[JargonCache] 没有运行中的事件循环，稍后加载")
    
    def get_all(self) -> List[Any]:
        """获取当前缓存的所有黑话
        
        Returns:
            黑话列表，缓冲区为空时返回空列表
        """
        with self.buffer_lock:
            if self.buffer_a is None:
                return []
            return list(self.buffer_a)  # 返回副本
    
    def get_by_content(self, content: str) -> List[Any]:
        """通过内容精确匹配查询黑话（返回候选列表）

        Args:
            content: 要查询的内容

        Returns:
            候选黑话对象列表；未加载/未命中时返回空列表
        """
        if not self.enable_content_index:
            return []
        key = (content or "").strip().lower()
        if not key:
            return []
        with self.buffer_lock:
            if self.content_index_a is None:
                return []
            return list(self.content_index_a.get(key, []))
    
    def is_loaded(self) -> bool:
        """检查缓存是否已加载完成
        
        Returns:
            True 表示缓存已加载
        """
        with self.buffer_lock:
            return self.buffer_a is not None
    
    def _should_full_rebuild(self) -> bool:
        """判断是否需要全量重建

        使用 ExpirationManager 进行决策。

        Returns:
            True 表示需要全量重建
        """
        # 使用过期管理器获取刷新决策
        decision = self._expiration_manager.get_refresh_decision(
            is_first_load=(self.buffer_a is None)
        )
        
        return decision == RefreshDecision.FULL_REBUILD
    
    def _get_refresh_decision(self) -> "RefreshDecision":
        """获取刷新决策

        使用 ExpirationManager 进行决策。

        Returns:
            刷新决策枚举值
        """
        return self._expiration_manager.get_refresh_decision(
            is_first_load=(self.buffer_a is None)
        )

    async def _load_incremental_to_buffer_b(self) -> None:
        """增量加载数据到缓冲区 B（异步）
        
        使用 ExpirationManager 管理增量刷新逻辑。
        """
        async with self.load_lock:
            if self.loading or self._stopped:
                return
            self.loading = True
        
        try:
            # 获取刷新决策
            decision = self._get_refresh_decision()
            
            # 根据决策执行不同操作
            if decision == RefreshDecision.FULL_REBUILD:
                logger.info("[JargonCache] 过期管理器决策：执行全量重建...")
                async with self.load_lock:
                    self.loading = False
                await self._load_to_buffer_b()
                return
            
            if decision == RefreshDecision.SKIP:
                logger.debug("[JargonCache] 过期管理器决策：跳过刷新")
                return
            
            # 处理删除检测
            if decision == RefreshDecision.DELETION_CHECK:
                logger.info("[JargonCache] 过期管理器决策：执行删除检测...")
                await self._check_deleted_records()
                return
            
            # 增量刷新
            logger.info("[JargonCache] 过期管理器决策：执行增量刷新...")
            t0 = time.time()

            incremental_data: List[Any] = []
            current_max_id = self._expiration_manager.get_last_max_id()

            # 尝试从数据库加载增量数据
            try:
                from src.common.database.database_model import Jargon
                from peewee import fn

                # 1. 查询当前最大 ID
                max_id_result = await asyncio.to_thread(
                    lambda: Jargon.select(fn.MAX(Jargon.id)).scalar()
                )
                current_max_id = max_id_result or 0

                # 2. 查询增量数据
                last_max_id = self._expiration_manager.get_last_max_id()

                # 检查是否可以跳过增量
                if self._expiration_manager.should_skip_incremental(current_max_id):
                    logger.debug("[JargonCache] 数据库无新数据，跳过增量刷新")
                    return

                if current_max_id > last_max_id:
                    offset = 0
                    while not self._stopped:
                        batch = await asyncio.to_thread(
                            lambda o=offset, l=last_max_id: list(
                                Jargon.select()
                                .where(Jargon.id > l)
                                .order_by(Jargon.id.asc())
                                .limit(self.batch_size)
                                .offset(o)
                            )
                        )
                        if not batch:
                            break

                        incremental_data.extend(batch)

                        # 记录进度
                        if len(incremental_data) % 500 == 0:
                            logger.debug(f"[JargonCache] 增量加载进度: {len(incremental_data)} 条")

                        # 休眠，避免 CPU 峰值
                        await asyncio.sleep(self.batch_delay)
                        offset += self.batch_size

            except ImportError as e:
                logger.warning(f"[JargonCache] 无法导入数据库模型: {e}")
                return
            except Exception as e:
                logger.error(f"[JargonCache] 增量加载数据失败: {e}")
                return

            if self._stopped:
                logger.info("[JargonCache] 增量加载被中止")
                return

            # 3. 合并增量数据
            if incremental_data:
                await self._merge_incremental_data(incremental_data)
                # 使用过期管理器记录增量刷新
                self._expiration_manager.record_incremental_refresh(
                    new_count=len(incremental_data),
                    new_max_id=current_max_id,
                )
                # 同步更新兼容追踪器
                self._sync_tracker_from_manager()

            self.last_refresh = time.time()

            load_time = time.time() - t0
            state = self._expiration_manager.state
            logger.info(
                f"[JargonCache] 增量加载完成: {len(incremental_data)} 条新数据, "
                f"累计增量: {state.incremental_count}, 耗时 {load_time:.2f}s"
            )

        except Exception as e:
            logger.error(f"[JargonCache] 增量加载失败: {e}")
        finally:
            async with self.load_lock:
                self.loading = False
    
    def _sync_tracker_from_manager(self) -> None:
        """从 ExpirationManager 同步状态到兼容追踪器"""
        state = self._expiration_manager.state
        self._tracker["last_max_id"] = state.last_max_id
        self._tracker["total_count"] = state.total_count
        self._tracker["incremental_count"] = state.incremental_count
        self._tracker["last_full_rebuild"] = state.last_full_rebuild
        self._tracker["incremental_refresh_count"] = state.incremental_refresh_count

    async def _merge_incremental_data(self, incremental_data: List[Any]) -> None:
        """合并增量数据到缓存

        Args:
            incremental_data: 增量数据列表
        """
        with self.buffer_lock:
            # 复制现有数据
            if self.buffer_a is not None:
                buffer_b_data = list(self.buffer_a)
                content_index_b = {}
                if self.content_index_a:
                    # 深拷贝索引
                    for k, v in self.content_index_a.items():
                        content_index_b[k] = list(v)
            else:
                buffer_b_data = []
                content_index_b = {}

            # 构建 ID 集合用于去重
            existing_ids = {getattr(item, 'id', None) for item in buffer_b_data}

            # 合并新数据
            for item in incremental_data:
                item_id = getattr(item, 'id', None)

                # 更新或追加
                if item_id in existing_ids:
                    # 替换已存在的记录
                    for i, existing in enumerate(buffer_b_data):
                        if getattr(existing, 'id', None) == item_id:
                            buffer_b_data[i] = item
                            break
                else:
                    # 追加新记录
                    buffer_b_data.append(item)
                    existing_ids.add(item_id)

                # 更新内容索引
                if self.enable_content_index:
                    content = getattr(item, 'content', None)
                    if content:
                        key = str(content).strip().lower()
                        if key:
                            # 移除旧记录的同 ID 项
                            if key in content_index_b:
                                content_index_b[key] = [
                                    x for x in content_index_b[key]
                                    if getattr(x, 'id', None) != item_id
                                ]
                            content_index_b.setdefault(key, []).append(item)

            # 更新总数
            self._tracker["total_count"] = len(buffer_b_data)

            # 原子切换
            self.buffer_b = buffer_b_data
            self.content_index_b = content_index_b
            self.buffer_a, self.buffer_b = self.buffer_b, None
            self.content_index_a, self.content_index_b = self.content_index_b, None

    async def _check_deleted_records(self) -> None:
        """检测并清理已删除的记录"""
        try:
            from src.common.database.database_model import Jargon
            from peewee import fn

            # 1. 快速检测：比较数据库记录数与缓存记录数
            db_count = await asyncio.to_thread(
                lambda: Jargon.select(fn.COUNT(Jargon.id)).scalar()
            )

            with self.buffer_lock:
                if self.buffer_a is None:
                    return
                cache_count = len(self.buffer_a)

            # 如果数量一致，跳过详细检测
            if db_count == cache_count:
                logger.debug(f"[JargonCache] 记录数一致 ({cache_count})，无需清理")
                return

            # 2. 数量不一致，执行详细检测
            logger.info(f"[JargonCache] 检测到记录数差异: 数据库 {db_count}, 缓存 {cache_count}，执行清理...")

            # 获取数据库中当前所有 ID
            db_ids = set(await asyncio.to_thread(
                lambda: [r.id for r in Jargon.select(Jargon.id)]
            ))

            # 获取缓存中的 ID
            with self.buffer_lock:
                if self.buffer_a is None:
                    return
                cache_ids = {getattr(item, 'id', None) for item in self.buffer_a}

            # 找出已删除的 ID
            deleted_ids = cache_ids - db_ids

            if not deleted_ids:
                logger.debug("[JargonCache] 未检测到已删除记录")
                return

            logger.info(f"[JargonCache] 检测到 {len(deleted_ids)} 条已删除记录，执行清理...")

            # 3. 清理已删除的记录
            with self.buffer_lock:
                if self.buffer_a is None:
                    return

                # 过滤已删除的记录
                buffer_b_data = [
                    item for item in self.buffer_a
                    if getattr(item, 'id', None) not in deleted_ids
                ]

                # 更新内容索引
                content_index_b = {}
                if self.enable_content_index and self.content_index_a:
                    for k, v in self.content_index_a.items():
                        filtered = [x for x in v if getattr(x, 'id', None) not in deleted_ids]
                        if filtered:
                            content_index_b[k] = filtered

                # 更新总数
                self._tracker["total_count"] = len(buffer_b_data)

                # 原子切换
                self.buffer_b = buffer_b_data
                self.content_index_b = content_index_b
                self.buffer_a, self.buffer_b = self.buffer_b, None
                self.content_index_a, self.content_index_b = self.content_index_b, None

            logger.info(f"[JargonCache] 已清理 {len(deleted_ids)} 条已删除记录")

        except ImportError as e:
            logger.warning(f"[JargonCache] 无法导入数据库模型进行删除检测: {e}")
        except Exception as e:
            logger.error(f"[JargonCache] 删除检测失败: {e}")

    async def _load_to_buffer_b(self) -> None:
        """缓慢加载数据到缓冲区 B（异步）- 全量加载"""
        async with self.load_lock:
            if self.loading or self._stopped:
                return
            self.loading = True
        
        try:
            logger.info("[JargonCache] 开始全量加载黑话缓存到缓冲区B...")
            t0 = time.time()
            
            # 清空缓冲区 B
            buffer_b_data: List[Any] = []
            content_index_b: Optional[Dict[str, List[Any]]] = {} if self.enable_content_index else None
            max_id = 0
            
            # 尝试从数据库加载
            try:
                from src.common.database.database_model import Jargon
                from peewee import fn
                
                # 分批加载
                offset = 0
                while not self._stopped:
                    # 查询一批数据
                    batch = await asyncio.to_thread(
                        lambda: list(Jargon.select().limit(self.batch_size).offset(offset))
                    )
                    if not batch:
                        break
                    
                    # 添加到缓冲区 B
                    buffer_b_data.extend(batch)

                    # 更新最大 ID
                    for item in batch:
                        item_id = getattr(item, 'id', 0)
                        if item_id > max_id:
                            max_id = item_id
                    
                    # 构建内容索引（同 content 可能多条记录，必须用列表避免覆盖）
                    if self.enable_content_index and content_index_b is not None:
                        for jargon in batch:
                            content = getattr(jargon, "content", None)
                            if not content:
                                continue
                            key = str(content).strip().lower()
                            if not key:
                                continue
                            content_index_b.setdefault(key, []).append(jargon)
                    
                    # 记录进度
                    if len(buffer_b_data) % 500 == 0:
                        logger.debug(f"[JargonCache] 加载进度: {len(buffer_b_data)} 条")
                    
                    # 休眠，避免 CPU 峰值
                    await asyncio.sleep(self.batch_delay)
                    offset += self.batch_size
                    
            except ImportError as e:
                logger.warning(f"[JargonCache] 无法导入数据库模型: {e}")
                buffer_b_data = []
                content_index_b = {} if self.enable_content_index else None
            except Exception as e:
                logger.error(f"[JargonCache] 加载数据失败: {e}")
                buffer_b_data = []
                content_index_b = {} if self.enable_content_index else None
            
            if self._stopped:
                logger.info("[JargonCache] 加载被中止")
                return
            
            # 加载完成，原子切换
            with self.buffer_lock:
                self.buffer_b = buffer_b_data
                self.content_index_b = content_index_b
                # 原子切换
                self.buffer_a, self.buffer_b = self.buffer_b, None
                self.content_index_a, self.content_index_b = self.content_index_b, None

            # 使用过期管理器记录全量重建
            self._expiration_manager.record_full_rebuild(
                total_count=len(buffer_b_data),
                max_id=max_id,
            )
            # 同步更新兼容追踪器
            self._sync_tracker_from_manager()
            
            self.last_refresh = time.time()
            load_time = time.time() - t0
            logger.info(f"[JargonCache] 全量缓存加载完成并切换: {len(buffer_b_data)} 条, max_id={max_id}, 耗时 {load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"[JargonCache] 缓存加载失败: {e}")
        finally:
            async with self.load_lock:
                self.loading = False
    
    async def _refresh_loop(self) -> None:
        """定期刷新循环（智能决策：增量优先 + 定期全量重建）"""
        while not self._stopped:
            # 使用增量刷新间隔
            await asyncio.sleep(self._incremental_refresh_interval)
            
            if self._stopped:
                break

            # 判断刷新策略
            if self._should_full_rebuild():
                logger.info("[JargonCache] 触发全量重建...")
                await self._load_to_buffer_b()
            else:
                logger.info("[JargonCache] 触发增量刷新...")
                await self._load_incremental_to_buffer_b()
    
    def start(self) -> None:
        """启动缓存模块"""
        self._stopped = False
        try:
            loop = asyncio.get_running_loop()
            if self._refresh_task is None or self._refresh_task.done():
                self._refresh_task = loop.create_task(self._refresh_loop())
                logger.info("[JargonCache] 定期刷新任务已启动（增量模式）")
            
            # 如果缓存未加载，立即触发加载
            if not self.is_loaded():
                loop.create_task(self._load_to_buffer_b())
        except RuntimeError:
            logger.warning("[JargonCache] 无法启动定期刷新：没有运行中的事件循环")
    
    def stop(self) -> None:
        """停止缓存模块。

        说明：本插件会在优化器停止时调用所有缓存模块的 `stop()`。
        为确保 monkey-patch 不残留，这里同步触发 `revert_patch()`。
        """
        self._stopped = True
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            self._refresh_task = None

        try:
            self.revert_patch()
        except Exception as e:
            logger.warning(f"[JargonCache] stop() 中恢复 patch 失败: {e}")

        logger.info("[JargonCache] 缓存模块已停止")
    
    def refresh(self) -> None:
        """手动刷新缓存（非阻塞，触发全量刷新）"""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._load_to_buffer_b())
            logger.info("[JargonCache] 已触发手动全量刷新")
        except RuntimeError:
            logger.warning("[JargonCache] 无法触发刷新：没有运行中的事件循环")
    
    def clear(self) -> None:
        """清空缓存"""
        with self.buffer_lock:
            self.buffer_a = None
            self.buffer_b = None
            self.content_index_a = None
            self.content_index_b = None
            # 重置追踪器
            self._tracker = {
                "last_max_id": 0,
                "total_count": 0,
                "incremental_count": 0,
                "last_full_rebuild": 0.0,
                "incremental_refresh_count": 0,
            }
        logger.info("[JargonCache] 缓存已清空")
    
    def size(self) -> int:
        """获取缓存大小
        
        Returns:
            缓存条目数
        """
        with self.buffer_lock:
            return len(self.buffer_a) if self.buffer_a else 0
    
    def get_memory_usage(self) -> int:
        """获取缓存内存使用量
        
        Returns:
            内存使用量（字节）
        """
        with self.buffer_lock:
            if self.buffer_a is None:
                return 0
            size = MemoryUtils.get_size(self.buffer_a)
            if self.content_index_a is not None:
                size += MemoryUtils.get_size(self.content_index_a)
            return size
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息

        Returns:
            统计信息字典
        """
        now = time.time()
        return {
            "size": self.size(),
            "memory_bytes": self.get_memory_usage(),
            "last_refresh": self.last_refresh,
            "loading": self.loading,
            "content_index_enabled": self.enable_content_index,
            "content_index_size": (
                sum(len(v) for v in self.content_index_a.values())
                if self.content_index_a
                else 0
            ),
            # 增量更新统计
            "last_max_id": self._tracker["last_max_id"],
            "total_count": self._tracker["total_count"],
            "incremental_count": self._tracker["incremental_count"],
            "last_full_rebuild": self._tracker["last_full_rebuild"],
            "time_since_last_rebuild": now - self._tracker["last_full_rebuild"],
            "incremental_refresh_count": self._tracker["incremental_refresh_count"],
            # 配置信息
            "incremental_refresh_interval": self._incremental_refresh_interval,
            "incremental_threshold_ratio": self._incremental_threshold_ratio,
            "full_rebuild_interval": self._full_rebuild_interval,
            **self.stats.total(),
        }

    def _patch_from_import_bindings(
        self,
        original_search_jargon: Callable[..., Any],
        patched_search_jargon: Callable[..., Any],
    ) -> None:
        """修复 `from module import search_jargon` 的名称绑定陷阱。

        说明：
        - 调用方使用 `from ... import search_jargon` 时，会在其模块 globals 中创建本地绑定。
        - 后续仅修改 `jargon_miner.search_jargon` 模块属性无法影响这些已绑定引用。
        - 因此扫描已加载模块，定位所有函数对象的 `__globals__['search_jargon']`，并替换。

        Args:
            original_search_jargon: 原始函数引用
            patched_search_jargon: patch 后函数引用
        """
        patched_owners: List[str] = []
        seen_globals: Set[int] = set()

        # 清理旧记录（避免重复 patch 时堆积）
        self._patched_globals.clear()

        for mod_name, mod in list(sys.modules.items()):
            if mod is None:
                continue
            # 只扫描 src. 开头的模块，控制范围
            if not str(mod_name).startswith("src."):
                continue

            try:
                for attr_name in dir(mod):
                    try:
                        attr = getattr(mod, attr_name, None)
                    except Exception:
                        continue

                    if not callable(attr) or not hasattr(attr, "__globals__"):
                        continue

                    g = getattr(attr, "__globals__", None)
                    if not isinstance(g, dict):
                        continue

                    if g.get("search_jargon") is not original_search_jargon:
                        continue

                    gid = id(g)
                    if gid in seen_globals:
                        continue
                    seen_globals.add(gid)

                    # 记录并替换
                    owner = f"{mod_name}.{attr_name}"
                    old_value = g.get("search_jargon")

                    try:
                        g["search_jargon"] = patched_search_jargon
                        self._patched_globals.append((owner, g, old_value))
                        patched_owners.append(owner)
                    except Exception as e:
                        logger.warning(
                            f"[JargonCache] __globals__ patch 失败: {owner}: {e}"
                        )
                        continue

            except Exception as e:
                # 单个模块失败不影响整体 patch
                logger.warning(f"[JargonCache] 扫描模块失败: {mod_name}: {e}")
                continue

        if patched_owners:
            logger.info(
                f"[JargonCache] 额外修复了 {len(patched_owners)} 个 from-import 绑定"
            )
            logger.debug(f"[JargonCache] 修复的绑定: {patched_owners}")

    def apply_patch(self) -> None:
        """应用 patch：

        1) patch `src.bw_learner.jargon_miner.search_jargon`
        2) 扫描并修复所有调用方 `__globals__['search_jargon']` 的 from-import 绑定

        注意：__globals__ 修复必须在模块属性 patch 之后执行。
        """
        with self._patch_lock:
            if self._patched:
                return

            try:
                from src.bw_learner import jargon_miner as jm
            except ImportError as e:
                logger.debug(f"[JargonCache] jargon_miner 不可用，跳过 patch: {e}")
                return

            current = getattr(jm, "search_jargon", None)
            if not callable(current):
                logger.warning("[JargonCache] jargon_miner.search_jargon 不存在或不可调用")
                return

            # 情况 A：已经被 patch（可能来自此前实例/重载），仍需要做 __globals__ 修复
            if getattr(current, "__cm_perf_opt_patched__", False):
                wrapped = getattr(current, "__wrapped__", None)
                if callable(wrapped):
                    self._original_search_jargon = wrapped
                    self._patched_search_jargon = current
                    self._patched_module_attr = False
                    self._patch_from_import_bindings(wrapped, current)
                    self._patched = True
                else:
                    logger.info("[JargonCache] search_jargon 已 patch（但缺少 __wrapped__），跳过")
                return

            original_search_jargon = cast(Callable[..., List[Dict[str, str]]], current)

            cache = self
            original = original_search_jargon

            def patched_search_jargon(
                keyword: str,
                chat_id: Optional[str] = None,
                limit: int = 10,
                case_sensitive: bool = False,
                fuzzy: bool = True,
            ) -> List[Dict[str, str]]:
                """补丁后的黑话搜索（仅加速精确匹配）。"""
                if not keyword or not str(keyword).strip():
                    return []

                # 仅对精确匹配启用缓存（fuzzy=True 的逻辑较复杂，保持与核心一致）
                if fuzzy or not cache.is_loaded():
                    return original(
                        keyword=keyword,
                        chat_id=chat_id,
                        limit=limit,
                        case_sensitive=case_sensitive,
                        fuzzy=fuzzy,
                    )

                t0 = time.time()
                kw = str(keyword).strip()

                # 通过内容索引获取同 content 的所有候选记录
                candidates = cache.get_by_content(kw)
                if not candidates:
                    cache.stats.hit()
                    return []

                # 复用核心模块中的 chat_id 解析工具（兼容新旧格式）
                parse_chat_id_list = getattr(jm, "parse_chat_id_list", None)
                chat_id_list_contains = getattr(jm, "chat_id_list_contains", None)

                results: List[Dict[str, str]] = []
                filtered = []

                for jargon in candidates:
                    content = getattr(jargon, "content", "") or ""
                    meaning = getattr(jargon, "meaning", "") or ""

                    if not meaning.strip():
                        continue

                    # 精确匹配：大小写敏感与否
                    if case_sensitive:
                        if content != kw:
                            continue
                    else:
                        if content.strip().lower() != kw.lower():
                            continue

                    # all_global 逻辑：开启时只返回 is_global=True
                    if getattr(jm.global_config.expression, "all_global_jargon", False):
                        if not getattr(jargon, "is_global", False):
                            continue
                    else:
                        # 关闭 all_global：如果提供 chat_id，需要允许 is_global 或 chat_id 命中
                        if chat_id and not getattr(jargon, "is_global", False):
                            if parse_chat_id_list is None or chat_id_list_contains is None:
                                return original(
                                    keyword=keyword,
                                    chat_id=chat_id,
                                    limit=limit,
                                    case_sensitive=case_sensitive,
                                    fuzzy=fuzzy,
                                )
                            chat_id_list = parse_chat_id_list(
                                getattr(jargon, "chat_id", None)
                            )
                            if not chat_id_list_contains(chat_id_list, chat_id):
                                continue

                    filtered.append(jargon)

                # 按 count 降序，匹配核心行为
                filtered.sort(
                    key=lambda j: int(getattr(j, "count", 0) or 0), reverse=True
                )

                for jargon in filtered:
                    results.append(
                        {
                            "content": getattr(jargon, "content", "") or "",
                            "meaning": getattr(jargon, "meaning", "") or "",
                        }
                    )
                    if len(results) >= max(0, int(limit)):
                        break

                cache.stats.hit()
                return results

            patched_search_jargon.__cm_perf_opt_patched__ = True  # type: ignore[attr-defined]
            patched_search_jargon.__wrapped__ = original_search_jargon  # type: ignore[attr-defined]

            # 1) patch 模块属性
            jm.search_jargon = patched_search_jargon

            self._original_search_jargon = original_search_jargon
            self._patched_search_jargon = patched_search_jargon
            self._patched_module_attr = True

            logger.info("[JargonCache] 已 patch src.bw_learner.jargon_miner.search_jargon")

            # 2) 修复 from-import 绑定（必须在模块属性 patch 之后）
            self._patch_from_import_bindings(original_search_jargon, patched_search_jargon)

            self._patched = True

    def revert_patch(self) -> None:
        """恢复 patch：

        - 恢复 `jargon_miner.search_jargon`（仅当本实例曾修改过模块属性）
        - 精确恢复所有被修改的 `__globals__['search_jargon']`

        若恢复失败，记录 warning 但不中断。
        """
        with self._patch_lock:
            if not self._patched:
                return

            original = self._original_search_jargon
            patched = self._patched_search_jargon

            # 1) 恢复模块属性（仅当本实例修改过）
            if self._patched_module_attr and callable(original) and callable(patched):
                try:
                    from src.bw_learner import jargon_miner as jm

                    if getattr(jm, "search_jargon", None) is patched:
                        jm.search_jargon = original
                        logger.info(
                            "[JargonCache] 已恢复 src.bw_learner.jargon_miner.search_jargon"
                        )
                    else:
                        logger.warning(
                            "[JargonCache] revert_patch：jargon_miner.search_jargon 已被其他逻辑修改，跳过恢复"
                        )
                except Exception as e:
                    logger.warning(f"[JargonCache] 恢复模块属性失败: {e}")

            # 2) 恢复所有 __globals__ 绑定
            if callable(original) and callable(patched):
                reverted: List[str] = []
                for owner, g, old_value in list(self._patched_globals):
                    try:
                        if not isinstance(g, dict):
                            continue
                        if g.get("search_jargon") is patched:
                            g["search_jargon"] = old_value
                            reverted.append(owner)
                    except Exception as e:
                        logger.warning(
                            f"[JargonCache] 恢复 __globals__ 失败: {owner}: {e}"
                        )
                        continue

                if reverted:
                    logger.info(
                        f"[JargonCache] 已恢复 {len(reverted)} 个 from-import 绑定"
                    )
                    logger.debug(f"[JargonCache] 恢复的绑定: {reverted}")

            self._patched_globals.clear()
            self._patched = False
            self._patched_module_attr = False
            self._original_search_jargon = None
            self._patched_search_jargon = None


def apply_jargon_cache(cache_manager) -> Optional[JargonCacheModule]:
    """应用黑话缓存补丁

    适配新版 MaiBot：黑话查询入口位于 [`src.bw_learner.jargon_miner.search_jargon()`](../src/bw_learner/jargon_miner.py:617)。
    本模块通过缓存加速 **精确匹配 (fuzzy=False)** 场景；模糊搜索仍回退核心实现。

    Args:
        cache_manager: 缓存管理器实例

    Returns:
        JargonCacheModule 实例，失败时返回 None
    """
    try:
        cache = JargonCacheModule()
        cache_manager.register_cache("jargon_cache", cache)

        try:
            cache.apply_patch()
        except Exception as e:
            logger.warning(f"[JargonCache] patch 失败: {e}")

        return cache

    except Exception as e:
        logger.error(f"[JargonCache] 初始化失败: {e}")
        return None
