"""
表达式缓存模块 - ExpressionCacheModule

采用动态导入 core 模块，避免相对导入问题。
支持增量更新机制，减少全量刷新开销。
集成 ExpirationManager 统一过期策略管理。
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

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
    def load_core_module(caller_path=None, module_name="CM_perf_opt_core", submodules=None):
        """Fallback load_core_module 实现"""
        import importlib.util
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
        logger.warning(f"[ExprCache] 无法加载核心模块，使用内置实现: {e}")
        
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


class ExpressionCacheModule:
    """表达式全量缓存 - 双缓冲 + 缓慢加载 + 原子切换 + 增量更新

    功能：
    - 双缓冲设计：buffer_a 为当前使用，buffer_b 为后台加载
    - 缓慢加载：分批从数据库加载，避免 CPU 峰值
    - 原子切换：加载完成后原子交换缓冲区
    - chat_id 索引：为高频按 chat_id 过滤场景准备的索引
    - 定期刷新：支持自动和手动刷新
    - 增量更新：优先增量刷新，定期全量重建

    Attributes:
        buffer_a: 当前使用的缓存数据
        buffer_b: 后台加载的缓存数据
        index_a: 当前使用的 chat_id 索引
        index_b: 后台加载的 chat_id 索引
        batch_size: 每批加载的条目数
        batch_delay: 批次间的延迟（秒）
        refresh_interval: 自动刷新间隔（秒）
        stats: 统计信息
        _tracker: 增量更新追踪器
    """
    
    def __init__(
        self,
        batch_size: int = 100,
        batch_delay: float = 0.05,
        refresh_interval: int = 3600,
        # 新增增量更新配置参数
        incremental_refresh_interval: int = 600,
        incremental_threshold_ratio: float = 0.1,
        full_rebuild_interval: int = 86400,
        deletion_check_interval: int = 10,
        # 支持传入 ExpirationConfig
        expiration_config: Optional["ExpirationConfig"] = None,
    ):
        """初始化表达式缓存模块

        Args:
            batch_size: 每批加载的条目数，默认 100
            batch_delay: 批次间的延迟秒数，默认 0.05
            refresh_interval: 自动刷新间隔秒数，默认 3600（作为增量刷新间隔的备选）
            incremental_refresh_interval: 增量刷新间隔秒数，默认 600（10分钟）
            incremental_threshold_ratio: 触发全量重建的增量比例阈值，默认 0.1（10%）
            full_rebuild_interval: 全量重建间隔秒数，默认 86400（24小时）
            deletion_check_interval: 删除检测间隔（每 N 次增量刷新），默认 10
            expiration_config: 过期配置对象，如果提供则覆盖单独的参数
        """
        # 双缓冲
        self.buffer_a: Optional[List[Any]] = None  # 当前使用的缓存（Expression 模型列表）
        self.buffer_b: Optional[List[Any]] = None  # 后台加载的缓存（Expression 模型列表）
        # 为高频按 chat_id 过滤场景准备的索引（与 buffer_a 同步切换）
        self.index_a: Optional[Dict[str, List[Any]]] = None
        self.index_b: Optional[Dict[str, List[Any]]] = None
        self.buffer_lock = threading.Lock()
        
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
            name="expression_cache",
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
        self.stats = ModuleStats("expression_cache")
        
        # 刷新任务
        self._refresh_task: Optional[asyncio.Task] = None
        self._stopped = False
        
        # 补丁状态：保存原始函数引用以便 remove_patch 恢复
        self._patched: bool = False
        self._orig_random_expressions: Optional[Callable] = None
        self._orig_select_simple: Optional[Callable] = None
        self._orig_select_classic: Optional[Callable] = None
        self._orig_get_all: Optional[Callable] = None
        self._expression_manager_ref: Any = None  # 旧核心 expression_manager 引用
        
        # 启动时立即开始加载
        self._schedule_initial_load()
    
    def _schedule_initial_load(self) -> None:
        """调度初始加载任务"""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._load_to_buffer_b())
            logger.debug("[ExprCache] 已调度初始加载任务")
        except RuntimeError:
            logger.debug("[ExprCache] 没有运行中的事件循环，稍后加载")
    
    def get_all(self) -> List[Any]:
        """获取当前缓存的所有表达式。

        Returns:
            表达式列表，缓冲区为空时返回空列表。
        """
        with self.buffer_lock:
            if self.buffer_a is None:
                return []
            return list(self.buffer_a)  # 返回副本，避免外部修改

    def get_by_chat_ids(self, chat_ids: List[str]) -> List[Any]:
        """按 chat_id 批量获取表达式（优先走索引）。

        该函数用于适配核心中高频读取点（如 ExpressionSelector）。

        Args:
            chat_ids: chat_id 列表

        Returns:
            匹配的 Expression 模型列表（副本）
        """
        if not chat_ids:
            return []

        with self.buffer_lock:
            if self.buffer_a is None:
                return []

            if self.index_a is not None:
                result: List[Any] = []
                for cid in chat_ids:
                    items = self.index_a.get(cid)
                    if items:
                        result.extend(items)
                return list(result)

            # fallback：没有索引时全表扫描
            cid_set = set(chat_ids)
            return [expr for expr in self.buffer_a if getattr(expr, "chat_id", None) in cid_set]
    
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
    
    def _sync_tracker_from_manager(self) -> None:
        """从 ExpirationManager 同步状态到兼容追踪器"""
        state = self._expiration_manager.state
        self._tracker["last_max_id"] = state.last_max_id
        self._tracker["total_count"] = state.total_count
        self._tracker["incremental_count"] = state.incremental_count
        self._tracker["last_full_rebuild"] = state.last_full_rebuild
        self._tracker["incremental_refresh_count"] = state.incremental_refresh_count

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
                logger.info("[ExprCache] 过期管理器决策：执行全量重建...")
                async with self.load_lock:
                    self.loading = False
                await self._load_to_buffer_b()
                return
            
            if decision == RefreshDecision.SKIP:
                logger.debug("[ExprCache] 过期管理器决策：跳过刷新")
                return
            
            # 处理删除检测
            if decision == RefreshDecision.DELETION_CHECK:
                logger.info("[ExprCache] 过期管理器决策：执行删除检测...")
                await self._check_deleted_records()
                return
            
            # 增量刷新
            logger.info("[ExprCache] 过期管理器决策：执行增量刷新...")
            t0 = time.time()

            incremental_data: List[Any] = []
            current_max_id = self._expiration_manager.get_last_max_id()

            # 尝试从数据库加载增量数据
            try:
                from src.common.database.database_model import Expression
                from peewee import fn

                # 1. 查询当前最大 ID
                max_id_result = await asyncio.to_thread(
                    lambda: Expression.select(fn.MAX(Expression.id)).scalar()
                )
                current_max_id = max_id_result or 0

                # 2. 查询增量数据
                last_max_id = self._expiration_manager.get_last_max_id()

                # 检查是否可以跳过增量
                if self._expiration_manager.should_skip_incremental(current_max_id):
                    logger.debug("[ExprCache] 数据库无新数据，跳过增量刷新")
                    return

                if current_max_id > last_max_id:
                    offset = 0
                    while not self._stopped:
                        batch = await asyncio.to_thread(
                            lambda o=offset, l=last_max_id: list(
                                Expression.select()
                                .where(Expression.id > l)
                                .order_by(Expression.id.asc())
                                .limit(self.batch_size)
                                .offset(o)
                            )
                        )
                        if not batch:
                            break

                        incremental_data.extend(batch)

                        # 记录进度
                        if len(incremental_data) % 500 == 0:
                            logger.debug(f"[ExprCache] 增量加载进度: {len(incremental_data)} 条")

                        # 休眠，避免 CPU 峰值
                        await asyncio.sleep(self.batch_delay)
                        offset += self.batch_size

            except ImportError as e:
                logger.warning(f"[ExprCache] 无法导入数据库模型: {e}")
                return
            except Exception as e:
                logger.error(f"[ExprCache] 增量加载数据失败: {e}")
                return

            if self._stopped:
                logger.info("[ExprCache] 增量加载被中止")
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
                f"[ExprCache] 增量加载完成: {len(incremental_data)} 条新数据, "
                f"累计增量: {state.incremental_count}, 耗时 {load_time:.2f}s"
            )

        except Exception as e:
            logger.error(f"[ExprCache] 增量加载失败: {e}")
        finally:
            async with self.load_lock:
                self.loading = False

    async def _merge_incremental_data(self, incremental_data: List[Any]) -> None:
        """合并增量数据到缓存（含 chat_id 索引）

        Args:
            incremental_data: 增量数据列表
        """
        with self.buffer_lock:
            # 复制现有数据和索引
            if self.buffer_a is not None:
                buffer_b_data = list(self.buffer_a)
                # 深拷贝 chat_id 索引
                index_b = {}
                if self.index_a:
                    for k, v in self.index_a.items():
                        index_b[k] = list(v)
            else:
                buffer_b_data = []
                index_b = {}

            # 构建 ID 集合用于去重
            existing_ids = {getattr(item, 'id', None) for item in buffer_b_data}

            # 合并新数据
            for item in incremental_data:
                item_id = getattr(item, 'id', None)
                old_chat_id = None

                # 更新或追加
                if item_id in existing_ids:
                    # 替换已存在的记录，并记录旧 chat_id
                    for i, existing in enumerate(buffer_b_data):
                        if getattr(existing, 'id', None) == item_id:
                            old_chat_id = getattr(existing, 'chat_id', None)
                            buffer_b_data[i] = item
                            break
                else:
                    # 追加新记录
                    buffer_b_data.append(item)
                    existing_ids.add(item_id)

                # 更新 chat_id 索引
                new_chat_id = getattr(item, 'chat_id', None)
                if new_chat_id:
                    chat_key = str(new_chat_id)

                    # 从旧 chat_id 索引中移除（如果 chat_id 发生变化）
                    if old_chat_id and str(old_chat_id) != chat_key:
                        old_key = str(old_chat_id)
                        if old_key in index_b:
                            index_b[old_key] = [
                                x for x in index_b[old_key]
                                if getattr(x, 'id', None) != item_id
                            ]
                            # 如果索引为空，移除该键
                            if not index_b[old_key]:
                                del index_b[old_key]

                    # 添加到新 chat_id 索引
                    if chat_key not in index_b:
                        index_b[chat_key] = []

                    # 移除同 ID 旧项（避免重复）
                    index_b[chat_key] = [
                        x for x in index_b[chat_key]
                        if getattr(x, 'id', None) != item_id
                    ]
                    index_b[chat_key].append(item)

            # 更新总数
            self._tracker["total_count"] = len(buffer_b_data)

            # 原子切换
            self.buffer_b = buffer_b_data
            self.index_b = index_b
            self.buffer_a, self.buffer_b = self.buffer_b, None
            self.index_a, self.index_b = self.index_b, None

    async def _check_deleted_records(self) -> None:
        """检测并清理已删除的记录"""
        try:
            from src.common.database.database_model import Expression
            from peewee import fn

            # 1. 快速检测：比较数据库记录数与缓存记录数
            db_count = await asyncio.to_thread(
                lambda: Expression.select(fn.COUNT(Expression.id)).scalar()
            )

            with self.buffer_lock:
                if self.buffer_a is None:
                    return
                cache_count = len(self.buffer_a)

            # 如果数量一致，跳过详细检测
            if db_count == cache_count:
                logger.debug(f"[ExprCache] 记录数一致 ({cache_count})，无需清理")
                return

            # 2. 数量不一致，执行详细检测
            logger.info(f"[ExprCache] 检测到记录数差异: 数据库 {db_count}, 缓存 {cache_count}，执行清理...")

            # 获取数据库中当前所有 ID
            db_ids = set(await asyncio.to_thread(
                lambda: [r.id for r in Expression.select(Expression.id)]
            ))

            # 获取缓存中的 ID
            with self.buffer_lock:
                if self.buffer_a is None:
                    return
                cache_ids = {getattr(item, 'id', None) for item in self.buffer_a}

            # 找出已删除的 ID
            deleted_ids = cache_ids - db_ids

            if not deleted_ids:
                logger.debug("[ExprCache] 未检测到已删除记录")
                return

            logger.info(f"[ExprCache] 检测到 {len(deleted_ids)} 条已删除记录，执行清理...")

            # 3. 清理已删除的记录
            with self.buffer_lock:
                if self.buffer_a is None:
                    return

                # 过滤已删除的记录
                buffer_b_data = [
                    item for item in self.buffer_a
                    if getattr(item, 'id', None) not in deleted_ids
                ]

                # 更新 chat_id 索引
                index_b = {}
                if self.index_a:
                    for k, v in self.index_a.items():
                        filtered = [
                            x for x in v
                            if getattr(x, 'id', None) not in deleted_ids
                        ]
                        if filtered:
                            index_b[k] = filtered

                # 更新总数
                self._tracker["total_count"] = len(buffer_b_data)

                # 原子切换
                self.buffer_b = buffer_b_data
                self.index_b = index_b
                self.buffer_a, self.buffer_b = self.buffer_b, None
                self.index_a, self.index_b = self.index_b, None

            logger.info(f"[ExprCache] 已清理 {len(deleted_ids)} 条已删除记录")

        except ImportError as e:
            logger.warning(f"[ExprCache] 无法导入数据库模型进行删除检测: {e}")
        except Exception as e:
            logger.error(f"[ExprCache] 删除检测失败: {e}")

    async def _load_to_buffer_b(self) -> None:
        """缓慢加载数据到缓冲区 B（异步）- 全量加载"""
        async with self.load_lock:
            if self.loading or self._stopped:
                return
            self.loading = True
        
        try:
            logger.info("[ExprCache] 开始全量加载表达式缓存到缓冲区B...")
            t0 = time.time()
            
            # 清空缓冲区 B
            buffer_b_data: List[Any] = []
            index_b: Dict[str, List[Any]] = {}
            max_id = 0
            
            # 尝试从数据库加载
            try:
                from src.common.database.database_model import Expression
                
                # 分批加载
                offset = 0
                while not self._stopped:
                    # 查询一批数据（使用 asyncio.to_thread 避免阻塞）
                    batch = await asyncio.to_thread(
                        lambda: list(Expression.select().limit(self.batch_size).offset(offset))
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
                    
                    # 构建索引（chat_id -> expressions）
                    for expr in batch:
                        chat_id = getattr(expr, "chat_id", None)
                        if not chat_id:
                            continue
                        index_b.setdefault(str(chat_id), []).append(expr)
                    
                    # 记录进度
                    if len(buffer_b_data) % 500 == 0:
                        logger.debug(f"[ExprCache] 加载进度: {len(buffer_b_data)} 条")
                    
                    # 休眠，避免 CPU 峰值
                    await asyncio.sleep(self.batch_delay)
                    offset += self.batch_size
                
            except ImportError as e:
                logger.warning(f"[ExprCache] 无法导入数据库模型: {e}")
                buffer_b_data = []
                index_b = {}
            except Exception as e:
                logger.error(f"[ExprCache] 加载数据失败: {e}")
                buffer_b_data = []
                index_b = {}
            
            if self._stopped:
                logger.info("[ExprCache] 加载被中止")
                return
            
            # 加载完成，原子切换
            with self.buffer_lock:
                self.buffer_b = buffer_b_data
                self.index_b = index_b

                # 原子切换：buffer_b/index_b → buffer_a/index_a
                self.buffer_a, self.buffer_b = self.buffer_b, None
                self.index_a, self.index_b = self.index_b, None

            # 使用过期管理器记录全量重建
            self._expiration_manager.record_full_rebuild(
                total_count=len(buffer_b_data),
                max_id=max_id,
            )
            # 同步更新兼容追踪器
            self._sync_tracker_from_manager()
            
            self.last_refresh = time.time()
            load_time = time.time() - t0
            logger.info(f"[ExprCache] 全量缓存加载完成并切换: {len(buffer_b_data)} 条, max_id={max_id}, 耗时 {load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"[ExprCache] 缓存加载失败: {e}")
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
                logger.info("[ExprCache] 触发全量重建...")
                await self._load_to_buffer_b()
            else:
                logger.info("[ExprCache] 触发增量刷新...")
                await self._load_incremental_to_buffer_b()
    
    def start(self) -> None:
        """启动缓存模块（启动定期刷新）"""
        self._stopped = False
        try:
            loop = asyncio.get_running_loop()
            if self._refresh_task is None or self._refresh_task.done():
                self._refresh_task = loop.create_task(self._refresh_loop())
                logger.info("[ExprCache] 定期刷新任务已启动（增量模式）")
            
            # 如果缓存未加载，立即触发加载
            if not self.is_loaded():
                loop.create_task(self._load_to_buffer_b())
        except RuntimeError:
            logger.warning("[ExprCache] 无法启动定期刷新：没有运行中的事件循环")
    
    def remove_patch(self) -> None:
        """移除所有补丁，恢复原始函数。

        在插件停用时调用，确保 ExpressionSelector 和 expression_manager
        恢复到未被 patch 的状态。
        """
        if not self._patched:
            return

        try:
            # 恢复 ExpressionSelector 的方法
            try:
                from src.bw_learner.expression_selector import ExpressionSelector  # type: ignore

                if self._orig_random_expressions is not None:
                    ExpressionSelector._random_expressions = self._orig_random_expressions
                if self._orig_select_simple is not None:
                    ExpressionSelector._select_expressions_simple = self._orig_select_simple
                if self._orig_select_classic is not None:
                    ExpressionSelector._select_expressions_classic = self._orig_select_classic
                logger.info("[ExprCache] 已恢复 ExpressionSelector 原始方法")
            except ImportError:
                pass
            except Exception as e:
                logger.error(f"[ExprCache] 恢复 ExpressionSelector 失败: {e}")

            # 恢复旧核心 expression_manager 的方法
            if self._expression_manager_ref is not None and self._orig_get_all is not None:
                try:
                    self._expression_manager_ref.get_all_expressions = self._orig_get_all
                    logger.info("[ExprCache] 已恢复 expression_manager.get_all_expressions")
                except Exception as e:
                    logger.error(f"[ExprCache] 恢复 expression_manager 失败: {e}")

            self._patched = False
            self._orig_random_expressions = None
            self._orig_select_simple = None
            self._orig_select_classic = None
            self._orig_get_all = None
            self._expression_manager_ref = None
            logger.info("[ExprCache] 所有补丁已移除")
        except Exception as e:
            logger.error(f"[ExprCache] 移除补丁失败: {e}")

    def stop(self) -> None:
        """停止缓存模块，移除补丁并取消刷新任务。"""
        self.remove_patch()
        self._stopped = True
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            self._refresh_task = None
        logger.info("[ExprCache] 缓存模块已停止")
    
    def refresh(self) -> None:
        """手动刷新缓存（非阻塞，触发全量刷新）"""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._load_to_buffer_b())
            logger.info("[ExprCache] 已触发手动全量刷新")
        except RuntimeError:
            logger.warning("[ExprCache] 无法触发刷新：没有运行中的事件循环")
    
    def clear(self) -> None:
        """清空缓存"""
        with self.buffer_lock:
            self.buffer_a = None
            self.buffer_b = None
            self.index_a = None
            self.index_b = None
            # 重置追踪器
            self._tracker = {
                "last_max_id": 0,
                "total_count": 0,
                "incremental_count": 0,
                "last_full_rebuild": 0.0,
                "incremental_refresh_count": 0,
            }
        logger.info("[ExprCache] 缓存已清空")
    
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
            if self.index_a is not None:
                size += MemoryUtils.get_size(self.index_a)
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
            # chat_id 索引统计
            "index_size": (
                sum(len(v) for v in self.index_a.values())
                if self.index_a
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


def apply_expression_cache(cache_manager) -> Optional[ExpressionCacheModule]:
    """应用表达式缓存补丁。

    MaiBot 核心已不再存在 `src.plugins.expression.expression_manager.get_all_expressions`。
    目前表达式高频读取点位于 `src.bw_learner.expression_selector.ExpressionSelector`
    内部对 `Expression.select().where(...)` 的全表/大表查询。

    本补丁策略：
    - 优先 patch `ExpressionSelector` 的查询路径，使用本模块的双缓冲缓存结果进行 Python 侧过滤。
    - 保留旧路径 patch（若某些分支仍存在 expression_manager，则继续兼容）。

    Args:
        cache_manager: 缓存管理器实例

    Returns:
        ExpressionCacheModule 实例，失败时返回 None
    """

    def _expr_to_dict(expr: Any) -> Dict[str, Any]:
        """将 Peewee Expression 模型转换为 selector 期望的 dict 结构。"""
        return {
            "id": getattr(expr, "id", None),
            "situation": getattr(expr, "situation", None),
            "style": getattr(expr, "style", None),
            "last_active_time": getattr(expr, "last_active_time", None),
            "source_id": getattr(expr, "chat_id", None),
            "create_date": (
                getattr(expr, "create_date", None)
                if getattr(expr, "create_date", None) is not None
                else getattr(expr, "last_active_time", None)
            ),
            "count": getattr(expr, "count", 1) if getattr(expr, "count", None) is not None else 1,
            "checked": bool(getattr(expr, "checked", False)),
            "rejected": bool(getattr(expr, "rejected", False)),
        }

    try:
        # 创建缓存实例
        cache = ExpressionCacheModule()

        # 注册到缓存管理器
        cache_manager.register_cache("expression_cache", cache)

        # ========== 兼容旧核心（若存在 expression_manager 则继续 patch） ==========
        try:
            from src.plugins.expression.expression_manager import expression_manager  # type: ignore

            original_get_all = expression_manager.get_all_expressions

            def patched_get_all():
                """补丁后的获取所有表达式函数（旧核心兼容）。"""
                if cache.is_loaded():
                    cache.stats.hit()
                    return cache.get_all()

                t0 = time.time()
                result = original_get_all()
                cache.stats.miss(time.time() - t0)
                return result

            expression_manager.get_all_expressions = patched_get_all
            cache._orig_get_all = original_get_all
            cache._expression_manager_ref = expression_manager
            cache._patched = True
            logger.info("[ExprCache] 已 patch expression_manager.get_all_expressions（兼容旧核心）")

        except ImportError:
            # 新核心无此模块
            pass
        except Exception as e:
            logger.warning(f"[ExprCache] patch expression_manager 失败: {e}")

        # ========== 新核心：patch ExpressionSelector 的查询 ==========
        try:
            from src.bw_learner.expression_selector import (  # type: ignore
                ExpressionSelector,
                expression_selector as expression_selector_instance,
            )
            from src.config.config import global_config  # type: ignore
            from src.bw_learner.learner_utils import weighted_sample  # type: ignore

            original_random_expressions = ExpressionSelector._random_expressions
            original_select_simple = ExpressionSelector._select_expressions_simple
            original_select_classic = ExpressionSelector._select_expressions_classic

            def _filter_models_for_selector(expr_models: List[Any], checked_only: bool) -> List[Dict[str, Any]]:
                style_exprs: List[Dict[str, Any]] = []
                for m in expr_models:
                    # 复刻 core 侧过滤：排除 rejected
                    if bool(getattr(m, "rejected", False)):
                        continue
                    if checked_only and not bool(getattr(m, "checked", False)):
                        continue
                    style_exprs.append(_expr_to_dict(m))
                return style_exprs

            def patched_random_expressions(self: Any, chat_id: str, total_num: int) -> List[Dict[str, Any]]:
                """从缓存中随机选择表达方式（替代 DB 全量查询）。"""
                if not cache.is_loaded():
                    t0 = time.time()
                    result = original_random_expressions(self, chat_id, total_num)
                    cache.stats.miss(time.time() - t0)
                    return result

                try:
                    related_chat_ids = self.get_related_chat_ids(chat_id)
                    expr_models = cache.get_by_chat_ids(related_chat_ids)
                    checked_only = bool(global_config.expression.expression_checked_only)
                    style_exprs = _filter_models_for_selector(expr_models, checked_only=checked_only)

                    if style_exprs:
                        selected_style = weighted_sample(style_exprs, min(len(style_exprs), total_num))
                    else:
                        selected_style = []

                    cache.stats.hit()
                    return selected_style

                except Exception as e:
                    logger.warning(f"[ExprCache] patched _random_expressions 失败，回退DB路径: {e}")
                    t0 = time.time()
                    result = original_random_expressions(self, chat_id, total_num)
                    cache.stats.miss(time.time() - t0)
                    return result

            def patched_select_simple(self: Any, chat_id: str, max_num: int):
                """simple 模式选择表达方式：优先使用缓存。"""
                if not cache.is_loaded():
                    t0 = time.time()
                    result = original_select_simple(self, chat_id, max_num)
                    cache.stats.miss(time.time() - t0)
                    return result

                try:
                    related_chat_ids = self.get_related_chat_ids(chat_id)
                    expr_models = cache.get_by_chat_ids(related_chat_ids)
                    checked_only = bool(global_config.expression.expression_checked_only)

                    style_exprs = []
                    for m in expr_models:
                        if bool(getattr(m, "rejected", False)):
                            continue
                        if checked_only and not bool(getattr(m, "checked", False)):
                            continue
                        count_val = getattr(m, "count", 1)
                        if (count_val or 1) <= 1:
                            continue
                        style_exprs.append(_expr_to_dict(m))

                    min_required = 8
                    if len(style_exprs) < min_required:
                        if not style_exprs:
                            fallback_num = min(3, max_num) if max_num > 0 else 3
                            fallback_selected = self._random_expressions(chat_id, fallback_num)
                            if fallback_selected:
                                self.update_expressions_last_active_time(fallback_selected)
                                selected_ids = [expr["id"] for expr in fallback_selected]
                                cache.stats.hit()
                                return fallback_selected, selected_ids
                            cache.stats.hit()
                            return [], []

                        select_count = min(3, len(style_exprs))
                    else:
                        select_count = 5

                    import random

                    selected_style = random.sample(style_exprs, select_count)
                    if selected_style:
                        self.update_expressions_last_active_time(selected_style)

                    selected_ids = [expr["id"] for expr in selected_style]
                    cache.stats.hit()
                    return selected_style, selected_ids

                except Exception as e:
                    logger.warning(f"[ExprCache] patched _select_expressions_simple 失败，回退DB路径: {e}")
                    t0 = time.time()
                    result = original_select_simple(self, chat_id, max_num)
                    cache.stats.miss(time.time() - t0)
                    return result

            async def patched_select_classic(
                self: Any,
                chat_id: str,
                chat_info: str,
                max_num: int = 10,
                target_message: Optional[str] = None,
                reply_reason: Optional[str] = None,
                think_level: int = 1,
            ):
                """classic 模式：用缓存替代 Expression.select().where(...) 的高频读取。"""
                if not cache.is_loaded():
                    t0 = time.time()
                    result = await original_select_classic(
                        self,
                        chat_id,
                        chat_info,
                        max_num=max_num,
                        target_message=target_message,
                        reply_reason=reply_reason,
                        think_level=think_level,
                    )
                    cache.stats.miss(time.time() - t0)
                    return result

                try:
                    # think_level == 0 直接复用 simple 模式（其内部已走缓存）
                    if think_level == 0:
                        result = self._select_expressions_simple(chat_id, max_num)
                        cache.stats.hit()
                        return result

                    related_chat_ids = self.get_related_chat_ids(chat_id)
                    expr_models = cache.get_by_chat_ids(related_chat_ids)
                    checked_only = bool(global_config.expression.expression_checked_only)

                    all_style_exprs = _filter_models_for_selector(expr_models, checked_only=checked_only)

                    high_count_exprs = [
                        expr
                        for expr in all_style_exprs
                        if (expr.get("count", 1) or 1) > 1
                    ]

                    min_high_count = 10
                    min_total_count = 10
                    select_high_count = 5
                    select_random_count = 5

                    if len(high_count_exprs) < min_high_count:
                        high_count_valid = False
                    else:
                        high_count_valid = True

                    if len(all_style_exprs) < min_total_count:
                        cache.stats.hit()
                        return [], []

                    selected_high = (
                        weighted_sample(high_count_exprs, min(len(high_count_exprs), select_high_count))
                        if high_count_valid
                        else []
                    )

                    selected_random = weighted_sample(
                        all_style_exprs, min(len(all_style_exprs), select_random_count)
                    )

                    candidate_exprs = selected_high.copy()
                    candidate_ids = {expr["id"] for expr in candidate_exprs}
                    for expr in selected_random:
                        if expr["id"] not in candidate_ids:
                            candidate_exprs.append(expr)
                            candidate_ids.add(expr["id"])

                    import random

                    random.shuffle(candidate_exprs)

                    all_expressions: List[Dict[str, Any]] = []
                    all_situations: List[str] = []
                    for expr in candidate_exprs:
                        expr = expr.copy()
                        all_expressions.append(expr)
                        all_situations.append(
                            f"{len(all_expressions)}.当 {expr['situation']} 时，使用 {expr['style']}"
                        )

                    if not all_expressions:
                        cache.stats.hit()
                        return [], []

                    all_situations_str = "\n".join(all_situations)

                    if target_message:
                        target_message_str = f'，现在你想要对这条消息进行回复："{target_message}"'
                        target_message_extra_block = "4.考虑你要回复的目标消息"
                    else:
                        target_message_str = ""
                        target_message_extra_block = ""

                    chat_context = f"以下是正在进行的聊天内容：{chat_info}"

                    if reply_reason:
                        reply_reason_block = f"你的回复理由是：{reply_reason}"
                        chat_context = ""
                    else:
                        reply_reason_block = ""

                    # 复用 ExpressionSelector 内部同名 prompt（通过 global_prompt_manager 获取）
                    from src.chat.utils.prompt_builder import global_prompt_manager  # type: ignore

                    prompt_tpl = await global_prompt_manager.get_prompt_async("expression_evaluation_prompt")
                    prompt = prompt_tpl.format(
                        bot_name=global_config.bot.nickname,
                        chat_observe_info=chat_context,
                        all_situations=all_situations_str,
                        max_num=max_num,
                        target_message=target_message_str,
                        target_message_extra_block=target_message_extra_block,
                        reply_reason_block=reply_reason_block,
                    )

                    content, _meta = await self.llm_model.generate_response_async(prompt=prompt)
                    if not content:
                        cache.stats.hit()
                        return [], []

                    from json_repair import repair_json  # type: ignore
                    import json as _json

                    result_obj = repair_json(content)
                    if isinstance(result_obj, str):
                        result_obj = _json.loads(result_obj)

                    if not isinstance(result_obj, dict) or "selected_situations" not in result_obj:
                        cache.stats.hit()
                        return [], []

                    selected_indices = result_obj["selected_situations"]
                    valid_expressions: List[Dict[str, Any]] = []
                    selected_ids: List[int] = []

                    for idx in selected_indices:
                        if isinstance(idx, int) and 1 <= idx <= len(all_expressions):
                            expression = all_expressions[idx - 1]
                            selected_ids.append(expression["id"])
                            valid_expressions.append(expression)

                    if valid_expressions:
                        self.update_expressions_last_active_time(valid_expressions)

                    cache.stats.hit()
                    return valid_expressions, selected_ids

                except Exception as e:
                    logger.warning(f"[ExprCache] patched _select_expressions_classic 失败，回退DB路径: {e}")
                    t0 = time.time()
                    result = await original_select_classic(
                        self,
                        chat_id,
                        chat_info,
                        max_num=max_num,
                        target_message=target_message,
                        reply_reason=reply_reason,
                        think_level=think_level,
                    )
                    cache.stats.miss(time.time() - t0)
                    return result

            ExpressionSelector._random_expressions = patched_random_expressions  # type: ignore[assignment]
            ExpressionSelector._select_expressions_simple = patched_select_simple  # type: ignore[assignment]
            ExpressionSelector._select_expressions_classic = patched_select_classic  # type: ignore[assignment]

            # 保存原始引用到 cache 实例，供 remove_patch() 恢复
            cache._orig_random_expressions = original_random_expressions
            cache._orig_select_simple = original_select_simple
            cache._orig_select_classic = original_select_classic
            cache._patched = True

            # 若实例已创建（模块导入时创建），确保其绑定到新方法：Python 会自动使用类方法解析
            if expression_selector_instance is not None:
                pass

            logger.info(
                "[ExprCache] 已 patch ExpressionSelector: _random_expressions/_select_expressions_simple/_select_expressions_classic"
            )

        except ImportError as e:
            logger.debug(f"[ExprCache] expression_selector 不可用，跳过 patch: {e}")
        except Exception as e:
            logger.warning(f"[ExprCache] patch expression_selector 失败: {e}")

        return cache

    except Exception as e:
        logger.error(f"[ExprCache] 初始化失败: {e}")
        return None
