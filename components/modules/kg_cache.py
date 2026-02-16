"""
知识库图谱缓存模块 - KGCacheModule

采用动态导入 core 模块，避免相对导入问题。
支持可选依赖的优雅降级（pandas, pyarrow, quick-algo）。
支持文件变更检测，避免无意义的重复加载。
集成 ExpirationManager 统一过期策略管理。
安全增强：路径遍历防护、JSON Schema 验证。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

# 为类型检查/静态分析提供更准确的提示；运行时依赖可缺失
if TYPE_CHECKING:
    import pandas as pd  # noqa: F401

try:
    import aiofiles

    AIOFILES_AVAILABLE = True
except ImportError:
    aiofiles = None
    AIOFILES_AVAILABLE = False

# 可选依赖检测（本插件必须在缺失依赖时仍可被导入）
PANDAS_AVAILABLE = False
PYARROW_AVAILABLE = False
QUICK_ALGO_AVAILABLE = False

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    pd = None

try:
    import pyarrow

    PYARROW_AVAILABLE = True
except ImportError:
    pyarrow = None

try:
    from quick_algo import di_graph

    QUICK_ALGO_AVAILABLE = True
except ImportError:
    di_graph = None


CACHE_REQUIRED_DEPENDENCIES: tuple[str, ...] = ("quick-algo", "pandas", "pyarrow")
CACHE_OPTIONAL_DEPENDENCIES: tuple[str, ...] = ("aiofiles",)


def _get_missing_required_cache_deps() -> List[str]:
    """获取"启用 KGCache 预加载/命中"所需的缺失依赖。

    约定：
    - 必需依赖仅包含：quick-algo
    - pandas/pyarrow 仅用于 parquet 加速读取；缺失时允许回退到"从图重建元数据"路径
    """

    missing: List[str] = []
    if not QUICK_ALGO_AVAILABLE:
        missing.append("quick-algo")
    return missing


def _dependency_status() -> Dict[str, Any]:
    """生成依赖状态信息（用于日志/诊断输出）。"""

    required = {
        "quick-algo": QUICK_ALGO_AVAILABLE,
    }
    optional = {
        "pandas": PANDAS_AVAILABLE,
        "pyarrow": PYARROW_AVAILABLE,
        "aiofiles": AIOFILES_AVAILABLE,
    }
    missing_required = [name for name, ok in required.items() if not ok]
    missing_optional = [name for name, ok in optional.items() if not ok]
    return {
        "required": required,
        "optional": optional,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "cache_ready": len(missing_required) == 0,
        "parquet_reader_ready": PANDAS_AVAILABLE and PYARROW_AVAILABLE,
    }

try:
    from src.common.logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger("CM_perf_opt")


# 从公共模块导入动态加载函数和安全验证
try:
    from core.compat import (
        load_core_module,
        validate_file_path,
        validate_json_schema,
        safe_load_json_file,
        PARAGRAPH_HASH_SCHEMA,
        CoreModuleLoadError,
        PathTraversalError,
        JsonValidationError,
    )
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
    
    def validate_file_path(file_path, base_dir, allow_create=False):
        """Fallback validate_file_path 实现"""
        file_path = Path(file_path).resolve()
        base_dir = Path(base_dir).resolve()
        try:
            file_path.relative_to(base_dir)
        except ValueError:
            raise ValueError(f"路径遍历风险: '{file_path}' 不在 '{base_dir}' 内")
        return file_path
    
    def validate_json_schema(data, schema, path_hint=None):
        """Fallback validate_json_schema 实现"""
        # 基础验证
        if "type" in schema:
            expected = schema["type"]
            if expected == "object" and not isinstance(data, dict):
                return False, f"期望 object 类型"
            if expected == "array" and not isinstance(data, list):
                return False, f"期望 array 类型"
        if "required" in schema and isinstance(data, dict):
            for field in schema["required"]:
                if field not in data:
                    return False, f"缺少必需字段 '{field}'"
        return True, None
    
    def safe_load_json_file(file_path, base_dir, schema=None, encoding="utf-8"):
        """Fallback safe_load_json_file 实现"""
        try:
            validated_path = validate_file_path(file_path, base_dir)
            with open(validated_path, "r", encoding=encoding) as f:
                data = json.load(f)
            if schema:
                valid, error = validate_json_schema(data, schema)
                if not valid:
                    return None, error
            return data, None
        except Exception as e:
            return None, str(e)
    
    PARAGRAPH_HASH_SCHEMA = {
        "type": "object",
        "required": ["stored_paragraph_hashes"],
        "properties": {
            "stored_paragraph_hashes": {"type": "array", "items": {"type": "string"}}
        }
    }
    
    class CoreModuleLoadError(ImportError):
        """Core 模块加载失败异常"""
        pass
    
    class PathTraversalError(ValueError):
        """路径遍历安全异常"""
        pass
    
    class JsonValidationError(ValueError):
        """JSON Schema 验证失败异常"""
        pass


# 动态导入核心模块
try:
    # 优先尝试相对导入（在正确的包结构中）
    from ..core import ModuleStats, MemoryUtils
    from ..core import (
        ExpirationConfig,
        ExpirationManager,
        FileExpirationManager,
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
        FileExpirationManager = _core.FileExpirationManager
        RefreshDecision = _core.RefreshDecision
    except (ImportError, CoreModuleLoadError) as e:
        logger.warning(f"[KGCache] 无法加载核心模块，使用内置实现: {e}")
        
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
        from enum import Enum
        from dataclasses import dataclass, field
        
        class RefreshDecision(Enum):
            """刷新决策枚举"""
            FULL_REBUILD = "full_rebuild"
            INCREMENTAL = "incremental"
            SKIP = "skip"
        
        @dataclass
        class ExpirationConfig:
            """过期配置"""
            incremental_refresh_interval: float = 600.0
            full_rebuild_interval: float = 86400.0
            incremental_threshold_ratio: float = 0.1
            deletion_check_interval: int = 10
        
        @dataclass
        class ExpirationState:
            """过期状态"""
            last_full_rebuild: float = 0.0
            last_incremental_refresh: float = 0.0
            total_count: int = 0
            incremental_count: int = 0
            incremental_refresh_count: int = 0
            cumulative_incremental_count: int = 0
        
        class ExpirationManager:
            """Fallback ExpirationManager 基类"""
            def __init__(self, config: "ExpirationConfig", name: str = "cache"):
                self.config = config
                self.name = name
                self._state = ExpirationState()
                self._lock = threading.Lock()
            
            def get_refresh_decision(self) -> "RefreshDecision":
                with self._lock:
                    now = time.time()
                    if self._state.last_full_rebuild == 0:
                        return RefreshDecision.FULL_REBUILD
                    if now - self._state.last_full_rebuild >= self.config.full_rebuild_interval:
                        return RefreshDecision.FULL_REBUILD
                    if self._state.cumulative_incremental_count > self._state.total_count * self.config.incremental_threshold_ratio:
                        return RefreshDecision.FULL_REBUILD
                    if now - self._state.last_incremental_refresh < self.config.incremental_refresh_interval:
                        return RefreshDecision.SKIP
                    return RefreshDecision.INCREMENTAL
            
            def record_full_rebuild(self, total_count: int, max_id: int = 0):
                with self._lock:
                    self._state.last_full_rebuild = time.time()
                    self._state.total_count = total_count
                    self._state.incremental_count = 0
                    self._state.incremental_refresh_count = 0
                    self._state.cumulative_incremental_count = 0
            
            def record_incremental_refresh(self, new_count: int, new_max_id: Optional[int] = None):
                with self._lock:
                    self._state.last_incremental_refresh = time.time()
                    self._state.incremental_count = new_count
                    self._state.incremental_refresh_count += 1
                    self._state.cumulative_incremental_count += new_count
            
            def get_state_dict(self) -> Dict[str, Any]:
                with self._lock:
                    return {
                        "last_full_rebuild": self._state.last_full_rebuild,
                        "last_incremental_refresh": self._state.last_incremental_refresh,
                        "total_count": self._state.total_count,
                        "incremental_count": self._state.incremental_count,
                        "incremental_refresh_count": self._state.incremental_refresh_count,
                        "cumulative_incremental_count": self._state.cumulative_incremental_count,
                    }
            
            @property
            def state(self) -> "ExpirationState":
                return self._state
        
        class FileExpirationManager(ExpirationManager):
            """Fallback FileExpirationManager - 用于文件缓存"""
            def __init__(self, config: "ExpirationConfig", name: str = "file_cache"):
                super().__init__(config, name)
                self._file_mtimes: Dict[str, float] = {}
                self._file_sizes: Dict[str, int] = {}
            
            def update_file_stats(self, file_path: str, mtime: float, size: int):
                with self._lock:
                    self._file_mtimes[file_path] = mtime
                    self._file_sizes[file_path] = size
            
            def get_file_stats(self, file_path: str) -> Dict[str, Any]:
                with self._lock:
                    return {
                        "mtime": self._file_mtimes.get(file_path, 0.0),
                        "size": self._file_sizes.get(file_path, 0),
                    }
            
            def has_file_changed(self, file_path: str, current_mtime: float, current_size: int) -> bool:
                with self._lock:
                    stored_mtime = self._file_mtimes.get(file_path, 0.0)
                    stored_size = self._file_sizes.get(file_path, 0)
                    return stored_mtime != current_mtime or stored_size != current_size


class KGCacheModule:
    """知识库图谱全量缓存 - 双缓冲 + 缓慢加载 + 原子切换。

    设计目标：
    - 插件必须可导入：即使缺少可选依赖也不会 ImportError
    - 缺失依赖时必须“明确告知影响范围”，避免静默失败

    功能：
    - 双缓冲设计：buffer_a 为当前使用，buffer_b 为后台加载
    - 缓慢加载：分批从文件加载，避免 CPU 峰值
    - 原子切换：加载完成后原子交换缓冲区
    - 定期刷新：支持自动和手动刷新

    依赖说明：
    - 必需（启用 KGCache 预加载/命中所需）：
        - quick-algo：提供 di_graph / DiGraph / load_from_file
    - 可选（仅影响性能/加载路径，不影响正确性）：
        - pandas + pyarrow：读取实体计数 parquet（rag-ent-cnt.parquet）。缺失时改为从图结构重建 ent_appear_cnt（更慢）。
        - aiofiles：异步读取段落 hash JSON（rag-pg-hash.json）。缺失时回退到同步读取 + asyncio.to_thread。

    Attributes:
        buffer_a: 当前使用的缓存标记
        graph_a: 当前使用的图对象
        nodes_a: 当前使用的节点列表
        edges_a: 当前使用的边列表
        ent_appear_cnt_a: 当前使用的实体出现次数
        stored_paragraph_hashes_a: 当前使用的段落 hash 集合
        batch_size: 每批加载的条目数
        batch_delay: 批次间的延迟（秒）
        refresh_interval: 自动刷新间隔（秒）
        stats: 统计信息
    """

    # 缓存依赖可用性（类级别）
    CACHE_ENABLED = QUICK_ALGO_AVAILABLE
    PARQUET_READER_AVAILABLE = PANDAS_AVAILABLE and PYARROW_AVAILABLE

    # 兼容旧字段名（外部可能引用），语义为“缓存是否可启用”
    DEPENDENCIES_AVAILABLE = CACHE_ENABLED
    
    def __init__(
        self,
        batch_size: int = 100,
        batch_delay: float = 0.05,
        refresh_interval: int = 3600,
        expiration_config: Optional["ExpirationConfig"] = None
    ):
        """初始化知识库图谱缓存模块
        
        Args:
            batch_size: 每批加载的条目数，默认 100
            batch_delay: 批次间的延迟秒数，默认 0.05
            refresh_interval: 自动刷新间隔秒数，默认 3600
            expiration_config: 过期管理配置，若为 None 则使用默认配置
        """
        # 双缓冲
        self.buffer_a: Optional[bool] = None
        self.buffer_b: Optional[bool] = None
        self.buffer_lock = threading.Lock()
        
        # 缓存内容 - 当前使用
        self.graph_a: Any = None
        self.nodes_a: Optional[List[Any]] = None
        self.edges_a: Optional[List[Any]] = None
        self.ent_appear_cnt_a: Optional[Dict[str, float]] = None
        self.stored_paragraph_hashes_a: Optional[Set[str]] = None
        
        # 缓存内容 - 后台加载
        self.graph_b: Any = None
        self.nodes_b: Optional[List[Any]] = None
        self.edges_b: Optional[List[Any]] = None
        self.ent_appear_cnt_b: Optional[Dict[str, float]] = None
        self.stored_paragraph_hashes_b: Optional[Set[str]] = None
        
        # 加载配置
        self.batch_size = max(1, int(batch_size))
        self.batch_delay = max(0.001, float(batch_delay))
        self.refresh_interval = max(60, int(refresh_interval))
        
        # 状态
        self.loading = False
        self.load_lock = asyncio.Lock()
        self._sync_load_lock = threading.Lock()
        self.last_refresh: float = 0.0
        self.stats = ModuleStats("kg_cache")

        # 启动时序补偿注入：缓存就绪后，将数据注入已创建的 qa_manager.kg_manager
        self._injected_once = False
        self._injection_lock = threading.Lock()

        # 无事件循环时的初始加载线程（用于尽可能提前预热，降低主线程启动卡顿）
        self._initial_loader_thread: Optional[threading.Thread] = None
        
        # 刷新任务
        self._refresh_task: Optional[asyncio.Task] = None
        self._stopped = False
        
        # 降级模式标记：仅当缺少 quick-algo 时完全禁用缓存
        # pandas/pyarrow 缺失仍可通过"从图重建元数据"实现缓存加载
        self._degraded_mode = not self.CACHE_ENABLED

        # 文件变更追踪器（用于增量更新检测）
        self._tracker: Dict[str, Any] = {
            "graph_mtime": 0.0,
            "graph_size": 0,
            "ent_cnt_mtime": 0.0,
            "ent_cnt_size": 0,
            "pg_hash_mtime": 0.0,
            "pg_hash_size": 0,
            "total_count": 0,
            "last_full_rebuild": 0.0,
            "skip_count": 0,  # 跳过刷新的次数
            "file_path": None,  # 缓存文件路径（首次加载后设置）
        }
        
        # 过期管理器 - 用于文件缓存
        if expiration_config is None:
            expiration_config = ExpirationConfig(
                incremental_refresh_interval=600.0,
                full_rebuild_interval=86400.0,
                incremental_threshold_ratio=0.1,
                deletion_check_interval=10
            )
        self._expiration_manager = FileExpirationManager(expiration_config, name="kg_cache")

        dep = _dependency_status()
        if self._degraded_mode:
            logger.warning(
                "[KGCache] 缓存已禁用（降级模式）：缺少必需依赖 %s。"
                " 影响：不会预加载 KG 图谱缓存，KGManager.load_from_file 将走原始读盘路径。",
                dep["missing_required"],
            )
        else:
            if dep["missing_optional"]:
                logger.info(
                    "[KGCache] 缓存已启用：可选依赖缺失 %s，将回退到较慢实现（不影响功能正确性）。",
                    dep["missing_optional"],
                )
            if not self.PARQUET_READER_AVAILABLE:
                logger.warning(
                    "[KGCache] 未检测到 pandas/pyarrow：无法读取 parquet 实体计数文件（rag-ent-cnt.parquet）。"
                    " 将改为从图结构重建 ent_appear_cnt（首次加载更慢）。"
                )
        
        # 启动时立即开始加载
        self._schedule_initial_load()
    
    def _schedule_initial_load(self) -> None:
        """调度初始加载任务。

        启动时序问题：`src.chat.knowledge.lpmm_start_up()` 会在启动阶段同步调用
        `KGManager.load_from_file()`。

        - 若插件补丁应用过晚：将导致 kg_cache 统计 0 命中，且无法降低主线程启动卡顿。
        - 若插件被较早导入但此时尚无 asyncio loop：这里启动后台线程进行“同步预热加载”，
          以尽可能抢在 `lpmm_start_up()` 之前将缓存准备好。
        """
        if self._degraded_mode:
            logger.info("[KGCache] 降级模式：跳过初始加载")
            return

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._load_to_buffer_b())
            logger.debug("[KGCache] 已调度初始加载任务（asyncio task）")
        except RuntimeError:
            logger.info("[KGCache] 未检测到运行中的事件循环，启用后台线程进行初始预热加载")
            self._start_background_initial_loader_thread()

    def _start_background_initial_loader_thread(self) -> None:
        """在没有运行中事件循环时启动后台线程预热加载。"""
        if self._stopped or self._degraded_mode:
            return

        t = self._initial_loader_thread
        if t is not None and t.is_alive():
            return

        def _runner() -> None:
            try:
                self._load_to_buffer_b_sync()
            except Exception as e:
                logger.warning(f"[KGCache] 后台初始加载线程异常: {e}")

        self._initial_loader_thread = threading.Thread(
            target=_runner,
            name="kg_cache_initial_loader",
            daemon=True,
        )
        self._initial_loader_thread.start()

    def _load_to_buffer_b_sync(self) -> None:
        """同步版预热加载（用于后台线程）。

        注意：
        - 该函数**不得**触碰 asyncio 相关对象（例如 self.load_lock），避免跨事件循环绑定问题。
        - 只在“没有运行中的事件循环”时作为兜底预热机制使用。
        """
        if self._degraded_mode or self._stopped:
            return

        with self._sync_load_lock:
            if self.loading or self._stopped:
                return
            self.loading = True

            try:
                logger.info("[KGCache] (sync) 开始后台预热加载知识库图谱缓存...")
                t0 = time.time()

                # 尝试获取 KGManager 路径
                from src.chat.knowledge.kg_manager import KGManager

                kg_manager = KGManager()
                graph_data_path = kg_manager.graph_data_path
                ent_cnt_data_path = kg_manager.ent_cnt_data_path
                pg_hash_file_path = kg_manager.pg_hash_file_path

                # 安全验证：确保路径在预期目录内（防止路径遍历攻击）
                try:
                    # 获取知识库数据目录作为基准目录
                    kg_data_dir = Path(graph_data_path).parent
                    validate_file_path(graph_data_path, kg_data_dir)
                    validate_file_path(ent_cnt_data_path, kg_data_dir)
                    validate_file_path(pg_hash_file_path, kg_data_dir)
                except (PathTraversalError, ValueError) as e:
                    logger.error(f"[KGCache] (sync) 路径安全验证失败: {e}")
                    return

                if not os.path.exists(graph_data_path):
                    logger.warning(f"[KGCache] (sync) 知识库图谱文件不存在: {graph_data_path}")
                    return

                if di_graph is None:
                    raise ImportError("quick-algo not available")

                graph_b = di_graph.load_from_file(graph_data_path)
                nodes_b = graph_b.get_node_list()
                edges_b = graph_b.get_edge_list()

                # 加载实体计数（优先 parquet；缺失依赖时回退到“从图重建”）
                ent_appear_cnt_b: Dict[str, float] = {}
                parquet_loaded = False
                if self.PARQUET_READER_AVAILABLE and pd is not None and os.path.exists(ent_cnt_data_path):
                    try:
                        ent_cnt_df = pd.read_parquet(ent_cnt_data_path, engine="pyarrow")
                        ent_appear_cnt_b = dict({
                            row["hash_key"]: row["appear_cnt"]
                            for _, row in ent_cnt_df.iterrows()
                        })
                        parquet_loaded = True
                    except Exception as e:
                        logger.warning(
                            "[KGCache] (sync) 读取 parquet 实体计数失败，将回退到从图重建。原因: %s",
                            e,
                        )

                if (not parquet_loaded) and (graph_b is not None):
                    try:
                        ent_appear_cnt_rebuilt: Dict[str, float] = {}
                        for edge_tuple in edges_b:
                            src, tgt = edge_tuple[0], edge_tuple[1]
                            if isinstance(src, str) and isinstance(tgt, str):
                                if src.startswith("entity") and tgt.startswith("paragraph"):
                                    try:
                                        edge_data = graph_b[src, tgt]
                                    except Exception:
                                        edge_data = {}
                                    weight = edge_data.get("weight", 1.0) if isinstance(edge_data, dict) else 1.0
                                    ent_appear_cnt_rebuilt[src] = ent_appear_cnt_rebuilt.get(src, 0.0) + float(weight)

                        ent_appear_cnt_b = ent_appear_cnt_rebuilt
                    except Exception as e:
                        logger.warning(f"[KGCache] (sync) 从图结构重建实体计数失败，将使用空 ent_appear_cnt。原因: {e}")
                        ent_appear_cnt_b = {}

                # 加载段落 hash（带 JSON Schema 验证）
                kg_data_dir = Path(graph_data_path).parent
                data, error = safe_load_json_file(
                    pg_hash_file_path,
                    kg_data_dir,
                    schema=PARAGRAPH_HASH_SCHEMA,
                )
                if data is not None:
                    stored_paragraph_hashes_b = set(data.get("stored_paragraph_hashes", []))
                elif error and "不存在" in str(error):
                    logger.warning(f"[KGCache] (sync) 段落 hash 文件不存在: {pg_hash_file_path}")
                    stored_paragraph_hashes_b = set()
                else:
                    logger.warning(f"[KGCache] (sync) 加载段落 hash 失败: {error}")
                    stored_paragraph_hashes_b = set()

                if self._stopped:
                    return

                # 分批延迟：避免 CPU 峰值
                total_items = len(nodes_b) + len(edges_b) + len(ent_appear_cnt_b)
                batches = max(1, (total_items + self.batch_size - 1) // self.batch_size)
                for _ in range(min(batches, 100)):
                    time.sleep(self.batch_delay)
                    if self._stopped:
                        return

                # 原子切换
                with self.buffer_lock:
                    self.buffer_b = True
                    self.graph_b = graph_b
                    self.nodes_b = nodes_b
                    self.edges_b = edges_b
                    self.ent_appear_cnt_b = ent_appear_cnt_b
                    self.stored_paragraph_hashes_b = stored_paragraph_hashes_b

                    self.buffer_a, self.buffer_b = self.buffer_b, None
                    self.graph_a, self.graph_b = self.graph_b, None
                    self.nodes_a, self.nodes_b = self.nodes_b, None
                    self.edges_a, self.edges_b = self.edges_b, None
                    self.ent_appear_cnt_a, self.ent_appear_cnt_b = self.ent_appear_cnt_b, None
                    self.stored_paragraph_hashes_a, self.stored_paragraph_hashes_b = self.stored_paragraph_hashes_b, None

                # 更新文件变更追踪器
                total_count = len(nodes_b) + len(edges_b) + len(ent_appear_cnt_b)
                self._update_tracker(graph_data_path, ent_cnt_data_path, pg_hash_file_path, total_count)

                self.last_refresh = time.time()
                load_time = time.time() - t0
                logger.info(
                    f"[KGCache] (sync) 缓存预热完成并切换: 节点 {len(nodes_b)} 个, 边 {len(edges_b)} 条, 耗时 {load_time:.2f}s"
                )

                # 预热完成后尝试注入已存在的 qa_manager.kg_manager
                try:
                    self.try_inject_into_runtime()
                except Exception:
                    pass

            finally:
                self.loading = False

    def try_inject_into_runtime(self) -> bool:
        """启动时序补偿：将缓存数据注入已经创建的 `qa_manager.kg_manager`。

        Returns:
            True 表示成功注入或已注入；False 表示条件不满足（缓存未就绪/qa_manager 不存在等）。
        """
        if self._degraded_mode:
            return False

        with self._injection_lock:
            if self._injected_once:
                return True

            cached = self.get_cached_data()
            if not cached or cached.get("graph") is None:
                return False

            try:
                import src.chat.knowledge as knowledge_pkg

                qa_manager = getattr(knowledge_pkg, "qa_manager", None)
                if qa_manager is None:
                    return False

                kg_manager = getattr(qa_manager, "kg_manager", None)
                if kg_manager is None:
                    return False

                kg_manager.graph = cached["graph"]
                kg_manager.ent_appear_cnt = cached.get("ent_appear_cnt") or {}
                kg_manager.stored_paragraph_hashes = cached.get("stored_paragraph_hashes") or set()

                # 统计：将“补偿注入”计为一次 hit，避免报告中持续 0 命中
                try:
                    self.stats.hit()
                except Exception:
                    pass

                self._injected_once = True
                logger.info("[KGCache] 已将缓存数据注入 qa_manager.kg_manager（启动时序补偿）")
                return True
            except Exception:
                return False
    
    def get_cached_data(self) -> Optional[Dict[str, Any]]:
        """获取当前缓存的图数据
        
        Returns:
            包含图数据的字典，未加载时返回 None
        """
        with self.buffer_lock:
            if self.buffer_a is None:
                return None
            return {
                "graph": self.graph_a,
                "nodes": self.nodes_a,
                "edges": self.edges_a,
                "ent_appear_cnt": self.ent_appear_cnt_a,
                "stored_paragraph_hashes": self.stored_paragraph_hashes_a,
            }
    
    def is_loaded(self) -> bool:
        """检查缓存是否已加载完成
        
        Returns:
            True 表示缓存已加载
        """
        with self.buffer_lock:
            return self.buffer_a is not None
    
    def is_degraded(self) -> bool:
        """检查是否处于降级模式
        
        Returns:
            True 表示处于降级模式
        """
        return self._degraded_mode

    def _get_file_stats(self, file_path: str) -> Dict[str, Any]:
        """获取文件状态信息（mtime 和 size）。

        Args:
            file_path: 文件路径

        Returns:
            包含 mtime 和 size 的字典，文件不存在时返回空值
        """
        try:
            if os.path.exists(file_path):
                stat_info = os.stat(file_path)
                return {
                    "mtime": stat_info.st_mtime,
                    "size": stat_info.st_size,
                }
        except OSError as e:
            logger.debug(f"[KGCache] 获取文件状态失败 {file_path}: {e}")
        return {"mtime": 0.0, "size": 0}

    def _check_file_changes(self, graph_path: str, ent_cnt_path: str, pg_hash_path: str) -> Dict[str, bool]:
        """检测文件变更。

        通过比较文件的 mtime 和 size 来判断是否有变更。

        Args:
            graph_path: 图谱文件路径
            ent_cnt_path: 实体计数文件路径
            pg_hash_path: 段落 hash 文件路径

        Returns:
            包含各文件变更状态的字典
        """
        changes = {
            "graph_changed": False,
            "ent_cnt_changed": False,
            "pg_hash_changed": False,
            "any_changed": False,
        }

        # 检查图谱文件
        graph_stats = self._get_file_stats(graph_path)
        if graph_stats["mtime"] > 0:
            if (graph_stats["mtime"] != self._tracker["graph_mtime"] or
                graph_stats["size"] != self._tracker["graph_size"]):
                changes["graph_changed"] = True

        # 检查实体计数文件
        ent_cnt_stats = self._get_file_stats(ent_cnt_path)
        if ent_cnt_stats["mtime"] > 0:
            if (ent_cnt_stats["mtime"] != self._tracker["ent_cnt_mtime"] or
                ent_cnt_stats["size"] != self._tracker["ent_cnt_size"]):
                changes["ent_cnt_changed"] = True

        # 检查段落 hash 文件
        pg_hash_stats = self._get_file_stats(pg_hash_path)
        if pg_hash_stats["mtime"] > 0:
            if (pg_hash_stats["mtime"] != self._tracker["pg_hash_mtime"] or
                pg_hash_stats["size"] != self._tracker["pg_hash_size"]):
                changes["pg_hash_changed"] = True

        # 汇总是否有任何变更
        changes["any_changed"] = any([
            changes["graph_changed"],
            changes["ent_cnt_changed"],
            changes["pg_hash_changed"],
        ])

        return changes

    def _update_tracker(self, graph_path: str, ent_cnt_path: str, pg_hash_path: str, total_count: int) -> None:
        """更新文件变更追踪器。

        Args:
            graph_path: 图谱文件路径
            ent_cnt_path: 实体计数文件路径
            pg_hash_path: 段落 hash 文件路径
            total_count: 缓存总条目数
        """
        graph_stats = self._get_file_stats(graph_path)
        ent_cnt_stats = self._get_file_stats(ent_cnt_path)
        pg_hash_stats = self._get_file_stats(pg_hash_path)

        self._tracker["graph_mtime"] = graph_stats["mtime"]
        self._tracker["graph_size"] = graph_stats["size"]
        self._tracker["ent_cnt_mtime"] = ent_cnt_stats["mtime"]
        self._tracker["ent_cnt_size"] = ent_cnt_stats["size"]
        self._tracker["pg_hash_mtime"] = pg_hash_stats["mtime"]
        self._tracker["pg_hash_size"] = pg_hash_stats["size"]
        self._tracker["total_count"] = total_count
        self._tracker["last_full_rebuild"] = time.time()
        self._tracker["file_path"] = graph_path

    def _should_reload(self, graph_path: str, ent_cnt_path: str, pg_hash_path: str) -> bool:
        """判断是否需要重新加载（基于文件变更检测）。

        Args:
            graph_path: 图谱文件路径
            ent_cnt_path: 实体计数文件路径
            pg_hash_path: 段落 hash 文件路径

        Returns:
            True 表示需要重新加载
        """
        # 首次加载
        with self.buffer_lock:
            if self.buffer_a is None:
                return True

        # 检测文件变更
        changes = self._check_file_changes(graph_path, ent_cnt_path, pg_hash_path)
        return changes["any_changed"]
    
    def _should_full_rebuild(self) -> bool:
        """判断是否需要全量重建（基于 ExpirationManager）。
        
        Returns:
            True 表示需要全量重建
        """
        decision = self._get_refresh_decision()
        return decision == RefreshDecision.FULL_REBUILD
    
    def _get_refresh_decision(self) -> "RefreshDecision":
        """获取刷新决策（基于 ExpirationManager）。
        
        Returns:
            RefreshDecision 枚举值
        """
        return self._expiration_manager.get_refresh_decision()
    
    def _sync_tracker_from_manager(self) -> None:
        """将 ExpirationManager 的状态同步到 _tracker（用于兼容旧接口）。"""
        state = self._expiration_manager.get_state_dict()
        self._tracker["last_full_rebuild"] = state.get("last_full_rebuild", 0.0)
        self._tracker["total_count"] = state.get("total_count", 0)
        self._tracker["skip_count"] = state.get("incremental_refresh_count", 0)
    
    async def _load_to_buffer_b(self) -> None:
        """缓慢加载数据到缓冲区 B（异步）"""
        if self._degraded_mode:
            logger.debug("[KGCache] 降级模式：跳过加载")
            return
        
        async with self.load_lock:
            if self.loading or self._stopped:
                return
            self.loading = True
        
        try:
            logger.info("[KGCache] 开始缓慢加载知识库图谱缓存到缓冲区B...")
            t0 = time.time()
            
            # 尝试获取 KGManager 路径
            try:
                from src.chat.knowledge.kg_manager import KGManager
                kg_manager = KGManager()
                graph_data_path = kg_manager.graph_data_path
                ent_cnt_data_path = kg_manager.ent_cnt_data_path
                pg_hash_file_path = kg_manager.pg_hash_file_path
            except ImportError as e:
                logger.warning(f"[KGCache] 无法导入 KGManager: {e}")
                async with self.load_lock:
                    self.loading = False
                return
            except Exception as e:
                logger.error(f"[KGCache] 获取 KGManager 失败: {e}")
                async with self.load_lock:
                    self.loading = False
                return
            
            # 检查文件是否存在
            file_exists = await asyncio.to_thread(os.path.exists, graph_data_path)
            if not file_exists:
                logger.warning(f"[KGCache] 知识库图谱文件不存在: {graph_data_path}")
                async with self.load_lock:
                    self.loading = False
                return
            
            # 加载图谱
            try:
                if di_graph is None:
                    raise ImportError("quick-algo not available")
                
                graph_b = await asyncio.to_thread(di_graph.load_from_file, graph_data_path)
                nodes_b = graph_b.get_node_list()
                edges_b = graph_b.get_edge_list()
                
                logger.debug(f"[KGCache] 加载图谱: {len(nodes_b)} 个节点, {len(edges_b)} 条边")
            except Exception as e:
                logger.error(f"[KGCache] 加载图谱失败: {e}")
                async with self.load_lock:
                    self.loading = False
                return
            
            # 加载实体计数（优先 parquet；缺失依赖时回退到“从图重建”）
            ent_appear_cnt_b: Dict[str, float] = {}
            parquet_loaded = False
            if self.PARQUET_READER_AVAILABLE and pd is not None and os.path.exists(ent_cnt_data_path):
                try:
                    ent_cnt_df = await asyncio.to_thread(
                        pd.read_parquet, ent_cnt_data_path, engine="pyarrow"
                    )
                    ent_appear_cnt_b = dict({
                        row["hash_key"]: row["appear_cnt"]
                        for _, row in ent_cnt_df.iterrows()
                    })
                    parquet_loaded = True
                    logger.debug(
                        "[KGCache] 已从 parquet 加载实体计数: %d 个实体",
                        len(ent_appear_cnt_b),
                    )
                except Exception as e:
                    # 不致命：可回退到从图重建
                    logger.warning(
                        "[KGCache] 读取 parquet 实体计数失败，将回退到从图重建。原因: %s",
                        e,
                    )

            if (not parquet_loaded) and (graph_b is not None):
                # 从图结构重建 ent_appear_cnt（参考 KGManager._rebuild_metadata_from_graph 逻辑）
                try:
                    ent_appear_cnt_rebuilt: Dict[str, float] = {}
                    for edge_tuple in edges_b:
                        src, tgt = edge_tuple[0], edge_tuple[1]
                        if isinstance(src, str) and isinstance(tgt, str):
                            if src.startswith("entity") and tgt.startswith("paragraph"):
                                try:
                                    edge_data = graph_b[src, tgt]
                                except Exception:
                                    edge_data = {}
                                weight = edge_data.get("weight", 1.0) if isinstance(edge_data, dict) else 1.0
                                ent_appear_cnt_rebuilt[src] = ent_appear_cnt_rebuilt.get(src, 0.0) + float(weight)

                    ent_appear_cnt_b = ent_appear_cnt_rebuilt
                    logger.debug(
                        f"[KGCache] 已从图结构重建实体计数: {len(ent_appear_cnt_b)} 个实体"
                    )
                except Exception as e:
                    logger.warning(f"[KGCache] 从图结构重建实体计数失败，将使用空 ent_appear_cnt。原因: {e}")
                    ent_appear_cnt_b = {}
            
            # 加载段落 hash（带 JSON Schema 验证）
            try:
                # 先验证路径安全性
                kg_data_dir = Path(graph_data_path).parent
                try:
                    validated_path = validate_file_path(pg_hash_file_path, kg_data_dir)
                except (PathTraversalError, ValueError) as path_err:
                    logger.error(f"[KGCache] 路径安全验证失败: {path_err}")
                    stored_paragraph_hashes_b = set()
                else:
                    if AIOFILES_AVAILABLE and aiofiles is not None:
                        async with aiofiles.open(validated_path, "r", encoding="utf-8") as f:
                            content = await f.read()
                            data = json.loads(content)
                    else:
                        # 降级到同步方式
                        def load_pg_hash():
                            with open(validated_path, "r", encoding="utf-8") as f:
                                return json.load(f)
                        
                        data = await asyncio.to_thread(load_pg_hash)
                    
                    # JSON Schema 验证
                    valid, error = validate_json_schema(data, PARAGRAPH_HASH_SCHEMA, str(validated_path))
                    if valid:
                        stored_paragraph_hashes_b = set(data.get("stored_paragraph_hashes", []))
                        logger.debug(f"[KGCache] 加载段落hash: {len(stored_paragraph_hashes_b)} 个段落")
                    else:
                        logger.warning(f"[KGCache] JSON Schema 验证失败: {error}")
                        stored_paragraph_hashes_b = set()
            except FileNotFoundError:
                logger.warning(f"[KGCache] 段落 hash 文件不存在: {pg_hash_file_path}")
                stored_paragraph_hashes_b = set()
            except json.JSONDecodeError as e:
                logger.warning(f"[KGCache] JSON 解析失败: {e}")
                stored_paragraph_hashes_b = set()
            except OSError as e:
                logger.warning(f"[KGCache] 文件读取失败: {e}")
                stored_paragraph_hashes_b = set()
            
            if self._stopped:
                logger.info("[KGCache] 加载被中止")
                return
            
            # 模拟分批加载的延迟（避免 CPU 峰值）
            total_items = len(nodes_b) + len(edges_b) + len(ent_appear_cnt_b)
            batches = max(1, (total_items + self.batch_size - 1) // self.batch_size)
            for i in range(min(batches, 100)):  # 最多延迟 100 批
                await asyncio.sleep(self.batch_delay)
                if self._stopped:
                    return
            
            # 加载完成，原子切换
            with self.buffer_lock:
                self.buffer_b = True
                self.graph_b = graph_b
                self.nodes_b = nodes_b
                self.edges_b = edges_b
                self.ent_appear_cnt_b = ent_appear_cnt_b
                self.stored_paragraph_hashes_b = stored_paragraph_hashes_b
                
                # 原子切换：buffer_b → buffer_a
                self.buffer_a, self.buffer_b = self.buffer_b, None
                self.graph_a, self.graph_b = self.graph_b, None
                self.nodes_a, self.nodes_b = self.nodes_b, None
                self.edges_a, self.edges_b = self.edges_b, None
                self.ent_appear_cnt_a, self.ent_appear_cnt_b = self.ent_appear_cnt_b, None
                self.stored_paragraph_hashes_a, self.stored_paragraph_hashes_b = self.stored_paragraph_hashes_b, None
            
            # 更新文件变更追踪器
            total_count = len(nodes_b) + len(edges_b) + len(ent_appear_cnt_b)
            self._update_tracker(graph_data_path, ent_cnt_data_path, pg_hash_file_path, total_count)
            
            # 更新 ExpirationManager 的文件统计信息
            graph_stats = self._get_file_stats(graph_data_path)
            ent_cnt_stats = self._get_file_stats(ent_cnt_data_path)
            pg_hash_stats = self._get_file_stats(pg_hash_file_path)
            self._expiration_manager.update_file_stats(graph_data_path, graph_stats["mtime"], graph_stats["size"])
            self._expiration_manager.update_file_stats(ent_cnt_data_path, ent_cnt_stats["mtime"], ent_cnt_stats["size"])
            self._expiration_manager.update_file_stats(pg_hash_file_path, pg_hash_stats["mtime"], pg_hash_stats["size"])
            
            # 记录全量重建到 ExpirationManager
            self._expiration_manager.record_full_rebuild(total_count)
            self._sync_tracker_from_manager()

            self.last_refresh = time.time()
            load_time = time.time() - t0
            logger.info(
                f"[KGCache] 缓存加载完成并切换: "
                f"节点 {len(nodes_b)} 个, 边 {len(edges_b)} 条, "
                f"耗时 {load_time:.2f}s"
            )

            # 若 KG 已在插件加载前初始化完成（启动时序问题），这里做一次补偿注入
            try:
                self.try_inject_into_runtime()
            except Exception:
                pass
            
        except Exception as e:
            logger.error(f"[KGCache] 缓存加载失败: {e}")
        finally:
            async with self.load_lock:
                self.loading = False
    
    async def _refresh_loop(self) -> None:
        """定期刷新循环（带 ExpirationManager 决策和文件变更检测）"""
        while not self._stopped:
            await asyncio.sleep(self.refresh_interval)
            if self._stopped:
                break

            # 获取文件路径
            try:
                from src.chat.knowledge.kg_manager import KGManager
                kg_manager = KGManager()
                graph_path = kg_manager.graph_data_path
                ent_cnt_path = kg_manager.ent_cnt_data_path
                pg_hash_path = kg_manager.pg_hash_file_path
            except Exception as e:
                logger.warning(f"[KGCache] 获取文件路径失败: {e}")
                continue

            # 获取刷新决策
            decision = self._get_refresh_decision()
            
            if decision == RefreshDecision.SKIP:
                # 跳过刷新
                self._tracker["skip_count"] += 1
                logger.debug(f"[KGCache] ExpirationManager 决策跳过刷新（累计跳过 {self._tracker['skip_count']} 次）")
                continue
            
            if decision == RefreshDecision.FULL_REBUILD:
                logger.info("[KGCache] ExpirationManager 决策触发全量重建...")
                await self._load_to_buffer_b()
                continue
            
            # INCREMENTAL 决策：检测文件变更
            if self._should_reload(graph_path, ent_cnt_path, pg_hash_path):
                logger.info("[KGCache] 检测到文件变更，触发刷新...")
                await self._load_to_buffer_b()
            else:
                self._tracker["skip_count"] += 1
                logger.debug(f"[KGCache] 无文件变更，跳过刷新（累计跳过 {self._tracker['skip_count']} 次）")
    
    def start(self) -> None:
        """启动缓存模块"""
        self._stopped = False
        
        if self._degraded_mode:
            logger.info("[KGCache] 降级模式：跳过启动")
            return
        
        try:
            loop = asyncio.get_running_loop()
            if self._refresh_task is None or self._refresh_task.done():
                self._refresh_task = loop.create_task(self._refresh_loop())
                logger.info("[KGCache] 定期刷新任务已启动")
            
            # 如果缓存未加载且当前不在加载中，立即触发加载
            if (not self.is_loaded()) and (not self.loading):
                loop.create_task(self._load_to_buffer_b())
        except RuntimeError:
            logger.warning("[KGCache] 无法启动定期刷新：没有运行中的事件循环")
    
    def stop(self) -> None:
        """停止缓存模块"""
        self._stopped = True
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            self._refresh_task = None
        logger.info("[KGCache] 缓存模块已停止")
    
    def refresh(self) -> None:
        """手动刷新缓存（非阻塞）"""
        if self._degraded_mode:
            logger.warning("[KGCache] 降级模式：无法刷新")
            return
        
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._load_to_buffer_b())
            logger.info("[KGCache] 已触发手动刷新")
        except RuntimeError:
            logger.warning("[KGCache] 无法触发刷新：没有运行中的事件循环")
    
    def clear(self) -> None:
        """清空缓存"""
        with self.buffer_lock:
            self.buffer_a = None
            self.buffer_b = None
            self.graph_a = None
            self.nodes_a = None
            self.edges_a = None
            self.ent_appear_cnt_a = None
            self.stored_paragraph_hashes_a = None
            self.graph_b = None
            self.nodes_b = None
            self.edges_b = None
            self.ent_appear_cnt_b = None
            self.stored_paragraph_hashes_b = None
        logger.info("[KGCache] 缓存已清空")
    
    def size(self) -> Dict[str, int]:
        """获取缓存大小
        
        Returns:
            包含各项大小的字典
        """
        with self.buffer_lock:
            if self.nodes_a is None:
                return {"nodes": 0, "edges": 0, "entities": 0, "paragraphs": 0}
            return {
                "nodes": len(self.nodes_a) if self.nodes_a else 0,
                "edges": len(self.edges_a) if self.edges_a else 0,
                "entities": len(self.ent_appear_cnt_a) if self.ent_appear_cnt_a else 0,
                "paragraphs": len(self.stored_paragraph_hashes_a) if self.stored_paragraph_hashes_a else 0,
            }
    
    def get_memory_usage(self) -> int:
        """获取缓存内存使用量
        
        Returns:
            内存使用量（字节）
        """
        with self.buffer_lock:
            if self.buffer_a is None:
                return 0
            size = 0
            if self.graph_a is not None:
                size += MemoryUtils.get_size(self.graph_a)
            if self.nodes_a is not None:
                size += MemoryUtils.get_size(self.nodes_a)
            if self.edges_a is not None:
                size += MemoryUtils.get_size(self.edges_a)
            if self.ent_appear_cnt_a is not None:
                size += MemoryUtils.get_size(self.ent_appear_cnt_a)
            if self.stored_paragraph_hashes_a is not None:
                size += MemoryUtils.get_size(self.stored_paragraph_hashes_a)
            return size
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            统计信息字典
        """
        size_info = self.size()
        return {
            "degraded_mode": self._degraded_mode,
            "loaded": self.is_loaded(),
            "memory_bytes": self.get_memory_usage(),
            "last_refresh": self.last_refresh,
            "loading": self.loading,
            # 文件变更检测相关统计
            "skip_count": self._tracker["skip_count"],
            "last_full_rebuild": self._tracker["last_full_rebuild"],
            "time_since_rebuild": time.time() - self._tracker["last_full_rebuild"] if self._tracker["last_full_rebuild"] > 0 else 0,
            "tracked_total_count": self._tracker["total_count"],
            **size_info,
            **self.stats.total(),
        }
    
    @classmethod
    def check_dependencies(cls) -> Dict[str, Any]:
        """检查依赖可用性（用于日志/诊断）。

        Returns:
            依赖状态字典
        """

        dep = _dependency_status()
        return {
            "required": dep["required"],
            "optional": dep["optional"],
            "missing_required": dep["missing_required"],
            "missing_optional": dep["missing_optional"],
            "cache_enabled": cls.CACHE_ENABLED,
            "parquet_reader_available": cls.PARQUET_READER_AVAILABLE,
        }


_KG_CACHE_SINGLETON: Optional[KGCacheModule] = None
_KG_CACHE_SINGLETON_LOCK = threading.Lock()


def apply_kg_cache(cache_manager) -> Optional[KGCacheModule]:
    """应用知识库图谱缓存补丁（幂等）。

    额外处理：
    - 修复启动时序导致的 0 命中：若 KG 已在插件加载前完成初始化，缓存加载完成后会尝试
      `try_inject_into_runtime()` 将缓存注入到 `qa_manager.kg_manager`。
    """

    global _KG_CACHE_SINGLETON

    try:
        with _KG_CACHE_SINGLETON_LOCK:
            if _KG_CACHE_SINGLETON is None:
                _KG_CACHE_SINGLETON = KGCacheModule()
            cache = _KG_CACHE_SINGLETON

        # 注册到缓存管理器（重复注册不影响：直接覆盖同名 key）
        try:
            cache_manager.register_cache("kg_cache", cache)
        except Exception:
            pass

        # 检查依赖并输出更清晰的影响说明
        deps = cache.check_dependencies()
        logger.info(f"[KGCache] 依赖状态: {deps}")

        if cache.is_degraded():
            logger.warning(
                "[KGCache] KGCache 已禁用：仅影响性能（不再预加载/命中缓存），不影响 KGManager 原始读盘功能。"
            )
            return cache

        if not cache.PARQUET_READER_AVAILABLE:
            logger.warning(
                "[KGCache] parquet 读取不可用：实体计数将从图结构重建，首次加载更慢。"
            )

        # 尝试 patch 原始函数（仅在缓存启用时；带幂等标记）
        try:
            from src.chat.knowledge.kg_manager import KGManager

            current = getattr(KGManager, "load_from_file", None)
            if callable(current) and getattr(current, "__cm_perf_opt_kgcache_patched__", False):
                logger.debug("[KGCache] KGManager.load_from_file 已被本模块 patch，跳过")
            else:
                original_load = KGManager.load_from_file

                def patched_load(self):
                    """补丁后的加载图谱函数。

                    行为：
                    - 若缓存已加载：直接注入 graph / ent_appear_cnt / stored_paragraph_hashes
                    - 若缓存未就绪：调用原始 KGManager.load_from_file（保持原有异常语义）
                    """

                    cached = cache.get_cached_data()
                    if cached and cached.get("graph") is not None:
                        cache.stats.hit()
                        self.graph = cached["graph"]
                        self.ent_appear_cnt = cached.get("ent_appear_cnt") or {}
                        self.stored_paragraph_hashes = cached.get("stored_paragraph_hashes") or set()
                        return None

                    t0 = time.time()
                    result = original_load(self)
                    cache.stats.miss(time.time() - t0)
                    return result

                setattr(patched_load, "__cm_perf_opt_kgcache_patched__", True)
                setattr(patched_load, "__cm_perf_opt_kgcache_original__", original_load)

                KGManager.load_from_file = patched_load
                logger.info("[KGCache] 已 patch KGManager.load_from_file")

        except ImportError:
            logger.debug("[KGCache] KGManager 不可用，跳过 patch")
        except Exception as e:
            logger.warning(f"[KGCache] patch 失败: {e}")

        # 若已加载，则尝试立刻注入；否则等待加载完成后由 loader 再尝试
        try:
            cache.try_inject_into_runtime()
        except Exception:
            pass

        return cache

    except Exception as e:
        logger.error(f"[KGCache] 初始化失败: {e}")
        return None
