"""SQLite 数据库调优模块（db_tuning）。

该模块合并两类优化：
1) SQLite 运行时 PRAGMA 参数调优（连接后追加设置，不覆盖已存在的 pragmas）
2) 关键索引自检与按需创建（CREATE INDEX IF NOT EXISTS，后台执行避免阻塞）

设计要点：
- 稳定性优先：每个 PRAGMA / 索引单独 try/except，失败不影响其他
- 幂等：索引使用 IF NOT EXISTS
- 低侵入：PRAGMA 无法回退；remove_patch() 仅停止后台任务并恢复 connect monkey-patch
- 连接级别：通过 patch db.connect()，确保未来连接也会应用额外 PRAGMA

模块风格参考：[`levenshtein_fast.py`](CM-performance-optimizer-plugin/components/modules/levenshtein_fast.py:1)
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from src.common.logger import get_logger
except ImportError:  # pragma: no cover
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger("CM_perf_opt")


def _load_core_module():
    """动态加载 core 模块，避免相对导入问题。"""

    module_name = "CM_perf_opt_core"
    if module_name in sys.modules:
        return sys.modules[module_name]

    current_dir = Path(__file__).parent
    plugin_dir = current_dir.parent.parent  # components/modules -> components -> plugin root
    core_init = plugin_dir / "core" / "__init__.py"

    if not core_init.exists():
        raise ImportError(f"Core module not found at {core_init}")

    spec = importlib.util.spec_from_file_location(module_name, core_init)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load core module from {core_init}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    # 预加载 core 子模块，避免 __init__ 中的相对导入失败
    for submodule in ["cache", "utils", "config", "monitor", "module_config"]:
        sub_path = plugin_dir / "core" / f"{submodule}.py"
        if not sub_path.exists():
            continue
        sub_name = f"CM_perf_opt_core_{submodule}"
        if sub_name in sys.modules:
            continue
        sub_spec = importlib.util.spec_from_file_location(sub_name, sub_path)
        if sub_spec is None or sub_spec.loader is None:
            continue
        sub_mod = importlib.util.module_from_spec(sub_spec)
        sys.modules[sub_name] = sub_mod
        sub_spec.loader.exec_module(sub_mod)

    spec.loader.exec_module(module)
    return module


# ========== 核心模块加载（ModuleStats） ==========
try:
    from ..core import ModuleStats
except Exception:
    try:
        _core = _load_core_module()
        ModuleStats = _core.ModuleStats
    except Exception as e:  # pragma: no cover
        logger.warning(f"[DbTuning] 无法加载核心模块，使用内置实现: {e}")

        class ModuleStats:  # type: ignore[no-redef]
            """Fallback ModuleStats（仅提供 total/reset_interval/hit/miss）。"""

            def __init__(self, name: str):
                self.name = str(name)
                self._lock = threading.Lock()
                self.t_hit = 0
                self.t_miss = 0
                self.i_hit = 0
                self.i_miss = 0

            def hit(self) -> None:
                with self._lock:
                    self.t_hit += 1
                    self.i_hit += 1

            def miss(self, elapsed: float = 0.0) -> None:
                _ = elapsed
                with self._lock:
                    self.t_miss += 1
                    self.i_miss += 1

            def total(self) -> Dict[str, Any]:
                with self._lock:
                    return {"t_hit": self.t_hit, "t_miss": self.t_miss}

            def reset_interval(self) -> Dict[str, Any]:
                with self._lock:
                    r = {"i_hit": self.i_hit, "i_miss": self.i_miss}
                    self.i_hit = 0
                    self.i_miss = 0
                    return r


def _quote_sqlite_pragma_value(value: Any) -> str:
    """将 PRAGMA 值转换为 SQL 片段。

    SQLite PRAGMA 对字符串值可接受不带引号的关键字（如 MEMORY），也接受带引号的文本。
    这里对字符串统一加引号，避免注入与转义问题。
    """

    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        # PRAGMA 通常期望整数；float 也允许，但这里保持原样
        return str(value)
    s = str(value)
    s = s.replace("'", "''")
    return f"'{s}'"


class DbTuningModule:
    """SQLite 调优补丁模块。"""

    def __init__(
        self,
        mmap_size: int = 268_435_456,
        checkpoint_interval: int = 300,
    ):
        """初始化配置。

        Args:
            mmap_size: `PRAGMA mmap_size`，默认 256MB。设置为 0 表示禁用。
            checkpoint_interval: WAL checkpoint 周期（秒），默认 300 秒。设置为 0 表示禁用。
        """

        self.stats = ModuleStats("db_tuning")

        self.mmap_size = int(mmap_size)
        self.checkpoint_interval = int(checkpoint_interval)

        self._lock = threading.RLock()
        self._patched = False
        self._removed = False

        self._db: Optional[Any] = None
        self._orig_connect: Optional[Any] = None

        # 统计与状态
        self._additional_pragmas: Dict[str, Any] = {}
        self._applied_pragmas: List[str] = []
        self._skipped_pragmas: List[str] = []
        self._failed_pragmas: List[str] = []

        self._created_indexes: List[str] = []
        self._skipped_indexes: List[str] = []
        self._failed_indexes: List[str] = []

        # 后台任务
        self._running = False
        self._index_thread: Optional[threading.Thread] = None
        self._checkpoint_thread: Optional[threading.Thread] = None
        self._checkpoint_count = 0

    def _load_db(self) -> Any:
        """加载 Peewee 数据库对象（SqliteDatabase）。"""

        if self._db is not None:
            return self._db

        from src.common.database.database import db  # 延迟导入，避免 import 时序问题

        self._db = db
        return db

    @staticmethod
    def _get_existing_pragma_keys(db: Any) -> set[str]:
        """尽量从 Peewee Database 对象中获取已配置的 pragmas key 集合。"""

        keys: set[str] = set()
        for attr in ("pragmas", "_pragmas"):
            try:
                v = getattr(db, attr, None)
                if isinstance(v, dict):
                    keys.update(str(k) for k in v.keys())
            except Exception:
                continue
        return keys

    def _build_additional_pragmas(self, db: Any) -> Dict[str, Any]:
        """构建额外 PRAGMA（不覆盖已有配置）。"""

        additional_pragmas: Dict[str, Any] = {
            "temp_store": "MEMORY",
            "mmap_size": int(self.mmap_size),
            "wal_autocheckpoint": 1000,
            "journal_size_limit": 67_108_864,
        }

        existing = self._get_existing_pragma_keys(db)
        out: Dict[str, Any] = {}
        skipped: List[str] = []

        for k, v in additional_pragmas.items():
            if k in existing:
                skipped.append(k)
                continue
            out[k] = v

        with self._lock:
            self._additional_pragmas = dict(out)
            self._skipped_pragmas = skipped

        return out

    def _apply_pragmas(self) -> None:
        """对当前连接应用额外 PRAGMA（逐项容错）。"""

        db = self._load_db()
        additional_pragmas = self._build_additional_pragmas(db)

        applied: List[str] = []
        failed: List[str] = []

        for pragma, value in additional_pragmas.items():
            try:
                value_sql = _quote_sqlite_pragma_value(value)
                db.execute_sql(f"PRAGMA {pragma} = {value_sql}")
                applied.append(str(pragma))
                try:
                    if hasattr(self.stats, "hit"):
                        self.stats.hit()
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"[DbTuning] PRAGMA {pragma} 设置失败: {e}")
                failed.append(str(pragma))
                try:
                    if hasattr(self.stats, "miss"):
                        try:
                            self.stats.miss(0.0)
                        except TypeError:
                            self.stats.miss()
                except Exception:
                    pass

        with self._lock:
            # 保留历史（同一 pragma 多次 apply 只记录一次）
            for p in applied:
                if p not in self._applied_pragmas:
                    self._applied_pragmas.append(p)
            for p in failed:
                if p not in self._failed_pragmas:
                    self._failed_pragmas.append(p)

    def _patch_db_connect(self) -> None:
        """patch db.connect()，确保未来连接也执行额外 PRAGMA。"""

        db = self._load_db()

        if self._orig_connect is not None:
            return

        self._orig_connect = getattr(db, "connect", None)
        if not callable(self._orig_connect):
            self._orig_connect = None
            return

        module = self
        original_connect = self._orig_connect

        def connect_wrapper(*args: Any, **kwargs: Any):
            conn = original_connect(*args, **kwargs)
            try:
                # connect() 之后连接应为打开状态，直接设置 PRAGMA
                module._apply_pragmas()
            except Exception as e:
                logger.debug(f"[DbTuning] connect 后 PRAGMA 应用失败: {e}")
            return conn

        try:
            db.connect = connect_wrapper  # type: ignore[assignment]
        except Exception as e:
            logger.debug(f"[DbTuning] patch db.connect 失败: {e}")
            self._orig_connect = None

    def _restore_db_connect(self) -> None:
        """恢复 db.connect() 原始实现。"""

        db = self._load_db()
        if self._orig_connect is None:
            return

        try:
            db.connect = self._orig_connect  # type: ignore[assignment]
        except Exception:
            pass
        finally:
            self._orig_connect = None

    @staticmethod
    def _candidate_indexes() -> List[Tuple[str, str, str]]:
        """候选索引列表（表名需与 Peewee Meta.table_name 一致）。"""

        # 注意：src/common/database/database_model.py 中明确指定了 table_name。
        return [
            # Messages：高频过滤与排序
            ("idx_messages_chat_time", "messages", "chat_id, time"),
            ("idx_messages_user_id", "messages", "user_id"),
            ("idx_messages_user_platform_user", "messages", "user_platform, user_id"),
            ("idx_messages_group_time", "messages", "chat_info_group_id, time"),
            # Images：按 description 查找
            ("idx_images_description", "images", "description"),
            # ThinkingBack：按 chat_id + update_time 查询（与主代码一致）
            ("idx_thinking_back_chat_update", "thinking_back", "chat_id, update_time"),
            ("idx_thinking_back_found_update", "thinking_back", "found_answer, update_time"),
            # PersonInfo：按 person_name 查找
            ("idx_person_info_name", "person_info", "person_name"),
        ]

    def _create_indexes_worker(self) -> None:
        """后台线程：创建索引（幂等）。"""

        db = self._load_db()
        created: List[str] = []
        skipped: List[str] = []
        failed: List[str] = []

        try:
            with db.connection_context():
                for idx_name, table, columns in self._candidate_indexes():
                    try:
                        if hasattr(db, "table_exists"):
                            try:
                                if not db.table_exists(table):
                                    skipped.append(idx_name)
                                    continue
                            except Exception:
                                # table_exists 失败时仍尝试创建（由 SQLite 报错）
                                pass

                        db.execute_sql(
                            f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table} ({columns})"
                        )
                        created.append(idx_name)
                        try:
                            if hasattr(self.stats, "hit"):
                                self.stats.hit()
                        except Exception:
                            pass
                    except Exception as e:
                        logger.warning(f"[DbTuning] 索引 {idx_name} 创建失败: {e}")
                        failed.append(idx_name)
                        try:
                            if hasattr(self.stats, "miss"):
                                try:
                                    self.stats.miss(0.0)
                                except TypeError:
                                    self.stats.miss()
                        except Exception:
                            pass
        except Exception as e:
            logger.warning(f"[DbTuning] 索引创建线程异常退出: {e}")

        with self._lock:
            for x in created:
                if x not in self._created_indexes:
                    self._created_indexes.append(x)
            for x in skipped:
                if x not in self._skipped_indexes:
                    self._skipped_indexes.append(x)
            for x in failed:
                if x not in self._failed_indexes:
                    self._failed_indexes.append(x)

    def _start_index_thread_if_needed(self) -> None:
        if self._index_thread is not None:
            return
        t = threading.Thread(
            target=self._create_indexes_worker,
            name="DbTuningIndexWorker",
            daemon=True,
        )
        self._index_thread = t
        t.start()

    async def _wal_checkpoint_loop(self) -> None:
        """周期性执行 WAL checkpoint，避免 WAL 无限增长。"""

        db = self._load_db()

        while self._running:
            await asyncio.sleep(max(1, int(self.checkpoint_interval)))
            if not self._running:
                break
            try:
                with db.connection_context():
                    db.execute_sql("PRAGMA wal_checkpoint(PASSIVE)")
                with self._lock:
                    self._checkpoint_count += 1
                try:
                    if hasattr(self.stats, "hit"):
                        self.stats.hit()
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"[DbTuning] WAL checkpoint 失败: {e}")
                try:
                    if hasattr(self.stats, "miss"):
                        try:
                            self.stats.miss(0.0)
                        except TypeError:
                            self.stats.miss()
                except Exception:
                    pass

    def _checkpoint_thread_worker(self) -> None:
        try:
            asyncio.run(self._wal_checkpoint_loop())
        except Exception as e:
            logger.debug(f"[DbTuning] checkpoint 线程退出: {e}")

    def _start_checkpoint_thread_if_needed(self) -> None:
        if int(self.checkpoint_interval) <= 0:
            return
        if self._checkpoint_thread is not None:
            return

        t = threading.Thread(
            target=self._checkpoint_thread_worker,
            name="DbTuningWalCheckpoint",
            daemon=True,
        )
        self._checkpoint_thread = t
        t.start()

    def apply_patch(self) -> None:
        """执行 PRAGMA 设置 + 创建索引（索引后台执行）。"""

        with self._lock:
            if self._removed:
                # removed 后允许重新启用
                self._removed = False

            if not self._patched:
                self._running = True
                self._patch_db_connect()

        # PRAGMA 设置：允许多次调用（用于外部修改 mmap_size 后再次应用）
        try:
            db = self._load_db()
            with db.connection_context():
                self._apply_pragmas()
        except Exception as e:
            logger.debug(f"[DbTuning] PRAGMA 应用阶段异常: {e}")

        # 索引创建：后台线程
        try:
            self._start_index_thread_if_needed()
        except Exception as e:
            logger.debug(f"[DbTuning] 启动索引线程失败: {e}")

        # checkpoint：可选后台任务
        try:
            self._start_checkpoint_thread_if_needed()
        except Exception as e:
            logger.debug(f"[DbTuning] 启动 checkpoint 线程失败: {e}")

        with self._lock:
            self._patched = True

        logger.info("[DbTuning] ✓ db_tuning 已应用")

    def remove_patch(self) -> None:
        """标记为已移除。

        - PRAGMA 无法可靠回退（且无需回退）
        - 索引无需删除
        - 停止后台 checkpoint 任务、恢复 connect monkey-patch
        """

        with self._lock:
            if self._removed:
                return
            self._removed = True
            self._patched = False
            self._running = False

        # best-effort 停止后台线程（daemon 线程无需强制 join）
        try:
            if self._checkpoint_thread is not None:
                self._checkpoint_thread.join(timeout=1.0)
        except Exception:
            pass

        self._restore_db_connect()
        logger.info("[DbTuning] 补丁已移除（PRAGMA 不回退，后台任务已停止）")

    def get_stats(self) -> Dict[str, Any]:
        """返回统计信息（应用了哪些 PRAGMA、创建了哪些索引）。"""

        with self._lock:
            return {
                "patched": bool(self._patched),
                "removed": bool(self._removed),
                "mmap_size": int(self.mmap_size),
                "checkpoint_interval": int(self.checkpoint_interval),
                "pragmas_additional": dict(self._additional_pragmas),
                "pragmas_applied": list(self._applied_pragmas),
                "pragmas_skipped": list(self._skipped_pragmas),
                "pragmas_failed": list(self._failed_pragmas),
                "indexes_created": list(self._created_indexes),
                "indexes_skipped": list(self._skipped_indexes),
                "indexes_failed": list(self._failed_indexes),
                "checkpoint_count": int(self._checkpoint_count),
            }

    def get_memory_usage(self) -> int:
        """返回 0（该模块无缓存持久数据）。"""

        return 0


def apply_db_tuning(cache_manager) -> Optional[DbTuningModule]:
    """工厂函数：创建模块、注册到 cache_manager 并应用补丁。"""

    try:
        mod = DbTuningModule()
        cache_manager.register_cache("db_tuning", mod)
        mod.apply_patch()
        logger.info("[DbTuning] ✓ 模块已初始化")
        return mod
    except Exception as e:
        logger.error(f"[DbTuning] 初始化失败: {e}")
        return None
