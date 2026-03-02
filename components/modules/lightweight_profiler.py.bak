"""Lightweight Profiler 轻量性能观测模块。

该模块为 CM-performance-optimizer-plugin 提供“纯观测层”的轻量剖析能力：
- 函数级计时：装饰器 [`profile_func()`](CM-performance-optimizer-plugin/components/modules/lightweight_profiler.py:362) / 上下文管理器 [`ProfileContext`](CM-performance-optimizer-plugin/components/modules/lightweight_profiler.py:402)
- DB 级计时：monkey-patch Peewee 的 `db.execute_sql`
- 采样模式：通过 sample_rate 降低开销（并非每次调用都计时）
- 周期性报告：后台任务定期输出摘要到日志

注意：该模块只观测，不修改任何业务逻辑。

模块风格参考：[`levenshtein_fast.py`](CM-performance-optimizer-plugin/components/modules/levenshtein_fast.py:1)
"""

from __future__ import annotations

import asyncio
import functools
import random
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:
    from src.common.logger import get_logger
except ImportError:  # pragma: no cover
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


# ========== 核心模块加载（ModuleStats / MemoryUtils） ==========
try:
    from ..core import MemoryUtils, ModuleStats
except Exception:
    try:
        _core = load_core_module(Path(__file__).parent)
        ModuleStats = _core.ModuleStats
        MemoryUtils = _core.MemoryUtils
    except (ImportError, CoreModuleLoadError) as e:  # pragma: no cover
        logger.warning(f"[LightweightProfiler] 无法加载核心模块，使用内置实现: {e}")

        class ModuleStats:  # type: ignore[no-redef]
            """Fallback ModuleStats（保持接口存在即可）。"""

            def __init__(self, name: str):
                self.name = str(name)

            def total(self) -> Dict[str, Any]:
                return {}

            def reset_interval(self) -> Dict[str, Any]:
                return {}

        class MemoryUtils:  # type: ignore[no-redef]
            """Fallback MemoryUtils（满足接口即可）。"""

            @staticmethod
            def get_size(obj: Any, seen: Optional[set[int]] = None) -> int:
                if seen is None:
                    seen = set()
                obj_id = id(obj)
                if obj_id in seen:
                    return 0
                seen.add(obj_id)
                try:
                    size = sys.getsizeof(obj)
                except Exception:
                    size = 0
                if isinstance(obj, dict):
                    size += sum(MemoryUtils.get_size(k, seen) for k in obj.keys())
                    size += sum(MemoryUtils.get_size(v, seen) for v in obj.values())
                elif isinstance(obj, (list, tuple, set, frozenset)):
                    size += sum(MemoryUtils.get_size(i, seen) for i in obj)
                return int(size)


@dataclass
class _AggTiming:
    """聚合计时统计（仅聚合，避免高内存开销）。"""

    calls: int = 0
    total_time: float = 0.0
    max_time: float = 0.0

    def record(self, elapsed: float) -> None:
        self.calls += 1
        self.total_time += float(elapsed)
        if elapsed > self.max_time:
            self.max_time = float(elapsed)


class _StatsAdapter:
    """StatsReporter 适配器：提供 total()/reset_interval() 两个方法。"""

    def __init__(self, module: "LightweightProfilerModule"):
        self._module = module

    def total(self) -> Dict[str, Any]:
        return self._module.get_stats()

    def reset_interval(self) -> Dict[str, Any]:
        return self._module._reset_interval()


_ACTIVE_PROFILER: Optional["LightweightProfilerModule"] = None
_ACTIVE_LOCK = threading.Lock()


def _get_active_profiler() -> Optional["LightweightProfilerModule"]:
    with _ACTIVE_LOCK:
        return _ACTIVE_PROFILER


def _set_active_profiler(mod: Optional["LightweightProfilerModule"]) -> None:
    global _ACTIVE_PROFILER
    with _ACTIVE_LOCK:
        _ACTIVE_PROFILER = mod


def profile_func(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """函数装饰器：记录函数调用次数和耗时（采样）。

    该装饰器会尝试使用全局“活动 profiler 实例”（由
    [`apply_lightweight_profiler()`](CM-performance-optimizer-plugin/components/modules/lightweight_profiler.py:507)
    设置）。若 profiler 未启用，则几乎零开销。

    Args:
        name: 统计项名称（建议使用稳定的短名称）。

    Returns:
        装饰器函数。
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            mod = _get_active_profiler()
            if mod is None or not mod.should_sample():
                return func(*args, **kwargs)

            t0 = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - t0
                mod.record_function_timing(name, elapsed)

        return wrapper

    return decorator


class ProfileContext:
    """上下文管理器：记录代码块耗时（采样）。"""

    def __init__(self, name: str):
        self._name = str(name)
        self._mod: Optional[LightweightProfilerModule] = None
        self._t0: float = 0.0
        self._enabled = False

    def __enter__(self) -> "ProfileContext":
        mod = _get_active_profiler()
        if mod is not None and mod.should_sample():
            self._mod = mod
            self._enabled = True
            self._t0 = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[Any],
    ) -> None:
        _ = exc_type, exc, tb
        if not self._enabled or self._mod is None:
            return
        elapsed = time.perf_counter() - self._t0
        self._mod.record_function_timing(self._name, elapsed)


class LightweightProfilerModule:
    """轻量性能剖析模块。

    - DB 计时：patch Peewee execute_sql
    - 函数计时：通过 [`profile_func()`](CM-performance-optimizer-plugin/components/modules/lightweight_profiler.py:362)
      和 [`ProfileContext`](CM-performance-optimizer-plugin/components/modules/lightweight_profiler.py:402)
      写入到本模块统计
    """

    def __init__(self, sample_rate: float = 0.1, report_interval: float = 300.0):
        """初始化。

        Args:
            sample_rate: 采样率（0.0-1.0），默认 10%。
            report_interval: 统计报告间隔（秒），默认 5 分钟。
        """

        self.sample_rate = float(sample_rate)
        self.report_interval = float(report_interval)

        self._start_time = time.monotonic()

        self._patch_lock = threading.RLock()
        self._stats_lock = threading.Lock()

        self._patched = False
        self._db: Optional[Any] = None
        self._original_execute_sql: Optional[Callable[..., Any]] = None

        self._task: Optional[asyncio.Task[None]] = None
        self._running = False

        # StatsReporter 适配：保持插件现有 get_stats() 汇总体系兼容
        self.stats = _StatsAdapter(self)

        # DB 统计（total + interval）
        self._db_total_queries = 0
        self._db_total_time = 0.0
        self._db_by_type: Dict[str, _AggTiming] = {}

        self._db_i_queries = 0
        self._db_i_time = 0.0

        # 函数统计（total + interval）
        self._func_total: Dict[str, _AggTiming] = {}
        self._func_i_calls = 0

    def should_sample(self) -> bool:
        """采样决策（低开销）。"""

        sr = self.sample_rate
        if sr <= 0.0:
            return False
        if sr >= 1.0:
            return True
        return random.random() <= sr

    def apply_patch(self) -> None:
        """包装 Peewee 的 `db.execute_sql` 进行 DB 查询计时。"""

        with self._patch_lock:
            if self._patched:
                return

            try:
                from src.common.database.database import db

                self._db = db
            except Exception as e:
                logger.warning(f"[LightweightProfiler] 无法导入 db，跳过 DB patch: {e}")
                self._db = None
                self._original_execute_sql = None
                self._patched = True  # 标记为已处理，避免重复警告
                return

            if not hasattr(self._db, "execute_sql"):
                logger.warning("[LightweightProfiler] db.execute_sql 不存在，跳过 DB patch")
                self._original_execute_sql = None
                self._patched = True
                return

            original_execute_sql = getattr(self._db, "execute_sql")
            if not callable(original_execute_sql):
                logger.warning("[LightweightProfiler] db.execute_sql 不可调用，跳过 DB patch")
                self._original_execute_sql = None
                self._patched = True
                return

            self._original_execute_sql = original_execute_sql
            module = self

            @functools.wraps(original_execute_sql)
            def profiled_execute_sql(sql: Any, params: Any = None, commit: Any = None):
                # 采样决策
                if not module.should_sample():
                    return original_execute_sql(sql, params, commit)

                t0 = time.perf_counter()
                ok = False
                try:
                    result = original_execute_sql(sql, params, commit)
                    ok = True
                    return result
                finally:
                    elapsed = time.perf_counter() - t0
                    module._record_db_timing(sql=sql, elapsed=elapsed, ok=ok)

            try:
                setattr(self._db, "execute_sql", profiled_execute_sql)
                self._patched = True
                logger.info(
                    f"[LightweightProfiler] ✓ DB patch 已应用（sample_rate={self.sample_rate:.3f}）"
                )
            except Exception as e:
                logger.error(f"[LightweightProfiler] ✗ DB patch 应用失败: {e}")

    def remove_patch(self) -> None:
        """恢复原始 execute_sql。"""

        with self._patch_lock:
            if not self._patched:
                return

            if self._db is not None and self._original_execute_sql is not None:
                try:
                    setattr(self._db, "execute_sql", self._original_execute_sql)
                    logger.info("[LightweightProfiler] DB patch 已移除")
                except Exception as e:
                    logger.error(f"[LightweightProfiler] DB patch 移除失败: {e}")

            self._db = None
            self._original_execute_sql = None
            self._patched = False

    def record_function_timing(self, name: str, elapsed: float) -> None:
        """记录函数/代码块耗时（线程安全）。"""

        key = str(name)
        with self._stats_lock:
            agg = self._func_total.get(key)
            if agg is None:
                agg = _AggTiming()
                self._func_total[key] = agg
            agg.record(elapsed)
            self._func_i_calls += 1

    def _record_db_timing(self, sql: Any, elapsed: float, ok: bool) -> None:
        """记录 DB 查询耗时（线程安全）。"""

        if isinstance(sql, str):
            sql_str = sql
        else:
            try:
                sql_str = str(sql)
            except Exception:
                sql_str = ""

        sql_type = "UNKNOWN"
        try:
            if sql_str:
                sql_type = sql_str.strip().split()[0].upper()
        except Exception:
            sql_type = "UNKNOWN"

        _ = ok  # 当前实现不额外区分错误；保留参数用于未来扩展

        with self._stats_lock:
            self._db_total_queries += 1
            self._db_total_time += float(elapsed)

            self._db_i_queries += 1
            self._db_i_time += float(elapsed)

            agg = self._db_by_type.get(sql_type)
            if agg is None:
                agg = _AggTiming()
                self._db_by_type[sql_type] = agg
            agg.record(elapsed)

    def _reset_interval(self) -> Dict[str, Any]:
        """返回并清零间隔统计（供 StatsReporter 使用）。"""

        with self._stats_lock:
            i_db_q = int(self._db_i_queries)
            i_db_ms = float(self._db_i_time) * 1000.0
            i_func_calls = int(self._func_i_calls)

            self._db_i_queries = 0
            self._db_i_time = 0.0
            self._func_i_calls = 0

        return {
            "i_db_sampled_queries": i_db_q,
            "i_db_total_time_ms": round(i_db_ms, 3),
            "i_functions_sampled_calls": i_func_calls,
        }

    def get_stats(self) -> Dict[str, Any]:
        """返回统计信息。"""

        uptime = max(0.0, time.monotonic() - self._start_time)

        with self._stats_lock:
            q = int(self._db_total_queries)
            t = float(self._db_total_time)
            by_type = dict(self._db_by_type)
            func_total = dict(self._func_total)

        db_out: Dict[str, Any] = {
            "sampled_queries": q,
            "total_time_ms": round(t * 1000.0, 3),
            "avg_time_ms": round((t * 1000.0 / q) if q > 0 else 0.0, 3),
            "by_type": {},
        }

        for k, agg in by_type.items():
            db_out["by_type"][k] = {
                "count": int(agg.calls),
                "time_ms": round(float(agg.total_time) * 1000.0, 3),
            }

        func_out: Dict[str, Any] = {}
        for fname, agg in func_total.items():
            calls = int(agg.calls)
            avg_ms = (float(agg.total_time) * 1000.0 / calls) if calls > 0 else 0.0
            func_out[fname] = {
                "calls": calls,
                "avg_ms": round(avg_ms, 3),
                "max_ms": round(float(agg.max_time) * 1000.0, 3),
            }

        return {
            "db": db_out,
            "functions": func_out,
            "sample_rate": float(self.sample_rate),
            "uptime_seconds": int(uptime),
        }

    def get_memory_usage(self) -> int:
        """返回统计数据占用的内存（字节，估算）。"""

        try:
            with self._stats_lock:
                snapshot = {
                    "db_total_queries": self._db_total_queries,
                    "db_total_time": self._db_total_time,
                    "db_by_type": self._db_by_type,
                    "func_total": self._func_total,
                }
            return int(MemoryUtils.get_size(snapshot))
        except Exception:
            return 0

    def start(self) -> None:
        """启动后台报告任务。"""

        if self._running:
            return

        self._running = True

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("[LightweightProfiler] 无运行中的事件循环，跳过后台报告任务")
            self._running = False
            return

        if self._task is None or self._task.done():
            self._task = loop.create_task(self._report_loop())
            logger.info(
                f"[LightweightProfiler] 后台报告任务已启动（interval={self.report_interval:.1f}s）"
            )

    def stop(self) -> None:
        """停止后台报告任务，并移除 patch。"""

        self._running = False

        if self._task is not None:
            try:
                self._task.cancel()
            except Exception:
                pass
            self._task = None

        self.remove_patch()

        # 若本实例是活动 profiler，则清空
        if _get_active_profiler() is self:
            _set_active_profiler(None)

    async def _report_loop(self) -> None:
        """后台循环：定期输出简洁统计摘要。"""

        try:
            while self._running:
                await asyncio.sleep(max(1.0, float(self.report_interval)))
                if not self._running:
                    break
                try:
                    logger.info(self._format_brief_report())
                except Exception as e:
                    logger.error(f"[LightweightProfiler] 输出报告失败: {e}")
        except asyncio.CancelledError:
            return

    def _format_brief_report(self) -> str:
        """生成简洁报告字符串。"""

        stats = self.get_stats()
        db = stats.get("db", {})
        funcs = stats.get("functions", {})

        q = int(db.get("sampled_queries", 0) or 0)
        total_ms = float(db.get("total_time_ms", 0.0) or 0.0)
        avg_ms = float(db.get("avg_time_ms", 0.0) or 0.0)

        # 选出 total_time 最大的前 3 个函数
        top_funcs = []
        for fname, v in funcs.items():
            try:
                calls = int(v.get("calls", 0) or 0)
                avg = float(v.get("avg_ms", 0.0) or 0.0)
                total = float(calls) * avg
                top_funcs.append(
                    (
                        total,
                        fname,
                        calls,
                        avg,
                        float(v.get("max_ms", 0.0) or 0.0),
                    )
                )
            except Exception:
                continue
        top_funcs.sort(reverse=True)
        top = top_funcs[:3]

        func_part = ""
        if top:
            func_part = " top_funcs=" + ",".join(
                f"{name}:{calls}x(avg={avg:.3f}ms,max={mx:.3f}ms)"
                for _, name, calls, avg, mx in top
            )

        return (
            "[LightweightProfiler] "
            f"db={q}q(total={total_ms:.3f}ms,avg={avg_ms:.3f}ms) "
            f"sample_rate={self.sample_rate:.3f} uptime={stats.get('uptime_seconds', 0)}s"
            + func_part
        )


def apply_lightweight_profiler(cache_manager) -> Optional[LightweightProfilerModule]:
    """工厂函数：创建模块、注册到 cache_manager 并应用 patch。"""

    try:
        mod = LightweightProfilerModule()
        cache_manager.register_cache("lightweight_profiler", mod)

        # 设置为活动 profiler，使装饰器/上下文管理器可用
        _set_active_profiler(mod)

        mod.apply_patch()
        mod.start()

        logger.info("[LightweightProfiler] ✓ 模块已初始化")
        return mod
    except Exception as e:
        logger.error(f"[LightweightProfiler] 初始化失败: {e}")
        return None
