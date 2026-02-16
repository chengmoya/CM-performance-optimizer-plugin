"""
监控模块 - 内存监控、性能指标收集和统计报告

功能：
- 内存使用监控
- 性能指标收集
- 扩展的统计报告格式
- 告警机制
- 通知集成（QQ/控制台）
"""

from __future__ import annotations

import asyncio
import gc
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .notification import NotificationManager, NotificationConfig

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None  # type: ignore
    PSUTIL_AVAILABLE = False

try:
    from src.common.logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger("CM_perf_opt")

# ========== 常量定义 ==========
# 内存单位转换常量
BYTES_PER_KB = 1024
BYTES_PER_MB = 1024 * 1024

# Python 内存块估算：sys.getallocatedblocks() 返回已分配内存块数量
# 每块约 64 字节（CPython 默认对齐大小，实际可能因对象类型而异）
ALLOC_BLOCK_SIZE = 64


@dataclass
class MemorySnapshot:
    """内存快照"""

    timestamp: float
    process_rss: int  # 进程常驻内存（字节）
    process_vms: int  # 进程虚拟内存（字节）
    python_allocated: int  # Python 分配的内存
    gc_counts: Tuple[int, int, int]  # GC 统计
    cache_memory: Dict[str, int] = field(default_factory=dict)  # 各缓存模块内存

    @property
    def total_cache_memory(self) -> int:
        """总缓存内存使用"""
        return sum(self.cache_memory.values())

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "process_rss_mb": round(self.process_rss / BYTES_PER_MB, 2),
            "process_vms_mb": round(self.process_vms / BYTES_PER_MB, 2),
            "python_allocated_mb": round(self.python_allocated / BYTES_PER_MB, 2),
            "gc_counts": list(self.gc_counts),
            "cache_memory": {k: round(v / BYTES_PER_MB, 2) for k, v in self.cache_memory.items()},
            "total_cache_memory_mb": round(self.total_cache_memory / BYTES_PER_MB, 2),
        }


@dataclass
class PerformanceMetrics:
    """性能指标"""

    # 缓存命中率
    cache_hit_rates: Dict[str, float] = field(default_factory=dict)
    # 平均响应时间（毫秒）
    avg_response_times: Dict[str, float] = field(default_factory=dict)
    # 请求计数
    request_counts: Dict[str, int] = field(default_factory=dict)
    # 错误计数
    error_counts: Dict[str, int] = field(default_factory=dict)
    # 慢查询计数
    slow_query_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "cache_hit_rates": self.cache_hit_rates,
            "avg_response_times": self.avg_response_times,
            "request_counts": self.request_counts,
            "error_counts": self.error_counts,
            "slow_query_counts": self.slow_query_counts,
        }


class MemoryMonitor:
    """内存监控器

    功能：
    - 定期采集内存快照
    - 内存阈值告警
    - 自动 GC 触发
    - 通知集成（QQ/控制台）
    """

    def __init__(
        self,
        warning_threshold: float = 0.8,
        critical_threshold: float = 0.9,
        check_interval: float = 30.0,
        history_size: int = 100,
        notification_manager: Optional["NotificationManager"] = None,
    ):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.check_interval = check_interval
        self.history_size = history_size

        self._lock = threading.Lock()
        self._history: List[MemorySnapshot] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._cache_memory_callbacks: Dict[str, Callable[[], int]] = {}

        # 告警状态
        self._last_warning_time: float = 0
        self._last_critical_time: float = 0
        self._warning_cooldown: float = 60.0  # 告警冷却时间

        # 通知管理器
        self._notification_manager = notification_manager

    def register_cache_memory_callback(self, name: str, callback: Callable[[], int]):
        """注册缓存内存获取回调

        Args:
            name: 缓存模块名称
            callback: 返回内存使用量（字节）的回调函数
        """
        with self._lock:
            self._cache_memory_callbacks[name] = callback

    def unregister_cache_memory_callback(self, name: str):
        """取消注册缓存内存回调"""
        with self._lock:
            self._cache_memory_callbacks.pop(name, None)

    def take_snapshot(self) -> MemorySnapshot:
        """采集内存快照"""
        now = time.time()

        # 进程内存（需要 psutil）
        process_rss = 0
        process_vms = 0
        if PSUTIL_AVAILABLE and psutil is not None:
            try:
                process = psutil.Process()
                mem_info = process.memory_info()
                process_rss = mem_info.rss
                process_vms = mem_info.vms
            except Exception:
                pass

        # Python 分配的内存（通过 sys.getallocatedblocks 估算）
        try:
            python_allocated = sys.getallocatedblocks() * ALLOC_BLOCK_SIZE
        except Exception:
            python_allocated = 0

        # GC 统计
        try:
            gc_counts = tuple(gc.get_count())  # type: ignore
        except Exception:
            gc_counts = (0, 0, 0)

        # 各缓存模块内存
        cache_memory: Dict[str, int] = {}
        with self._lock:
            for name, callback in self._cache_memory_callbacks.items():
                try:
                    cache_memory[name] = callback()
                except Exception:
                    cache_memory[name] = 0

        return MemorySnapshot(
            timestamp=now,
            process_rss=process_rss,
            process_vms=process_vms,
            python_allocated=python_allocated,
            gc_counts=gc_counts,  # type: ignore
            cache_memory=cache_memory,
        )

    def get_memory_usage_ratio(self) -> float:
        """获取当前内存使用率（0-1）"""
        if not PSUTIL_AVAILABLE or psutil is None:
            return 0.0

        try:
            mem = psutil.virtual_memory()
            return mem.percent / 100.0
        except Exception:
            return 0.0

    def check_thresholds(self) -> Tuple[bool, bool, str]:
        """检查内存阈值

        Returns:
            Tuple[warning, critical, message]
        """
        ratio = self.get_memory_usage_ratio()

        if ratio >= self.critical_threshold:
            return False, True, f"内存使用率 {ratio:.1%} 超过临界阈值 {self.critical_threshold:.1%}"
        if ratio >= self.warning_threshold:
            return True, False, f"内存使用率 {ratio:.1%} 超过警告阈值 {self.warning_threshold:.1%}"

        return False, False, ""

    async def _monitor_loop(self):
        """监控循环"""
        while self._running:
            try:
                # 采集快照
                snapshot = self.take_snapshot()

                with self._lock:
                    self._history.append(snapshot)
                    # 限制历史记录数量
                    while len(self._history) > self.history_size:
                        self._history.pop(0)

                # 检查阈值
                warning, critical, message = self.check_thresholds()
                now = time.time()

                if critical:
                    if now - self._last_critical_time > self._warning_cooldown:
                        logger.error(f"[MemMonitor] ⚠️ {message}")
                        self._last_critical_time = now
                        # 触发 GC
                        gc.collect()
                        logger.info("[MemMonitor] 已触发垃圾回收")
                        # 发送严重告警通知
                        await self._send_critical_notification()
                elif warning:
                    if now - self._last_warning_time > self._warning_cooldown:
                        logger.warning(f"[MemMonitor] {message}")
                        self._last_warning_time = now
                        # 发送警告通知
                        await self._send_warning_notification()

            except Exception as e:
                logger.error(f"[MemMonitor] 监控循环错误: {e}")

            await asyncio.sleep(self.check_interval)

    def set_notification_manager(self, notification_manager: "NotificationManager"):
        """设置通知管理器

        Args:
            notification_manager: 通知管理器实例
        """
        self._notification_manager = notification_manager

    async def _send_warning_notification(self):
        """发送内存警告通知"""
        if self._notification_manager is None:
            return

        try:
            ratio = self.get_memory_usage_ratio()
            await self._notification_manager.send(
                template_key="memory_warning",
                memory_percent=ratio * 100,
                threshold=self.warning_threshold * 100,
            )
        except Exception as e:
            logger.error(f"[MemMonitor] 发送警告通知失败: {e}")

    async def _send_critical_notification(self):
        """发送内存严重告警通知"""
        if self._notification_manager is None:
            return

        try:
            ratio = self.get_memory_usage_ratio()
            await self._notification_manager.send(
                template_key="memory_critical",
                memory_percent=ratio * 100,
                threshold=self.critical_threshold * 100,
            )
        except Exception as e:
            logger.error(f"[MemMonitor] 发送严重告警通知失败: {e}")

    def start(self):
        """启动监控"""
        if self._running:
            return

        self._running = True
        try:
            loop = asyncio.get_running_loop()
            self._task = loop.create_task(self._monitor_loop())
            logger.info("[MemMonitor] 内存监控已启动")
        except RuntimeError:
            logger.warning("[MemMonitor] 无法启动监控（没有运行中的事件循环）")
            self._running = False

    def stop(self):
        """停止监控"""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        logger.info("[MemMonitor] 内存监控已停止")

    def get_history(self, limit: int = 10) -> List[MemorySnapshot]:
        """获取历史快照"""
        with self._lock:
            return list(self._history[-limit:])

    def get_latest_snapshot(self) -> Optional[MemorySnapshot]:
        """获取最新快照"""
        with self._lock:
            if self._history:
                return self._history[-1]
            return None


class StatsReporter:
    """统计报告器

    功能：
    - 收集各模块统计信息
    - 生成格式化报告
    - 支持多种输出格式
    """

    def __init__(self, report_interval: float = 60.0):
        self.report_interval = report_interval
        self._lock = threading.Lock()
        self._stats_callbacks: Dict[str, Callable[[], Dict[str, Any]]] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_report_time: float = 0
        self._memory_monitor: Optional[MemoryMonitor] = None

    def set_memory_monitor(self, monitor: MemoryMonitor):
        """设置内存监控器"""
        self._memory_monitor = monitor

    def register_stats_callback(self, name: str, callback: Callable[[], Dict[str, Any]]):
        """注册统计回调

        Args:
            name: 模块名称
            callback: 返回统计信息的回调函数
        """
        with self._lock:
            self._stats_callbacks[name] = callback

    def unregister_stats_callback(self, name: str):
        """取消注册统计回调"""
        with self._lock:
            self._stats_callbacks.pop(name, None)

    def collect_stats(self) -> Dict[str, Any]:
        """收集所有统计信息"""
        stats: Dict[str, Any] = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "modules": {},
        }

        with self._lock:
            for name, callback in self._stats_callbacks.items():
                try:
                    stats["modules"][name] = callback()
                except Exception as e:
                    stats["modules"][name] = {"error": str(e)}

        # 添加内存信息
        if self._memory_monitor:
            snapshot = self._memory_monitor.get_latest_snapshot()
            if snapshot:
                stats["memory"] = snapshot.to_dict()

        return stats

    def generate_report(self, format_type: str = "text") -> str:
        """生成统计报告。

        Args:
            format_type: 输出格式 ("text", "json", "markdown")
        """
        stats = self.collect_stats()

        if format_type == "json":
            import json

            return json.dumps(stats, indent=2, ensure_ascii=False)
        if format_type == "markdown":
            return self._format_markdown(stats)
        return self._format_text(stats)

    @staticmethod
    def _safe_int(v: Any, default: int = 0) -> int:
        try:
            if v is None:
                return default
            return int(v)
        except Exception:
            return default

    @staticmethod
    def _safe_float(v: Any, default: float = 0.0) -> float:
        try:
            if v is None:
                return default
            return float(v)
        except Exception:
            return default

    @classmethod
    def _calc_rate_pct(
        cls,
        hit: Any,
        miss: Any,
        filtered: Any = 0,
        *,
        include_filtered_in_denom: bool = True,
    ) -> float:
        """计算命中率（百分比）。

        约定：
        - hit/miss/filtered 只统计“进入缓存判定”的请求（通常不包含 skipped/unselected）。
        - filtered 表示“命中但未采用”，对用户而言更接近“未命中”。

        Args:
            hit: 命中次数
            miss: 未命中次数
            filtered: 命中但未采用次数
            include_filtered_in_denom: 是否将 filtered 计入分母。
                - True（默认）：有效命中率 = hit / (hit + miss + filtered)
                - False：命中判定率 = hit / (hit + miss)

        Returns:
            命中率百分比（0-100）。
        """

        h = cls._safe_int(hit)
        m = cls._safe_int(miss)
        f = cls._safe_int(filtered)
        denom = h + m + (f if include_filtered_in_denom else 0)
        return (h / denom * 100.0) if denom > 0 else 0.0

    @classmethod
    def _calc_simple_rate_pct(cls, ok: Any, fail: Any) -> float:
        """计算简单二元命中率（百分比）：ok / (ok + fail)。"""

        o = cls._safe_int(ok)
        f = cls._safe_int(fail)
        denom = o + f
        return (o / denom * 100.0) if denom > 0 else 0.0

    @classmethod
    def _extract_counters(cls, module_stats: Dict[str, Any]) -> Dict[str, Any]:
        """从模块 stats dict 提取并归一化常用统计字段。

        统一字段（缺失按 0）：
        - 命中/未命中/过滤/跳过/未走缓存路径：t_hit/t_miss/t_filtered/t_skipped/t_unselected
        - DB 查询耗时（由 miss(elapsed) 记录，按 0.1s 快慢分桶）：t_fast/t_slow + time
        - message_cache 专用：t_hotset_hit/t_hotset_miss（热集缓存命中/未命中）

        兼容性：
        - 某些 fallback 统计使用 hits/misses/... 命名，这里会做映射。
        """

        # ---- legacy mapping (fallback ModuleStats) ----
        if "t_hit" not in module_stats and "hits" in module_stats:
            module_stats = dict(module_stats)
            module_stats.setdefault("t_hit", module_stats.get("hits"))
            module_stats.setdefault("t_miss", module_stats.get("misses"))
            module_stats.setdefault("t_skipped", module_stats.get("skipped"))
            module_stats.setdefault("t_filtered", module_stats.get("filtered"))
            module_stats.setdefault("t_unselected", module_stats.get("unselected"))

        t_hit = cls._safe_int(module_stats.get("t_hit"))
        t_miss = cls._safe_int(module_stats.get("t_miss"))
        t_filtered = cls._safe_int(module_stats.get("t_filtered"))
        t_skipped = cls._safe_int(module_stats.get("t_skipped"))
        t_unselected = cls._safe_int(module_stats.get("t_unselected"))
        t_fast = cls._safe_int(module_stats.get("t_fast"))
        t_slow = cls._safe_int(module_stats.get("t_slow"))
        t_fast_time = cls._safe_float(module_stats.get("t_fast_time"))
        t_slow_time = cls._safe_float(module_stats.get("t_slow_time"))

        i_hit = cls._safe_int(module_stats.get("i_hit"))
        i_miss = cls._safe_int(module_stats.get("i_miss"))
        i_filtered = cls._safe_int(module_stats.get("i_filtered"))
        i_skipped = cls._safe_int(module_stats.get("i_skipped"))
        i_unselected = cls._safe_int(module_stats.get("i_unselected"))
        i_fast = cls._safe_int(module_stats.get("i_fast"))
        i_slow = cls._safe_int(module_stats.get("i_slow"))
        i_fast_time = cls._safe_float(module_stats.get("i_fast_time"))
        i_slow_time = cls._safe_float(module_stats.get("i_slow_time"))

        # message_cache extra (hotset)
        t_hotset_hit = cls._safe_int(module_stats.get("t_hotset_hit"))
        t_hotset_miss = cls._safe_int(module_stats.get("t_hotset_miss"))
        i_hotset_hit = cls._safe_int(module_stats.get("i_hotset_hit"))
        i_hotset_miss = cls._safe_int(module_stats.get("i_hotset_miss"))

        # 命中率（默认将 filtered 计入分母，避免“命中但未采用”导致的误导）
        t_effective_hit_rate = cls._calc_rate_pct(
            t_hit, t_miss, t_filtered, include_filtered_in_denom=True
        )
        i_effective_hit_rate = cls._calc_rate_pct(
            i_hit, i_miss, i_filtered, include_filtered_in_denom=True
        )
        t_decision_hit_rate = cls._calc_rate_pct(
            t_hit, t_miss, t_filtered, include_filtered_in_denom=False
        )
        i_decision_hit_rate = cls._calc_rate_pct(
            i_hit, i_miss, i_filtered, include_filtered_in_denom=False
        )

        t_hotset_hit_rate = cls._calc_simple_rate_pct(t_hotset_hit, t_hotset_miss)
        i_hotset_hit_rate = cls._calc_simple_rate_pct(i_hotset_hit, i_hotset_miss)

        t_fast_avg = (t_fast_time / t_fast) if t_fast > 0 else 0.0
        t_slow_avg = (t_slow_time / t_slow) if t_slow > 0 else 0.0
        i_fast_avg = (i_fast_time / i_fast) if i_fast > 0 else 0.0
        i_slow_avg = (i_slow_time / i_slow) if i_slow > 0 else 0.0

        return {
            "t_hit": t_hit,
            "t_miss": t_miss,
            "t_filtered": t_filtered,
            "t_skipped": t_skipped,
            "t_unselected": t_unselected,
            "t_fast": t_fast,
            "t_slow": t_slow,
            "t_fast_time": t_fast_time,
            "t_slow_time": t_slow_time,
            "i_hit": i_hit,
            "i_miss": i_miss,
            "i_filtered": i_filtered,
            "i_skipped": i_skipped,
            "i_unselected": i_unselected,
            "i_fast": i_fast,
            "i_slow": i_slow,
            "i_fast_time": i_fast_time,
            "i_slow_time": i_slow_time,
            "t_effective_hit_rate": t_effective_hit_rate,
            "i_effective_hit_rate": i_effective_hit_rate,
            "t_decision_hit_rate": t_decision_hit_rate,
            "i_decision_hit_rate": i_decision_hit_rate,
            "t_fast_avg": t_fast_avg,
            "t_slow_avg": t_slow_avg,
            "i_fast_avg": i_fast_avg,
            "i_slow_avg": i_slow_avg,
            "t_hotset_hit": t_hotset_hit,
            "t_hotset_miss": t_hotset_miss,
            "i_hotset_hit": i_hotset_hit,
            "i_hotset_miss": i_hotset_miss,
            "t_hotset_hit_rate": t_hotset_hit_rate,
            "i_hotset_hit_rate": i_hotset_hit_rate,
        }

    def _format_text(self, stats: Dict[str, Any]) -> str:
        """文本格式报告（中文 + 更清晰的分组）。"""

        width = 72
        lines = [
            "=" * width,
            f"CM 性能优化插件统计报告（中文） - {stats.get('datetime', 'N/A')}",
            "=" * width,
        ]

        modules = stats.get("modules", {})
        if modules:
            lines.append("\n【模块统计】")

        for module_name, module_stats in modules.items():
            lines.append(f"\n📊 模块：{module_name}")

            if not isinstance(module_stats, dict):
                lines.append(f"  统计：{module_stats}")
                continue

            if "error" in module_stats:
                lines.append(f"  ❌ 错误：{module_stats.get('error')}")
                continue

            c = self._extract_counters(module_stats)

            # ---- message_cache: hotset + query-cache 分离展示 ----
            has_hotset = (c.get("t_hotset_hit", 0) + c.get("t_hotset_miss", 0)) > 0
            if module_name == "message_cache" or has_hotset:
                lines.append(
                    f"  热集缓存: 命中 {c['t_hotset_hit']} 次 ({c['t_hotset_hit_rate']:.1f}%)"
                )

                lines.append(
                    "  查询缓存: "
                    f"命中 {c['t_hit']} 次 | 未命中 {c['t_miss']} 次 ({c['t_effective_hit_rate']:.1f}%)"
                )

                # unselected 对 message_cache 常见含义：热集命中时未走 query-cache
                t_unselected = int(c.get("t_unselected", 0) or 0)
                if t_unselected > 0 and t_unselected != int(c.get("t_hotset_hit", 0) or 0):
                    lines.append(f"  未走查询缓存: {t_unselected} 次")

                lines.append(
                    f"  跳过: {c['t_skipped']} 次 | 过滤: {c['t_filtered']} 次"
                )

                # miss(elapsed) 代表 DB 查询耗时（仅统计进入缓存判定且未命中的请求）
                lines.append(
                    "  数据库查询平均耗时: "
                    f"快 {c['t_fast_avg']:.4f}s | 慢 {c['t_slow_avg']:.4f}s"
                )
                continue

            # ---- generic modules ----
            lines.append(
                "  缓存: "
                f"命中 {c['t_hit']} 次 | 未命中 {c['t_miss']} 次 ({c['t_effective_hit_rate']:.1f}%)"
            )
            lines.append(
                "  "
                f"跳过: {c['t_skipped']} 次 | 过滤: {c['t_filtered']} 次 | 未走缓存: {c['t_unselected']} 次"
            )
            lines.append(
                "  数据库查询平均耗时: "
                f"快 {c['t_fast_avg']:.4f}s | 慢 {c['t_slow_avg']:.4f}s"
            )

        # 内存信息
        memory = stats.get("memory")
        if memory:
            lines.append("\n【内存使用】")
            lines.append(f"  进程 RSS：{memory.get('process_rss_mb', 0)} MB")
            lines.append(f"  Python 分配：{memory.get('python_allocated_mb', 0)} MB")
            lines.append(f"  缓存总计：{memory.get('total_cache_memory_mb', 0)} MB")

            cache_memory = memory.get("cache_memory", {})
            if cache_memory:
                lines.append("  各缓存模块：")
                for cache_name, size_mb in cache_memory.items():
                    lines.append(f"    - {cache_name}: {size_mb} MB")

        lines.append("=" * width)
        return "\n".join(lines)

    def _format_markdown(self, stats: Dict[str, Any]) -> str:
        """Markdown 格式报告（中文 + 更清晰的分组）。"""

        lines = [
            "# CM 性能优化插件统计报告（中文）",
            f"**时间**：{stats.get('datetime', 'N/A')}",
            "",
        ]

        modules = stats.get("modules", {})
        if modules:
            lines.append("## 模块统计")

        for module_name, module_stats in modules.items():
            lines.append(f"\n### {module_name}")

            if not isinstance(module_stats, dict):
                lines.append(f"统计：{module_stats}")
                continue

            if "error" in module_stats:
                lines.append(f"❌ **错误**：{module_stats.get('error')}")
                continue

            c = self._extract_counters(module_stats)
            has_hotset = (c.get("t_hotset_hit", 0) + c.get("t_hotset_miss", 0)) > 0

            if module_name == "message_cache" or has_hotset:
                lines.append(
                    f"- **热集缓存**：命中 {c['t_hotset_hit']} 次（{c['t_hotset_hit_rate']:.1f}%）"
                )
                lines.append(
                    f"- **查询缓存**：命中 {c['t_hit']} 次｜未命中 {c['t_miss']} 次（{c['t_effective_hit_rate']:.1f}%）"
                )

                t_unselected = int(c.get("t_unselected", 0) or 0)
                if t_unselected > 0 and t_unselected != int(c.get("t_hotset_hit", 0) or 0):
                    lines.append(f"- **未走查询缓存**：{t_unselected} 次")

                lines.append(f"- 跳过：{c['t_skipped']} 次｜过滤：{c['t_filtered']} 次")
                lines.append(
                    f"- 数据库查询平均耗时：快 {c['t_fast_avg']:.4f}s｜慢 {c['t_slow_avg']:.4f}s"
                )
                continue

            lines.append(
                f"- **缓存**：命中 {c['t_hit']} 次｜未命中 {c['t_miss']} 次（{c['t_effective_hit_rate']:.1f}%）"
            )
            lines.append(
                f"- 跳过：{c['t_skipped']} 次｜过滤：{c['t_filtered']} 次｜未走缓存：{c['t_unselected']} 次"
            )
            lines.append(
                f"- 数据库查询平均耗时：快 {c['t_fast_avg']:.4f}s｜慢 {c['t_slow_avg']:.4f}s"
            )

        memory = stats.get("memory")
        if memory:
            lines.append("\n## 内存使用")
            lines.append("| 指标 | 值 (MB) |")
            lines.append("|------|---------|")
            lines.append(f"| 进程 RSS | {memory.get('process_rss_mb', 0)} |")
            lines.append(f"| Python 分配 | {memory.get('python_allocated_mb', 0)} |")
            lines.append(f"| 缓存总计 | {memory.get('total_cache_memory_mb', 0)} |")

            cache_memory = memory.get("cache_memory", {})
            if cache_memory:
                lines.append("\n### 缓存模块内存")
                lines.append("| 模块 | 大小 (MB) |")
                lines.append("|------|-----------|")
                for cache_name, size_mb in cache_memory.items():
                    lines.append(f"| {cache_name} | {size_mb} |")

        return "\n".join(lines)

    async def _report_loop(self):
        """报告循环"""
        while self._running:
            try:
                report = self.generate_report("text")
                logger.info(f"\n{report}")
                self._last_report_time = time.time()
            except Exception as e:
                logger.error(f"[StatsReporter] 生成报告失败: {e}")

            await asyncio.sleep(self.report_interval)

    def start(self):
        """启动定期报告"""
        if self._running:
            return

        self._running = True
        try:
            loop = asyncio.get_running_loop()
            self._task = loop.create_task(self._report_loop())
            logger.info("[StatsReporter] 统计报告已启动")
        except RuntimeError:
            logger.warning("[StatsReporter] 无法启动（没有运行中的事件循环）")
            self._running = False

    def stop(self):
        """停止定期报告"""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        logger.info("[StatsReporter] 统计报告已停止")


class PerformanceCollector:
    """性能指标收集器

    用于收集和聚合各模块的性能指标
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._metrics: Dict[str, PerformanceMetrics] = {}
        self._interval_start: float = time.time()

    def record_hit(self, module: str):
        """记录缓存命中"""
        with self._lock:
            if module not in self._metrics:
                self._metrics[module] = PerformanceMetrics()
            self._metrics[module].request_counts[module] = (
                self._metrics[module].request_counts.get(module, 0) + 1
            )

    def record_miss(self, module: str, response_time_ms: float):
        """记录缓存未命中"""
        with self._lock:
            if module not in self._metrics:
                self._metrics[module] = PerformanceMetrics()

            m = self._metrics[module]
            m.request_counts[module] = m.request_counts.get(module, 0) + 1

            # 更新平均响应时间
            count = m.request_counts.get(module, 1)
            current_avg = m.avg_response_times.get(module, 0)
            m.avg_response_times[module] = (
                (current_avg * (count - 1) + response_time_ms) / count
            )

            # 慢查询（超过 100ms）
            if response_time_ms > 100:
                m.slow_query_counts[module] = m.slow_query_counts.get(module, 0) + 1

    def record_error(self, module: str):
        """记录错误"""
        with self._lock:
            if module not in self._metrics:
                self._metrics[module] = PerformanceMetrics()
            self._metrics[module].error_counts[module] = (
                self._metrics[module].error_counts.get(module, 0) + 1
            )

    def get_metrics(self) -> Dict[str, PerformanceMetrics]:
        """获取所有指标"""
        with self._lock:
            return {k: v for k, v in self._metrics.items()}

    def reset_interval(self) -> Dict[str, Dict[str, Any]]:
        """重置间隔统计并返回"""
        with self._lock:
            result = {k: v.to_dict() for k, v in self._metrics.items()}
            self._metrics.clear()
            self._interval_start = time.time()
            return result


# 全局实例
_memory_monitor: Optional[MemoryMonitor] = None
_stats_reporter: Optional[StatsReporter] = None
_perf_collector: Optional[PerformanceCollector] = None


def get_memory_monitor() -> MemoryMonitor:
    """获取内存监控器单例"""
    global _memory_monitor
    if _memory_monitor is None:
        _memory_monitor = MemoryMonitor()
    return _memory_monitor


def get_stats_reporter() -> StatsReporter:
    """获取统计报告器单例"""
    global _stats_reporter
    if _stats_reporter is None:
        _stats_reporter = StatsReporter()
        _stats_reporter.set_memory_monitor(get_memory_monitor())
    return _stats_reporter


def get_perf_collector() -> PerformanceCollector:
    """获取性能收集器单例"""
    global _perf_collector
    if _perf_collector is None:
        _perf_collector = PerformanceCollector()
    return _perf_collector
