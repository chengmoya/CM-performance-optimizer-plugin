"""message_repository.count_messages 快速路径（仅 count_messages）。

该模块通过 monkey-patch 将 [`src.common.message_repository.count_messages()`](../src/common/message_repository.py:132)
替换为带短期缓存的 COUNT 查询，从而降低高频 count 的数据库压力。

重要约束：
- 只 patch count_messages，不 patch find_messages。
- 与 message_cache 的 find_messages patch 不冲突。

实现要点：
- 使用 peewee 的 [`peewee.fn.COUNT()`](https://docs.peewee-orm.com/en/latest/peewee/querying.html) 仅选择 COUNT 字段。
- 生成稳定的 filter cache key（排序 + 递归规范化）。
- 对相同 filter 的计数结果做短期 TTL 缓存。
- 对 filter → where 条件编译结果做中期缓存（减少重复构造表达式的开销）。

"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

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

    # 尝试预加载 core 子模块，避免 __init__ 中的相对导入失败
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


# ========== 核心模块加载（ModuleStats / MemoryUtils） ==========
try:
    from ..core import MemoryUtils, ModuleStats
except Exception:
    try:
        _core = _load_core_module()
        ModuleStats = _core.ModuleStats
        MemoryUtils = _core.MemoryUtils
    except Exception as e:  # pragma: no cover
        logger.warning(f"[MessageRepositoryFastpath] 无法加载核心模块，使用内置实现: {e}")

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


def _stats_hit(stats: Any) -> None:
    try:
        if hasattr(stats, "hit"):
            stats.hit()
    except Exception:
        pass


def _stats_miss(stats: Any) -> None:
    """对不同 ModuleStats 实现保持兼容：miss(elapsed) / miss()。"""

    try:
        if not hasattr(stats, "miss"):
            return
        try:
            stats.miss(0.0)
        except TypeError:
            stats.miss()
    except Exception:
        pass


def _canonicalize(obj: Any) -> Any:
    """将任意对象递归规范化为“稳定可序列化”的结构。

    目标：
    - 同一 filter 的逻辑结构相同 → 规范化后结构相同
    - 支持 dict/list/tuple/set 等
    - 对不可 JSON 序列化的对象使用 repr 作为稳定表示
    """

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, bytes):
        # bytes 直接 repr，避免 decode 误差
        return {"__bytes__": repr(obj)}

    if isinstance(obj, dict):
        items = []
        for k, v in obj.items():
            items.append((str(k), _canonicalize(v)))
        items.sort(key=lambda x: x[0])
        return {k: v for k, v in items}

    if isinstance(obj, (list, tuple)):
        return [_canonicalize(i) for i in obj]

    if isinstance(obj, (set, frozenset)):
        # set 无序，转为排序后的列表
        canon_items = [_canonicalize(i) for i in obj]
        try:
            canon_items.sort(key=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False, default=repr))
        except Exception:
            canon_items.sort(key=lambda x: repr(x))
        return canon_items

    # 兜底
    return {"__repr__": repr(obj)}


def _quantize_filter_for_key(
    message_filter: Dict[str, Any],
    quantum: float = 5.0,
) -> Dict[str, Any]:
    """对 filter 中的 time 字段做量化处理，用于生成稳定的缓存 key。

    注意：该量化仅用于缓存 key（count 结果缓存），不改变实际 DB 查询的 filter。

    Args:
        message_filter: 原始 filter。
        quantum: 量化窗口（秒）。默认 5 秒。

    Returns:
        量化后的 filter（浅拷贝）。若无需量化则返回原 dict。
    """

    if not isinstance(message_filter, dict):
        return message_filter

    try:
        q = float(quantum)
    except Exception:
        return message_filter

    if q <= 0:
        return message_filter

    time_cond = message_filter.get("time")
    if not isinstance(time_cond, dict):
        return message_filter

    changed = False
    quantized_time: Dict[str, Any] = {}
    for op, val in time_cond.items():
        if isinstance(val, (int, float)):
            qv = (float(val) // q) * q
            if qv != float(val):
                changed = True
            quantized_time[op] = qv
            continue

        if isinstance(val, str):
            try:
                fval = float(val)
                qv = (fval // q) * q
                if qv != fval:
                    changed = True
                quantized_time[op] = qv
                continue
            except Exception:
                pass

        quantized_time[op] = val

    if not changed:
        return message_filter

    out = dict(message_filter)
    out["time"] = quantized_time

    try:
        logger.debug(
            f"[MessageRepositoryFastpath] key time-quantize quantum={q}s time={time_cond}->{quantized_time}"
        )
    except Exception:
        pass

    return out


def _stable_filter_key(message_filter: Dict[str, Any]) -> str:
    """生成 filter 的稳定序列化 key。

    说明：
    - 先做递归规范化，再 JSON 序列化（sort_keys），最后 sha1 压缩为短 key。
    - 避免 key 过长导致缓存字典内存膨胀。
    """

    try:
        canon = _canonicalize(message_filter)
        raw = json.dumps(canon, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=repr)
    except Exception:
        # 兼容用户给出的参考实现
        items = sorted(message_filter.items(), key=lambda x: str(x[0]))
        raw = str(items)

    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()


class _TTLCache:
    """简单 TTL 缓存（OrderedDict + timestamp），支持大小限制与过期清理。"""

    def __init__(self, ttl_seconds: float, max_entries: int = 1000):
        self.ttl_seconds = float(ttl_seconds)
        self.max_entries = int(max_entries)
        self._lock = threading.RLock()
        self._data: "OrderedDict[str, Tuple[Any, float]]" = OrderedDict()

        # 轻量统计
        self.hits = 0
        self.misses = 0
        self.expired = 0
        self.evicted = 0

    def _purge_expired(self, now: float) -> None:
        if self.ttl_seconds <= 0:
            return

        while self._data:
            _, (_, ts) = next(iter(self._data.items()))
            if (now - ts) < self.ttl_seconds:
                break
            self._data.popitem(last=False)
            self.expired += 1

    def _enforce_size(self) -> None:
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)
            self.evicted += 1

    def get(self, key: str, now: float) -> Optional[Any]:
        with self._lock:
            self._purge_expired(now)
            if key not in self._data:
                self.misses += 1
                return None

            value, ts = self._data.get(key, (None, 0.0))
            if self.ttl_seconds > 0 and (now - ts) >= self.ttl_seconds:
                # 理论上已在 purge 处理，但这里再兜底一次
                self._data.pop(key, None)
                self.misses += 1
                self.expired += 1
                return None

            # LRU-ish：命中时 move_to_end
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: str, value: Any, now: float) -> None:
        with self._lock:
            self._data[key] = (value, now)
            self._data.move_to_end(key)
            self._purge_expired(now)
            self._enforce_size()

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def size(self) -> int:
        with self._lock:
            return int(len(self._data))

    def get_memory_usage(self) -> int:
        try:
            with self._lock:
                return int(MemoryUtils.get_size(self._data))
        except Exception:
            return 0


class MessageRepositoryFastpathModule:
    """message_repository.count_messages 的快速路径补丁模块。"""

    def __init__(self, count_cache_ttl: float = 3.0):
        """初始化模块。

        Args:
            count_cache_ttl: count 结果缓存时间（秒）。建议 2-5 秒。
        """

        self.stats = ModuleStats("message_repository_fastpath")

        self.count_cache_ttl = float(count_cache_ttl)
        self.count_cache_max_entries = 1000

        # filter 编译缓存（where 条件表达式），TTL 适当更长
        self._compiled_cache_ttl = 30.0
        self._compiled_cache_max_entries = 1000

        self._count_cache = _TTLCache(self.count_cache_ttl, self.count_cache_max_entries)
        self._compiled_where_cache = _TTLCache(
            self._compiled_cache_ttl, self._compiled_cache_max_entries
        )

        self._lock = threading.RLock()
        self._patched = False

        self._orig_func: Optional[Callable[[Dict[str, Any]], int]] = None
        self._fast_func: Optional[Callable[[Dict[str, Any]], int]] = None

        self._calls = 0
        self._fast_calls = 0
        self._fallback_calls = 0
        self._fast_errors = 0

    def _compile_conditions(self, message_filter: Dict[str, Any], now: float) -> Tuple[Any, ...]:
        """将 filter 编译为 Peewee where 条件（带缓存）。"""

        # 仅对 key 做 time 量化（默认 5 秒），提升短时间内相同 filter 的编译缓存命中率。
        # 注意：实际 where 条件仍使用原始 message_filter。
        key_filter = _quantize_filter_for_key(message_filter, quantum=5.0)
        cache_key = _stable_filter_key(key_filter)
        cached = self._compiled_where_cache.get(cache_key, now)
        if cached is not None:
            if isinstance(cached, tuple):
                return cached

        try:
            from src.common.database.database_model import Messages

            conditions = []
            if message_filter:
                for key, value in message_filter.items():
                    if hasattr(Messages, key):
                        field = getattr(Messages, key)
                        if isinstance(value, dict):
                            for op, op_value in value.items():
                                if op == "$gt":
                                    conditions.append(field > op_value)
                                elif op == "$lt":
                                    conditions.append(field < op_value)
                                elif op == "$gte":
                                    conditions.append(field >= op_value)
                                elif op == "$lte":
                                    conditions.append(field <= op_value)
                                elif op == "$ne":
                                    conditions.append(field != op_value)
                                elif op == "$in":
                                    conditions.append(field.in_(op_value))
                                elif op == "$nin":
                                    conditions.append(field.not_in(op_value))
                                else:
                                    logger.warning(
                                        f"[MessageRepositoryFastpath] 计数时，过滤器中遇到未知操作符 '{op}' (字段: '{key}')。将跳过此操作符。"
                                    )
                        else:
                            conditions.append(field == value)
                    else:
                        logger.warning(
                            f"[MessageRepositoryFastpath] 计数时，过滤器键 '{key}' 在 Messages 模型中未找到。将跳过此条件。"
                        )

            # 与原始实现一致：排除 message_id == "notice"
            conditions.append(Messages.message_id != "notice")

            compiled = tuple(conditions)
            self._compiled_where_cache.set(cache_key, compiled, now)
            return compiled
        except Exception as e:
            logger.debug(f"[MessageRepositoryFastpath] 编译 where 条件失败，降级为不缓存: {e}")
            return tuple()

    def apply_patch(self) -> None:
        """应用 monkey-patch（仅 count_messages）。"""

        with self._lock:
            if self._patched:
                return

            try:
                from src.common import message_repository

                self._orig_func = getattr(message_repository, "count_messages", None)
                if not callable(self._orig_func):
                    raise AttributeError("src.common.message_repository.count_messages 不存在或不可调用")

                module = self
                original_count_messages = self._orig_func

                def fast_count_messages(message_filter: Dict[str, Any]) -> int:
                    """count_messages 快速路径：短期缓存 + COUNT 查询 + 异常回退。"""

                    now = time.time()

                    with module._lock:
                        module._calls += 1

                    try:
                        key_filter = _quantize_filter_for_key(message_filter, quantum=5.0)
                        cache_key = _stable_filter_key(key_filter)
                    except Exception:
                        cache_key = "__key_error__"

                    cached = module._count_cache.get(cache_key, now)
                    if cached is not None:
                        _stats_hit(module.stats)
                        try:
                            return int(cached)
                        except Exception:
                            # 缓存污染时，忽略并继续走 miss
                            pass

                    # miss：走快速 COUNT
                    try:
                        from peewee import fn
                        from src.common.database.database_model import Messages

                        conditions = module._compile_conditions(message_filter, now)

                        query = Messages.select(fn.COUNT(Messages.id))
                        if conditions:
                            query = query.where(*conditions)

                        value = query.scalar()
                        result = int(value or 0)

                        with module._lock:
                            module._fast_calls += 1

                    except Exception as e:
                        with module._lock:
                            module._fast_errors += 1
                            module._fallback_calls += 1

                        logger.debug(
                            f"[MessageRepositoryFastpath] fast COUNT 失败，回退原始实现: {e}"
                        )
                        result = int(original_count_messages(message_filter))

                    module._count_cache.set(cache_key, result, now)
                    _stats_miss(module.stats)
                    return int(result)

                # 1) patch 主目标：模块函数本体
                message_repository.count_messages = fast_count_messages  # type: ignore[assignment]

                # 2) patch alias import（在其他模块中 from ... import count_messages 的情况）
                for _, mod in list(sys.modules.items()):
                    try:
                        if mod is None:
                            continue
                        if getattr(mod, "count_messages", None) is original_count_messages:
                            setattr(mod, "count_messages", fast_count_messages)
                    except Exception:
                        continue

                self._fast_func = fast_count_messages
                self._patched = True
                logger.info("[MessageRepositoryFastpath] ✓ 补丁应用成功（仅 count_messages）")
            except Exception as e:
                logger.error(f"[MessageRepositoryFastpath] ✗ 补丁失败: {e}")

    def remove_patch(self) -> None:
        """移除 monkey-patch 并恢复原始实现。"""

        with self._lock:
            if not self._patched:
                return

            try:
                from src.common import message_repository

                if self._orig_func is not None:
                    # 恢复 message_repository 模块中的引用
                    message_repository.count_messages = self._orig_func  # type: ignore[assignment]

                # 恢复 alias import 的引用
                for _, mod in list(sys.modules.items()):
                    try:
                        if mod is None:
                            continue
                        if self._fast_func is not None and getattr(mod, "count_messages", None) is self._fast_func:
                            if self._orig_func is not None:
                                setattr(mod, "count_messages", self._orig_func)
                    except Exception:
                        continue

                self._patched = False
                logger.info("[MessageRepositoryFastpath] 补丁已移除")
            except Exception as e:
                logger.error(f"[MessageRepositoryFastpath] 移除补丁失败: {e}")

    def get_memory_usage(self) -> int:
        """返回缓存内存占用（字节）。"""

        try:
            return int(
                MemoryUtils.get_size(
                    {
                        "count_cache": self._count_cache,
                        "compiled_where_cache": self._compiled_where_cache,
                        "calls": self._calls,
                        "fast_calls": self._fast_calls,
                        "fallback_calls": self._fallback_calls,
                        "fast_errors": self._fast_errors,
                    }
                )
            )
        except Exception:
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """返回统计信息。"""

        with self._lock:
            calls = int(self._calls)
            fast_calls = int(self._fast_calls)
            fallback_calls = int(self._fallback_calls)
            fast_errors = int(self._fast_errors)

        out: Dict[str, Any] = {
            "patched": bool(self._patched),
            "count_cache_ttl": float(self.count_cache_ttl),
            "count_cache_size": self._count_cache.size(),
            "compiled_cache_ttl": float(self._compiled_cache_ttl),
            "compiled_cache_size": self._compiled_where_cache.size(),
            "calls": calls,
            "fast_calls": fast_calls,
            "fallback_calls": fallback_calls,
            "fast_errors": fast_errors,
            "count_cache_hits": int(self._count_cache.hits),
            "count_cache_misses": int(self._count_cache.misses),
            "count_cache_expired": int(self._count_cache.expired),
            "count_cache_evicted": int(self._count_cache.evicted),
            "compiled_cache_hits": int(self._compiled_where_cache.hits),
            "compiled_cache_misses": int(self._compiled_where_cache.misses),
            "compiled_cache_expired": int(self._compiled_where_cache.expired),
            "compiled_cache_evicted": int(self._compiled_where_cache.evicted),
        }

        try:
            if hasattr(self.stats, "total"):
                total = self.stats.total()
                if isinstance(total, dict):
                    out.update(total)
        except Exception:
            pass

        return out


def apply_message_repository_fastpath(
    cache_manager: Any,
) -> Optional[MessageRepositoryFastpathModule]:
    """工厂函数：创建模块、注册到 cache_manager 并应用补丁。"""

    try:
        mod = MessageRepositoryFastpathModule()
        cache_manager.register_cache("message_repository_fastpath", mod)
        mod.apply_patch()
        logger.info("[MessageRepositoryFastpath] ✓ 模块已初始化")
        return mod
    except Exception as e:
        logger.error(f"[MessageRepositoryFastpath] 初始化失败: {e}")
        return None
