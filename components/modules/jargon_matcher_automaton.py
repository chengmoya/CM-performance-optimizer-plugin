"""jargon_matcher_automaton - 黑话匹配自动机加速模块。

该模块通过 monkey-patch 将黑话匹配从「对每条黑话逐个正则扫描」优化为
Aho-Corasick 自动机的线性扫描。

Patch 目标：
- [`src.bw_learner.jargon_explainer.JargonExplainer.match_jargon_from_messages()`](../src/bw_learner/jargon_explainer.py:52)
- [`src.bw_learner.jargon_explainer.match_jargon_from_text()`](../src/bw_learner/jargon_explainer.py:259)

实现要点：
- 优先从 jargon_cache（本插件的 [`JargonCacheModule.get_all()`](CM-performance-optimizer-plugin/components/modules/jargon_cache.py:224)）
  获取全量黑话数据作为构建源。
- 若缓存不可用/未加载，则回退到 DB 查询。
- 优先使用 `pyahocorasick`（C 扩展）；缺失时使用纯 Python Aho-Corasick（性能较弱但仍为线性）。

注意：
- 需要保持原始行为的关键过滤逻辑：meaning 非空、all_global_jargon 开关、chat_id 列表过滤、
  跳过包含机器人昵称的词条，以及英文/数字词条的“单词边界”匹配。
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

try:
    from src.common.logger import get_logger
except ImportError:  # pragma: no cover
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger("CM_perf_opt")


# ========== pyahocorasick 探测（模块级） ==========
try:
    import ahocorasick  # type: ignore

    _HAS_PYAHO = True
except Exception:  # pragma: no cover
    ahocorasick = None  # type: ignore[assignment]
    _HAS_PYAHO = False


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
        logger.warning(f"[JargonMatcherAutomaton] 无法加载核心模块，使用内置实现: {e}")

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


def _contains_cjk(text: str) -> bool:
    """粗略判断字符串是否包含 CJK（用于决定是否启用 word boundary）。"""

    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            return True
    return False


def _is_word_char(ch: str) -> bool:
    """模拟 regex \b 的 \w 判定（足够贴近当前黑话匹配需求）。"""

    if not ch:
        return False
    # 仅需要服务于“纯英文/数字”内容，使用 ASCII 语义即可
    return ch.isalnum() or ch == "_"


@dataclass(frozen=True)
class _JargonMeta:
    """匹配元数据。"""

    content: str
    length: int
    requires_word_boundary: bool


class PyACAutomaton:
    """纯 Python Aho-Corasick 自动机（用于无 pyahocorasick 时降级）。"""

    def __init__(self) -> None:
        self._next: List[Dict[str, int]] = [dict()]
        self._fail: List[int] = [0]
        self._out: List[List[str]] = [[]]
        self._built = False

    def add_word(self, word: str, value: str) -> None:
        if self._built:
            raise RuntimeError("automaton already built")
        node = 0
        for ch in word:
            nxt = self._next[node].get(ch)
            if nxt is None:
                nxt = len(self._next)
                self._next[node][ch] = nxt
                self._next.append(dict())
                self._fail.append(0)
                self._out.append([])
            node = nxt
        self._out[node].append(value)

    def make_automaton(self) -> None:
        from collections import deque

        q: deque[int] = deque()
        # depth-1 nodes
        for ch, nxt in self._next[0].items():
            self._fail[nxt] = 0
            q.append(nxt)

        while q:
            r = q.popleft()
            for ch, u in self._next[r].items():
                q.append(u)
                v = self._fail[r]
                while v and ch not in self._next[v]:
                    v = self._fail[v]
                self._fail[u] = self._next[v].get(ch, 0)
                self._out[u].extend(self._out[self._fail[u]])

        self._built = True

    def iter(self, text: str) -> Iterator[Tuple[int, str]]:
        if not self._built:
            raise RuntimeError("automaton not built")
        state = 0
        for i, ch in enumerate(text):
            while state and ch not in self._next[state]:
                state = self._fail[state]
            state = self._next[state].get(ch, 0)
            if self._out[state]:
                for v in self._out[state]:
                    yield i, v


class AutomatonManager:
    """管理每个 chat 维度的自动机（带重建间隔、缓存源优先）。"""

    def __init__(self, cache_manager: Any, rebuild_interval: int = 300) -> None:
        self._cache_manager = cache_manager
        self._lock = threading.Lock()
        self._rebuild_interval = max(30, int(rebuild_interval))

        # key -> (automaton, meta_map, built_at, backend, jargon_count)
        self._automatons: Dict[str, Tuple[Any, Dict[str, _JargonMeta], float, str, int]] = {}

        # 统计
        self.builds = 0
        self.rebuilds = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.last_error: Optional[str] = None

        # 运行后端
        self._backend = "pyahocorasick" if _HAS_PYAHO else "python"

    @property
    def backend(self) -> str:
        return str(self._backend)

    def _cache_key(self, chat_id: str) -> str:
        """将 all_global_jargon 纳入 key，避免配置热切换导致行为不一致。"""

        try:
            from src.config.config import global_config

            ag = bool(getattr(getattr(global_config, "expression", None), "all_global_jargon", False))
        except Exception:
            ag = False
        return f"{chat_id}|ag:{1 if ag else 0}"

    def rebuild_automaton(self, chat_id: str) -> None:
        """强制重建指定 chat 的自动机。"""

        key = self._cache_key(chat_id)
        with self._lock:
            self._automatons.pop(key, None)
        _ = self.get_or_build(chat_id, force_rebuild=True)

    def get_or_build(self, chat_id: str, force_rebuild: bool = False) -> Tuple[Any, Dict[str, _JargonMeta]]:
        key = self._cache_key(chat_id)

        with self._lock:
            cached = self._automatons.get(key)
            if cached and not force_rebuild:
                age = time.time() - float(cached[2])
                if age < self._rebuild_interval:
                    self.cache_hits += 1
                    return cached[0], cached[1]

        self.cache_misses += 1
        return self._build_and_store(chat_id, key=key, forced=force_rebuild)

    def _build_and_store(self, chat_id: str, key: str, forced: bool) -> Tuple[Any, Dict[str, _JargonMeta]]:
        try:
            t0 = time.time()
            jargons = self._get_jargons(chat_id)

            # 后端选择
            if _HAS_PYAHO and ahocorasick is not None:
                automaton: Any = ahocorasick.Automaton()  # type: ignore[attr-defined]
                backend = "pyahocorasick"
                add_word = automaton.add_word
                finalize = automaton.make_automaton
            else:
                automaton = PyACAutomaton()
                backend = "python"
                add_word = automaton.add_word
                finalize = automaton.make_automaton

            meta_map: Dict[str, _JargonMeta] = {}

            for jargon in jargons:
                content = getattr(jargon, "content", None)
                meaning = getattr(jargon, "meaning", None)
                if content is None:
                    # 兼容 dict
                    content = (jargon.get("content") if isinstance(jargon, dict) else "")
                    meaning = (jargon.get("meaning") if isinstance(jargon, dict) else None)

                content_s = (str(content or "").strip())
                if not content_s:
                    continue

                # 仅匹配有 meaning 的记录（保持原函数行为）
                if not str(meaning or "").strip():
                    continue

                # 跳过包含机器人昵称的词条（保持原函数行为）
                try:
                    from src.bw_learner.learner_utils import contains_bot_self_name

                    if contains_bot_self_name(content_s):
                        continue
                except Exception:
                    pass

                key_lower = content_s.lower()
                if not key_lower:
                    continue

                # 纯英文/数字：需要单词边界
                requires_boundary = not _contains_cjk(content_s)

                # 去重：同 key 出现多次时，保留更“长”的（更具体）
                prev = meta_map.get(key_lower)
                if prev is None or len(key_lower) > prev.length:
                    meta_map[key_lower] = _JargonMeta(
                        content=content_s,
                        length=len(key_lower),
                        requires_word_boundary=requires_boundary,
                    )

            # 将 meta_map 中的 key 写入自动机
            for k in meta_map.keys():
                try:
                    add_word(k, k)
                except Exception:
                    # 某些后端/输入异常时跳过该条，保证稳定性
                    continue

            finalize()

            built_at = time.time()
            with self._lock:
                if forced:
                    self.rebuilds += 1
                else:
                    self.builds += 1
                self._automatons[key] = (automaton, meta_map, built_at, backend, len(meta_map))

            dt = time.time() - t0
            logger.debug(
                f"[JargonMatcherAutomaton] 自动机构建完成 key={key} backend={backend} "
                f"jargons={len(meta_map)} dt={dt:.3f}s"
            )

            return automaton, meta_map
        except Exception as e:
            self.last_error = str(e)
            raise

    def _get_jargons(self, chat_id: str) -> List[Any]:
        """优先从 jargon_cache 获取，否则回退 DB 查询。"""

        # 1) 尝试从 cache_manager 获取 jargon_cache
        try:
            jargon_cache = None
            if self._cache_manager is not None and hasattr(self._cache_manager, "get_cache"):
                jargon_cache = self._cache_manager.get_cache("jargon_cache")

            if jargon_cache and hasattr(jargon_cache, "is_loaded") and jargon_cache.is_loaded():
                all_jargons = list(getattr(jargon_cache, "get_all")() or [])
                return self._filter_and_sort_jargons(all_jargons, chat_id)
        except Exception:
            pass

        # 2) DB 回退
        try:
            from src.common.database.database_model import Jargon
            from src.config.config import global_config

            query = Jargon.select().where((Jargon.meaning.is_null(False)) & (Jargon.meaning != ""))
            if getattr(getattr(global_config, "expression", None), "all_global_jargon", False):
                query = query.where(Jargon.is_global)
            query = query.order_by(Jargon.count.desc())
            return self._filter_and_sort_jargons(list(query), chat_id)
        except Exception as e:
            self.last_error = str(e)
            return []

    @staticmethod
    def _filter_and_sort_jargons(jargons: List[Any], chat_id: str) -> List[Any]:
        """复用 jargon_explainer 的过滤逻辑（尽量保持一致）。"""

        try:
            from src.config.config import global_config
            from src.bw_learner.learner_utils import (
                chat_id_list_contains,
                parse_chat_id_list,
            )

            all_global = bool(getattr(getattr(global_config, "expression", None), "all_global_jargon", False))
        except Exception:
            all_global = False
            chat_id_list_contains = None  # type: ignore[assignment]
            parse_chat_id_list = None  # type: ignore[assignment]

        out: List[Any] = []
        for j in jargons:
            try:
                meaning = getattr(j, "meaning", None)
                if not str(meaning or "").strip():
                    continue

                if all_global:
                    if not bool(getattr(j, "is_global", False)):
                        continue
                else:
                    if bool(getattr(j, "is_global", False)):
                        pass
                    else:
                        chat_id_raw = getattr(j, "chat_id", "")
                        if callable(parse_chat_id_list) and callable(chat_id_list_contains):
                            chat_id_list = parse_chat_id_list(chat_id_raw)
                            if not chat_id_list_contains(chat_id_list, chat_id):
                                continue
            except Exception:
                continue
            out.append(j)

        # 尽量按 count desc 排序（保持与原实现一致的优先级），无 count 时保持原顺序
        try:
            out.sort(key=lambda x: int(getattr(x, "count", 0)), reverse=True)
        except Exception:
            pass
        return out

    def get_memory_usage(self) -> int:
        """估算自动机管理器内存占用（字节）。"""

        try:
            with self._lock:
                snapshot = {
                    "automatons": {
                        k: {
                            "meta": v[1],
                            "built_at": v[2],
                            "backend": v[3],
                            "jargon_count": v[4],
                        }
                        for k, v in self._automatons.items()
                    },
                    "stats": {
                        "builds": self.builds,
                        "rebuilds": self.rebuilds,
                        "cache_hits": self.cache_hits,
                        "cache_misses": self.cache_misses,
                    },
                }
            return int(MemoryUtils.get_size(snapshot))
        except Exception:
            return 0


class JargonMatcherAutomatonModule:
    """黑话匹配自动机补丁模块。"""

    def __init__(self, cache_manager: Any):
        """初始化并探测后端能力。"""

        self.stats = ModuleStats("jargon_matcher_automaton")

        self._cache_manager = cache_manager
        self._lock = threading.RLock()
        self._patched = False

        self._has_pyaho = bool(_HAS_PYAHO)
        self._degraded = not self._has_pyaho

        self._automaton_manager = AutomatonManager(cache_manager)

        # 原始函数引用
        self._orig_match_messages: Optional[Callable[..., Any]] = None
        self._orig_match_text: Optional[Callable[..., Any]] = None

        # 统计
        self._calls_messages = 0
        self._calls_text = 0
        self._fallback_calls = 0

    @property
    def degraded(self) -> bool:
        """是否处于降级模式（无 pyahocorasick 时）。"""

        return bool(self._degraded)

    def rebuild_automaton(self, chat_id: str) -> None:
        """为指定 chat 强制重建自动机。"""

        self._automaton_manager.rebuild_automaton(chat_id)

    def apply_patch(self) -> None:
        """应用 monkey-patch。"""

        with self._lock:
            if self._patched:
                return

            try:
                from src.bw_learner import jargon_explainer
                from src.bw_learner.jargon_explainer import JargonExplainer

                self._orig_match_messages = getattr(JargonExplainer, "match_jargon_from_messages", None)
                if not callable(self._orig_match_messages):
                    raise AttributeError("JargonExplainer.match_jargon_from_messages 不存在或不可调用")

                self._orig_match_text = getattr(jargon_explainer, "match_jargon_from_text", None)
                if not callable(self._orig_match_text):
                    raise AttributeError("jargon_explainer.match_jargon_from_text 不存在或不可调用")

                module = self
                original_match_messages = self._orig_match_messages
                original_match_text = self._orig_match_text
                automaton_manager = self._automaton_manager

                def _boundary_ok(text: str, start: int, end: int) -> bool:
                    left_ok = True
                    right_ok = True

                    if start > 0:
                        left_ok = not _is_word_char(text[start - 1])
                    if end + 1 < len(text):
                        right_ok = not _is_word_char(text[end + 1])

                    return left_ok and right_ok

                def patched_match_from_messages(self_explainer: Any, messages: List[Any]) -> List[Dict[str, str]]:
                    """自动机加速版：从消息列表中匹配黑话（保持原过滤逻辑）。"""

                    # CPython GIL 保证整数 += 原子性，无需加锁
                    module._calls_messages += 1

                    try:
                        if not messages:
                            return []

                        # 复用原逻辑：收集消息文本（跳过机器人消息）
                        try:
                            from src.bw_learner.learner_utils import is_bot_message
                        except Exception:
                            is_bot_message = None  # type: ignore[assignment]

                        message_texts: List[str] = []
                        for msg in messages:
                            try:
                                if callable(is_bot_message) and is_bot_message(msg):
                                    continue
                            except Exception:
                                pass

                            msg_text = (
                                getattr(msg, "display_message", None)
                                or getattr(msg, "processed_plain_text", None)
                                or ""
                            )
                            msg_text = str(msg_text or "").strip()
                            if msg_text:
                                message_texts.append(msg_text)

                        if not message_texts:
                            return []

                        combined_text = " ".join(message_texts)
                        text_lower = combined_text.lower()

                        automaton, meta_map = automaton_manager.get_or_build(str(getattr(self_explainer, "chat_id", "")))

                        matched: Dict[str, None] = {}
                        for end_idx, key in automaton.iter(text_lower):
                            meta = meta_map.get(key)
                            if meta is None:
                                continue
                            start_idx = int(end_idx) - meta.length + 1
                            if start_idx < 0:
                                continue

                            if meta.requires_word_boundary:
                                if not _boundary_ok(text_lower, start_idx, int(end_idx)):
                                    continue

                            matched[meta.content] = None

                        _stats_hit(module.stats)
                        return [{"content": c} for c in matched.keys()]

                    except Exception:
                        module._fallback_calls += 1
                        _stats_miss(module.stats)
                        return list(original_match_messages(self_explainer, messages))

                def patched_match_from_text(chat_text: str, chat_id: str) -> List[str]:
                    """自动机加速版：从纯文本中匹配黑话（保持原过滤逻辑）。"""

                    # CPython GIL 保证整数 += 原子性，无需加锁
                    module._calls_text += 1

                    try:
                        if not chat_text or not str(chat_text).strip():
                            return []

                        text_lower = str(chat_text).lower()
                        automaton, meta_map = automaton_manager.get_or_build(str(chat_id))

                        matched: Dict[str, None] = {}
                        for end_idx, key in automaton.iter(text_lower):
                            meta = meta_map.get(key)
                            if meta is None:
                                continue
                            start_idx = int(end_idx) - meta.length + 1
                            if start_idx < 0:
                                continue

                            if meta.requires_word_boundary:
                                if not _boundary_ok(text_lower, start_idx, int(end_idx)):
                                    continue

                            matched[meta.content] = None

                        try:
                            from src.bw_learner.jargon_explainer import logger as j_logger

                            j_logger.info(f"匹配到 {len(matched)} 个黑话")
                        except Exception:
                            pass

                        _stats_hit(module.stats)
                        return list(matched.keys())

                    except Exception:
                        module._fallback_calls += 1
                        _stats_miss(module.stats)
                        return list(original_match_text(chat_text, chat_id))

                # patch 实例方法
                JargonExplainer.match_jargon_from_messages = patched_match_from_messages  # type: ignore[assignment]

                # patch 模块级函数
                jargon_explainer.match_jargon_from_text = patched_match_from_text  # type: ignore[assignment]

                self._patched = True
                logger.info(
                    "[JargonMatcherAutomaton] ✓ 补丁应用成功"
                    + ("（pyahocorasick 可用）" if self._has_pyaho else "（降级：纯 Python 自动机）")
                )

            except Exception as e:
                logger.error(f"[JargonMatcherAutomaton] ✗ 补丁失败: {e}")

    def remove_patch(self) -> None:
        """移除 monkey-patch 并恢复原始实现。"""

        with self._lock:
            if not self._patched:
                return

            try:
                from src.bw_learner import jargon_explainer
                from src.bw_learner.jargon_explainer import JargonExplainer

                if self._orig_match_messages is not None:
                    JargonExplainer.match_jargon_from_messages = self._orig_match_messages  # type: ignore[assignment]

                if self._orig_match_text is not None:
                    jargon_explainer.match_jargon_from_text = self._orig_match_text  # type: ignore[assignment]

                self._patched = False
                logger.info("[JargonMatcherAutomaton] 补丁已移除")
            except Exception as e:
                logger.error(f"[JargonMatcherAutomaton] 移除补丁失败: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """返回统计信息（包含累计和间隔统计字段）。"""

        # CPython GIL 保证整数读取原子性，无需加锁
        calls_messages = int(self._calls_messages)
        calls_text = int(self._calls_text)
        fallback_calls = int(self._fallback_calls)

        out: Dict[str, Any] = {
            "degraded": bool(self._degraded),
            "has_pyahocorasick": bool(self._has_pyaho),
            "backend": self._automaton_manager.backend,
            "calls_messages": calls_messages,
            "calls_text": calls_text,
            "fallback_calls": fallback_calls,
            "automaton_builds": int(self._automaton_manager.builds),
            "automaton_rebuilds": int(self._automaton_manager.rebuilds),
            "automaton_cache_hits": int(self._automaton_manager.cache_hits),
            "automaton_cache_misses": int(self._automaton_manager.cache_misses),
        }

        try:
            if hasattr(self.stats, "total"):
                total = self.stats.total()
                if isinstance(total, dict):
                    out.update(total)
        except Exception:
            pass

        try:
            if hasattr(self.stats, "reset_interval"):
                interval = self.stats.reset_interval()
                if isinstance(interval, dict):
                    out.update(interval)
        except Exception:
            pass

        if self._automaton_manager.last_error:
            out["last_error"] = str(self._automaton_manager.last_error)

        return out

    def get_memory_usage(self) -> int:
        """返回模块内存占用（字节，估算）。"""

        try:
            return int(
                MemoryUtils.get_size(
                    {
                        "automatons": self._automaton_manager.get_memory_usage(),
                        "orig": {
                            "match_messages": self._orig_match_messages,
                            "match_text": self._orig_match_text,
                        },
                        "stats": self.get_stats(),
                    }
                )
            )
        except Exception:
            return 0


def apply_jargon_matcher_automaton(cache_manager: Any) -> Optional[JargonMatcherAutomatonModule]:
    """工厂函数：创建模块、注册到 cache_manager 并应用补丁。"""

    try:
        mod = JargonMatcherAutomatonModule(cache_manager)
        cache_manager.register_cache("jargon_matcher_automaton", mod)
        mod.apply_patch()
        logger.info("[JargonMatcherAutomaton] ✓ 模块已初始化")
        return mod
    except Exception as e:
        logger.error(f"[JargonMatcherAutomaton] 初始化失败: {e}")
        return None
