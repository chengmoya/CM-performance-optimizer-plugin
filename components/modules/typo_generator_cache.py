"""ChineseTypoGenerator 启动/运行时瓶颈优化：pinyin_dict 持久化缓存 + jieba 词典内存缓存。

该模块通过 monkey-patch 对 [`src.chat.utils.typo_generator.ChineseTypoGenerator`](../src/chat/utils/typo_generator.py:21)
做两处优化：

1) `_create_pinyin_dict()`：首次计算后持久化到 `depends-data/pinyin_dict_cache.json`，后续启动直接加载。
2) `_get_word_homophones()`：jieba 词典解析结果 `valid_words` 缓存为类属性，避免每次调用重读 35 万行。

注意：不修改 MaiBot 源码，仅在运行期替换方法实现；支持安全 revert。
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
        logger.warning(f"[TypoGeneratorCache] 无法加载核心模块，使用内置实现: {e}")

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


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return None
    except Exception:
        return None


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def _get_pypinyin_version_tag() -> str:
    """获取 pypinyin 版本标记，用于缓存失效。"""

    try:
        import pypinyin  # type: ignore

        ver = getattr(pypinyin, "__version__", None)
        ver_str = str(ver) if ver else "unknown"
        return f"pypinyin_{ver_str}"
    except Exception:
        return "pypinyin_unknown"


class TypoGeneratorCacheModule:
    """typo_generator 的缓存优化补丁模块。"""

    def __init__(self):
        self.stats = ModuleStats("typo_generator_cache")

        self._lock = threading.RLock()
        self._patched = False
        self._degraded = False
        self._degraded_reason: Optional[str] = None

        self._original_create_pinyin_dict: Optional[Callable[[], Any]] = None
        self._original_get_word_homophones: Optional[Callable[[Any, str], Any]] = None

        # 统计：pinyin_dict 持久化缓存
        self._pinyin_cache_hit = 0
        self._pinyin_cache_miss = 0
        self._pinyin_cache_rebuild = 0
        self._pinyin_cache_save_errors = 0
        self._pinyin_cache_load_errors = 0

        # 统计：jieba valid_words 内存缓存
        self._valid_words_cache_hit = 0
        self._valid_words_cache_miss = 0
        self._valid_words_cache_load_errors = 0

        # 内存缓存：pinyin_dict 加载后缓存，避免重复磁盘读取
        self._pinyin_dict: Optional[Any] = None

        # 缓存路径（与 MaiBot 自身依赖数据目录保持一致）
        self._pinyin_cache_path = Path("depends-data") / "pinyin_dict_cache.json"

    def apply_patch(self) -> None:
        """应用 monkey-patch。"""

        with self._lock:
            if self._patched:
                return

            try:
                from src.chat.utils import typo_generator as tg

                ChineseTypoGenerator = getattr(tg, "ChineseTypoGenerator", None)
                if ChineseTypoGenerator is None:
                    raise AttributeError("src.chat.utils.typo_generator.ChineseTypoGenerator 不存在")

            except Exception as e:
                self._degraded = True
                self._degraded_reason = f"import_failed: {e}"
                logger.warning(f"[TypoGeneratorCache] 降级：无法导入 typo_generator: {e}")
                return

            # 备份原始实现
            try:
                self._original_create_pinyin_dict = getattr(
                    ChineseTypoGenerator, "_create_pinyin_dict", None
                )
                self._original_get_word_homophones = getattr(
                    ChineseTypoGenerator, "_get_word_homophones", None
                )

                if not callable(self._original_create_pinyin_dict):
                    raise AttributeError("ChineseTypoGenerator._create_pinyin_dict 不存在或不可调用")
                if not callable(self._original_get_word_homophones):
                    raise AttributeError("ChineseTypoGenerator._get_word_homophones 不存在或不可调用")

                module = self
                original_create = self._original_create_pinyin_dict

                def cached_create_pinyin_dict() -> Any:
                    """带持久化缓存的 `_create_pinyin_dict()`。

                    优化：文件 IO 在锁外执行，使用双重检查锁定避免重复加载。
                    """

                    # 第一次��查（无锁）：内存缓存命中则直接返回
                    if module._pinyin_dict is not None:
                        with module._lock:
                            module._pinyin_cache_hit += 1
                        try:
                            module.stats.hit()
                        except Exception:
                            pass
                        return module._pinyin_dict

                    version_tag = _get_pypinyin_version_tag()
                    cache_path = module._pinyin_cache_path

                    # 在锁外执行文件 IO，避免阻塞其他线程
                    file_exists = cache_path.exists()
                    cached_file_data: Optional[Dict[str, Any]] = None
                    if file_exists:
                        cached_file_data = _safe_read_json(cache_path)

                    # 锁内处理缓存数据、更新统计
                    with module._lock:
                        # 双重检查：另一个线程可能已完成加载
                        if module._pinyin_dict is not None:
                            module._pinyin_cache_hit += 1
                            try:
                                module.stats.hit()
                            except Exception:
                                pass
                            return module._pinyin_dict

                        # 处理从磁盘读取的缓存数据
                        if file_exists and cached_file_data is not None:
                            try:
                                cached_ver = str(cached_file_data.get("version", ""))
                                cached_data = cached_file_data.get("data")
                                if (
                                    cached_ver == version_tag
                                    and isinstance(cached_data, dict)
                                    and cached_data
                                ):
                                    module._pinyin_cache_hit += 1
                                    # 命中时也记入 ModuleStats.hit 方便统一报表
                                    try:
                                        module.stats.hit()
                                    except Exception:
                                        pass

                                    dd: "defaultdict[str, List[str]]" = defaultdict(list)
                                    for k, v in cached_data.items():
                                        if isinstance(k, str) and isinstance(v, list):
                                            dd[k] = [str(x) for x in v]
                                    # 存入内存缓存
                                    module._pinyin_dict = dd
                                    return dd

                                # 版本不一致/结构异常 → 重建
                                module._pinyin_cache_miss += 1
                                module._pinyin_cache_rebuild += 1
                            except Exception:
                                module._pinyin_cache_load_errors += 1
                                module._pinyin_cache_miss += 1
                        elif file_exists:
                            # 文件存在但读取失败
                            module._pinyin_cache_load_errors += 1
                            module._pinyin_cache_miss += 1
                        else:
                            module._pinyin_cache_miss += 1

                        # miss 记入 ModuleStats.miss
                        try:
                            module.stats.miss(0.0)
                        except TypeError:
                            try:
                                module.stats.miss()
                            except Exception:
                                pass
                        except Exception:
                            pass

                    # 2) 计算（不把长计算锁在 module._lock 内，避免阻塞其他路径）
                    result = original_create()

                    # 3) 保存
                    try:
                        payload: Dict[str, Any] = {
                            "version": version_tag,
                            "data": dict(result) if result is not None else {},
                        }
                        _atomic_write_json(module._pinyin_cache_path, payload)
                    except Exception as e:
                        with module._lock:
                            module._pinyin_cache_save_errors += 1
                        logger.debug(f"[TypoGeneratorCache] pinyin_dict 缓存写入失败: {e}")

                    # 重要：确保返回值具备 defaultdict(list) 行为，并存入内存缓存
                    try:
                        if isinstance(result, defaultdict):
                            with module._lock:
                                module._pinyin_dict = result
                            return result
                        dd2: "defaultdict[str, List[str]]" = defaultdict(list)
                        if isinstance(result, dict):
                            for k, v in result.items():
                                if isinstance(k, str) and isinstance(v, list):
                                    dd2[k] = [str(x) for x in v]
                        with module._lock:
                            module._pinyin_dict = dd2
                        return dd2
                    except Exception:
                        # 兜底：直接返回原结果
                        return result

                original_get_word_homophones = self._original_get_word_homophones

                def cached_get_word_homophones(self_obj: Any, word: str):
                    """带 valid_words 类属性缓存的 `_get_word_homophones()`。

                    目标：保持原始逻辑不变，仅把 jieba dict.txt 的解析结果缓存。
                    """

                    # 与原始逻辑保持一致
                    if len(word) == 1:
                        return []

                    # 获取词的拼音
                    word_pinyin = self_obj._get_word_pinyin(word)

                    # 遍历所有可能的同音字组合
                    candidates: List[List[str]] = []
                    for py in word_pinyin:
                        chars = self_obj.pinyin_dict.get(py, [])
                        if not chars:
                            return []
                        candidates.append(chars)

                    # 生成所有可能的组合
                    import itertools

                    all_combinations = itertools.product(*candidates)

                    # 获取/构建 valid_words 缓存（类属性）
                    cls = type(self_obj)
                    cached_valid_words = getattr(cls, "_cached_valid_words", None)
                    if isinstance(cached_valid_words, dict) and cached_valid_words:
                        with module._lock:
                            module._valid_words_cache_hit += 1
                        valid_words = cached_valid_words
                    else:
                        with module._lock:
                            module._valid_words_cache_miss += 1

                        try:
                            import jieba  # type: ignore

                            dict_path = os.path.join(os.path.dirname(jieba.__file__), "dict.txt")
                            vw: Dict[str, float] = {}
                            with open(dict_path, "r", encoding="utf-8") as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if len(parts) >= 2:
                                        word_text = parts[0]
                                        try:
                                            word_freq = float(parts[1])
                                        except Exception:
                                            continue
                                        vw[word_text] = word_freq

                            setattr(cls, "_cached_valid_words", vw)
                            valid_words = vw
                        except Exception as e:
                            with module._lock:
                                module._valid_words_cache_load_errors += 1
                            logger.debug(
                                f"[TypoGeneratorCache] valid_words 构建失败，回退原始实现: {e}"
                            )
                            # 回退原始实现，确保行为稳定
                            return original_get_word_homophones(self_obj, word)

                    # 获取原词的词频作为参考
                    original_word_freq = valid_words.get(word, 0)
                    min_word_freq = original_word_freq * 0.1

                    # 过滤和计算频率
                    homophones: List[Tuple[str, float]] = []
                    for combo in all_combinations:
                        new_word = "".join(combo)
                        if new_word != word and new_word in valid_words:
                            new_word_freq = valid_words[new_word]
                            if new_word_freq >= min_word_freq:
                                char_avg_freq = (
                                    sum(self_obj.char_frequency.get(c, 0) for c in new_word)
                                    / len(new_word)
                                )
                                combined_score = new_word_freq * 0.7 + char_avg_freq * 0.3
                                if combined_score >= self_obj.min_freq:
                                    homophones.append((new_word, combined_score))

                    sorted_homophones = sorted(homophones, key=lambda x: x[1], reverse=True)
                    return [w for w, _ in sorted_homophones[:5]]

                # patch
                ChineseTypoGenerator._create_pinyin_dict = staticmethod(  # type: ignore[assignment]
                    cached_create_pinyin_dict
                )
                ChineseTypoGenerator._get_word_homophones = cached_get_word_homophones  # type: ignore[assignment]

                self._patched = True
                logger.info("[TypoGeneratorCache] ✓ 补丁已应用")

            except Exception as e:
                logger.error(f"[TypoGeneratorCache] ✗ 补丁失败: {e}")
                # 失败时尽量恢复
                try:
                    self.revert_patch()
                except Exception:
                    pass

    def revert_patch(self) -> None:
        """恢复原始函数。"""

        with self._lock:
            if not self._patched:
                return

            try:
                from src.chat.utils import typo_generator as tg

                ChineseTypoGenerator = getattr(tg, "ChineseTypoGenerator", None)
                if ChineseTypoGenerator is None:
                    raise AttributeError("src.chat.utils.typo_generator.ChineseTypoGenerator 不存在")

                if self._original_create_pinyin_dict is not None:
                    ChineseTypoGenerator._create_pinyin_dict = staticmethod(  # type: ignore[assignment]
                        self._original_create_pinyin_dict
                    )

                if self._original_get_word_homophones is not None:
                    ChineseTypoGenerator._get_word_homophones = self._original_get_word_homophones  # type: ignore[assignment]

                self._patched = False
                self._pinyin_dict = None  # 清理内存缓存
                logger.info("[TypoGeneratorCache] 补丁已移除")
            except Exception as e:
                logger.error(f"[TypoGeneratorCache] 移除补丁失败: {e}")

    def is_degraded(self) -> bool:
        """是否降级（无法导入/无法 patch）。"""

        return bool(self._degraded)

    def get_memory_usage(self) -> int:
        """返回模块内存占用（字节）。"""

        try:
            return int(
                MemoryUtils.get_size(
                    {
                        "patched": self._patched,
                        "degraded": self._degraded,
                        "pinyin_cache_path": str(self._pinyin_cache_path),
                        "pinyin_stats": {
                            "hit": self._pinyin_cache_hit,
                            "miss": self._pinyin_cache_miss,
                            "rebuild": self._pinyin_cache_rebuild,
                            "save_errors": self._pinyin_cache_save_errors,
                            "load_errors": self._pinyin_cache_load_errors,
                        },
                        "valid_words_stats": {
                            "hit": self._valid_words_cache_hit,
                            "miss": self._valid_words_cache_miss,
                            "load_errors": self._valid_words_cache_load_errors,
                        },
                    }
                )
            )
        except Exception:
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """返回统计信息。"""

        out: Dict[str, Any] = {
            "patched": bool(self._patched),
            "degraded": bool(self._degraded),
            "degraded_reason": self._degraded_reason,
            "pinyin_cache": {
                "path": str(self._pinyin_cache_path),
                "hit": int(self._pinyin_cache_hit),
                "miss": int(self._pinyin_cache_miss),
                "rebuild": int(self._pinyin_cache_rebuild),
                "save_errors": int(self._pinyin_cache_save_errors),
                "load_errors": int(self._pinyin_cache_load_errors),
                "version_tag": _get_pypinyin_version_tag(),
            },
            "valid_words_cache": {
                "hit": int(self._valid_words_cache_hit),
                "miss": int(self._valid_words_cache_miss),
                "load_errors": int(self._valid_words_cache_load_errors),
            },
        }

        try:
            if hasattr(self.stats, "total"):
                total = self.stats.total()
                if isinstance(total, dict):
                    out.update(total)
        except Exception:
            pass

        return out


def apply_typo_generator_cache(cache_manager) -> Optional[TypoGeneratorCacheModule]:
    """工厂函数：创建模块、注册到 cache_manager 并应用补丁。"""

    try:
        mod = TypoGeneratorCacheModule()
        cache_manager.register_cache("typo_generator_cache", mod)
        mod.apply_patch()
        logger.info("[TypoGeneratorCache] ✓ 模块已初始化")
        return mod
    except Exception as e:
        logger.error(f"[TypoGeneratorCache] 初始化失败: {e}")
        return None
