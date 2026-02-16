"""asyncio_loop_pool 性能优化模块。

该模块用于通过 monkey-patch 将
[`EmbeddingStore._get_embeddings_batch_threaded()`](../src/chat/knowledge/embedding_store.py:167)
替换为“线程本地（thread-local）常驻 event loop”版本，避免在 embedding 工作线程中
为每次请求频繁创建/销毁 event loop。

重要：该模块 **默认关闭**（配置项 `performance.enable_asyncio_loop_pool = false`），
原因是 threading + asyncio 交互存在潜在风险。

实现要点：
- 每个线程持有一个常驻 event loop（thread-local）
- 需要时创建，复用时计数
- 若 loop 异常或处于不可用状态（closed/running/执行失败后疑似损坏），将失效并重建
- 不主动 close loop：线程结束时由系统清理（但在异常重建时会 best-effort close）

模块风格参考：[`levenshtein_fast.py`](CM-performance-optimizer-plugin/components/modules/levenshtein_fast.py:1)
"""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:
    from src.common.logger import get_logger
except ImportError:  # pragma: no cover
    import logging

    def get_logger(name: str):  # type: ignore[no-redef]
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


class _StatsAdapter:
    """StatsReporter 适配器：提供 total()/reset_interval() 两个方法。"""

    def __init__(self, module: "AsyncioLoopPoolModule"):
        self._module = module

    def total(self) -> Dict[str, Any]:
        return self._module.get_stats()

    def reset_interval(self) -> Dict[str, Any]:
        return self._module._reset_interval()


_thread_local = threading.local()


class AsyncioLoopPoolModule:
    """EmbeddingStore 多线程 embedding 的 asyncio loop 复用补丁模块。"""

    def __init__(self):
        """初始化统计与 patch 状态。"""

        self._lock = threading.RLock()
        self._patched = False

        self._orig_func: Optional[Callable[..., Any]] = None

        # 统计（total + interval）
        self._loops_created_total = 0
        self._loops_reused_total = 0
        self._loops_created_i = 0
        self._loops_reused_i = 0

        # StatsReporter 适配：保持插件现有 stats 汇总体系兼容
        self.stats = _StatsAdapter(self)

    def _inc_created(self) -> None:
        with self._lock:
            self._loops_created_total += 1
            self._loops_created_i += 1

    def _inc_reused(self) -> None:
        with self._lock:
            self._loops_reused_total += 1
            self._loops_reused_i += 1

    def _reset_interval(self) -> Dict[str, Any]:
        """返回并清零间隔统计（供 StatsReporter 使用）。"""

        with self._lock:
            created = int(self._loops_created_i)
            reused = int(self._loops_reused_i)
            self._loops_created_i = 0
            self._loops_reused_i = 0

        return {
            "i_loops_created": created,
            "i_loops_reused": reused,
        }

    def _invalidate_thread_local_loop(self) -> None:
        """失效当前线程的 thread-local loop（best-effort close）。"""

        loop = getattr(_thread_local, "loop", None)
        try:
            if loop is not None and hasattr(loop, "is_closed") and not loop.is_closed():
                try:
                    loop.close()
                except Exception:
                    pass
        finally:
            try:
                if hasattr(_thread_local, "loop"):
                    delattr(_thread_local, "loop")
            except Exception:
                pass

    def _get_thread_local_loop(self) -> asyncio.AbstractEventLoop:
        """获取当前线程的常驻 event loop。

        Returns:
            当前线程绑定的可用 event loop。
        """

        loop = getattr(_thread_local, "loop", None)

        need_new = False
        if loop is None:
            need_new = True
        else:
            try:
                if loop.is_closed():
                    need_new = True
            except Exception:
                need_new = True

        if need_new:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            _thread_local.loop = loop
            self._inc_created()
        else:
            # 确保当前线程 event loop 引用正确（可能被外部代码覆盖）
            try:
                asyncio.set_event_loop(loop)
            except Exception:
                # set_event_loop 失败说明 loop 可能不可用，重建
                self._invalidate_thread_local_loop()
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                _thread_local.loop = loop
                self._inc_created()

            self._inc_reused()

        # 检查 loop 是否处于可用状态
        try:
            if loop.is_running():
                # 理论上不应该发生；若发生，创建新的
                self._invalidate_thread_local_loop()
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                _thread_local.loop = loop
                self._inc_created()
        except Exception:
            # 保险起见，异常也重建
            self._invalidate_thread_local_loop()
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            _thread_local.loop = loop
            self._inc_created()

        return loop

    def apply_patch(self) -> None:
        """patch `EmbeddingStore._get_embeddings_batch_threaded`。"""

        with self._lock:
            if self._patched:
                return

            try:
                from src.chat.knowledge.embedding_store import EmbeddingStore

                self._orig_func = getattr(EmbeddingStore, "_get_embeddings_batch_threaded", None)
                if not callable(self._orig_func):
                    raise AttributeError("EmbeddingStore._get_embeddings_batch_threaded 不存在或不可调用")

                module = self
                original_func = self._orig_func

                def patched_get_embeddings_batch_threaded(
                    self_store: Any,
                    strs: Any,
                    chunk_size: int = 10,
                    max_workers: int = 10,
                    progress_callback: Optional[Callable[[int], Any]] = None,
                ):
                    """替换版本：使用 thread-local event loop。

                    说明：
                    - 逻辑保持与原始实现一致，只替换 event loop 的获取/重建方式。
                    """

                    # 尽量保持原逻辑：空输入直接返回
                    if not strs:
                        return []

                    # 延迟导入（与原始实现一致）
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    from src.chat.knowledge.embedding_store import logger  # 复用目标模块 logger

                    # 分块
                    chunks = []
                    for i in range(0, len(strs), int(chunk_size)):
                        chunk = strs[i : i + int(chunk_size)]
                        chunks.append((i, chunk))

                    results: Dict[int, Any] = {}

                    def process_chunk_with_pool(chunk_data):
                        """处理单个数据块的函数（线程内执行）。"""

                        start_idx, chunk_strs = chunk_data
                        chunk_results = []

                        # 为每个线程创建独立的LLMRequest实例（保持原实现）
                        from src.llm_models.utils_model import LLMRequest
                        from src.config.config import model_config

                        try:
                            llm = LLMRequest(
                                model_set=model_config.model_task_config.embedding,
                                request_type="embedding",
                            )

                            # 线程本地 loop（一次 chunk 获取一次，避免频繁 getattr）
                            loop = module._get_thread_local_loop()

                            for i, s in enumerate(chunk_strs):
                                try:
                                    try:
                                        embedding = loop.run_until_complete(llm.get_embedding(s))
                                    except Exception as run_exc:
                                        # loop 可能被异常打坏：失效并重建（可选重试一次）
                                        module._invalidate_thread_local_loop()
                                        loop = module._get_thread_local_loop()
                                        try:
                                            embedding = loop.run_until_complete(llm.get_embedding(s))
                                        except Exception:
                                            raise run_exc

                                    if embedding and len(embedding) > 0:
                                        chunk_results.append(
                                            (start_idx + i, s, embedding[0])
                                        )
                                    else:
                                        logger.error(f"获取嵌入失败: {s}")
                                        chunk_results.append((start_idx + i, s, []))

                                    if progress_callback:
                                        progress_callback(1)

                                except Exception as e:
                                    logger.error(f"获取嵌入时发生异常: {s}, 错误: {e}")
                                    chunk_results.append((start_idx + i, s, []))

                                    if progress_callback:
                                        progress_callback(1)

                        except Exception as e:
                            logger.error(f"创建LLM实例失败: {e}")
                            for i, s in enumerate(chunk_strs):
                                chunk_results.append((start_idx + i, s, []))
                                if progress_callback:
                                    progress_callback(1)

                        return chunk_results

                    # 使用线程池处理
                    with ThreadPoolExecutor(max_workers=int(max_workers)) as executor:
                        future_to_chunk = {
                            executor.submit(process_chunk_with_pool, chunk): chunk
                            for chunk in chunks
                        }

                        for future in as_completed(future_to_chunk):
                            try:
                                chunk_results = future.result()
                                for idx, s, embedding in chunk_results:
                                    results[int(idx)] = (s, embedding)
                            except Exception as e:
                                chunk = future_to_chunk[future]
                                logger.error(f"处理数据块时发生异常: {chunk}, 错误: {e}")
                                start_idx, chunk_strs = chunk
                                for i, s in enumerate(chunk_strs):
                                    results[int(start_idx) + int(i)] = (s, [])

                    ordered_results = []
                    for i in range(len(strs)):
                        if i in results:
                            ordered_results.append(results[i])
                        else:
                            ordered_results.append((strs[i], []))

                    return ordered_results

                # 将 patched 函数挂到类上
                EmbeddingStore._get_embeddings_batch_threaded = patched_get_embeddings_batch_threaded  # type: ignore[assignment]

                self._patched = True
                logger.info("[AsyncioLoopPool] ✓ 补丁应用成功")

            except Exception as e:
                logger.error(f"[AsyncioLoopPool] ✗ 补丁失败: {e}")
                # 失败时尽量恢复
                try:
                    if self._orig_func is not None:
                        from src.chat.knowledge.embedding_store import EmbeddingStore

                        EmbeddingStore._get_embeddings_batch_threaded = self._orig_func  # type: ignore[assignment]
                except Exception:
                    pass

    def remove_patch(self) -> None:
        """恢复原始 `_get_embeddings_batch_threaded`。"""

        with self._lock:
            if not self._patched:
                return

            try:
                from src.chat.knowledge.embedding_store import EmbeddingStore

                if self._orig_func is not None:
                    EmbeddingStore._get_embeddings_batch_threaded = self._orig_func  # type: ignore[assignment]

                self._patched = False
                logger.info("[AsyncioLoopPool] 补丁已移除")
            except Exception as e:
                logger.error(f"[AsyncioLoopPool] 移除补丁失败: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """返回统计信息（复用次数、创建次数）。"""

        with self._lock:
            created = int(self._loops_created_total)
            reused = int(self._loops_reused_total)
            patched = bool(self._patched)

        return {
            "patched": patched,
            "loops_created": created,
            "loops_reused": reused,
        }

    def get_memory_usage(self) -> int:
        """返回内存占用（该模块按要求返回 0）。"""

        return 0


def apply_asyncio_loop_pool(cache_manager: Any) -> Optional[AsyncioLoopPoolModule]:
    """工厂函数：创建模块、注册到 cache_manager 并应用补丁。"""

    try:
        mod = AsyncioLoopPoolModule()
        cache_manager.register_cache("asyncio_loop_pool", mod)
        mod.apply_patch()
        logger.info("[AsyncioLoopPool] ✓ 模块已初始化")
        return mod
    except Exception as e:
        logger.error(f"[AsyncioLoopPool] 初始化失败: {e}")
        return None
