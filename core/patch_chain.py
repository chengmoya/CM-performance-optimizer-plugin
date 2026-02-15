"""PatchChain 管理器 - 解决多模块函数 monkey-patch 冲突。

当多个优化模块需要 patch 同一个函数时，后加载的模块会覆盖前者的 patch，
导致前者的优化逻辑被跳过。PatchChain 通过维护一个有序链表来跟踪每个函数
的所有 patch，确保：

1. 每个模块 patch 时能感知当前函数已被谁 patch（冲突检测）
2. 后 patch 的模块包装的是"最新版本"的函数（链式调用）
3. 取消 patch 时能正确移除自己的注册信息

冲突矩阵：
| 目标函数                   | 冲突模块                                              |
|----------------------------|-------------------------------------------------------|
| replace_user_references    | regex_precompile ↔ user_reference_batch_resolve       |
| replace_image_descriptions | regex_precompile ↔ image_desc_bulk_lookup             |
| store_message              | regex_precompile ↔ message_cache ↔ full_message_cache |
| find_messages              | message_cache ↔ full_message_cache                    |
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

try:
    from src.common.logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str):  # type: ignore[misc]
        return logging.getLogger(name)


logger = get_logger("CM_perf_opt")


class PatchChain:
    """管理函数 patch 链，确保多个模块可以链式 patch 同一函数。

    单例模式：全局只有一个 PatchChain 实例，所有模块共享。
    """

    _instance: Optional["PatchChain"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "PatchChain":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        # {函数标识: [(模块名, 原始函数, 补丁函数), ...]}
        self._chains: Dict[str, List[Tuple[str, Any, Any]]] = {}
        self._rlock = threading.RLock()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def register_patch(
        self,
        func_id: str,
        module_name: str,
        original_func: Any,
        patched_func: Any,
    ) -> Any:
        """注册一个 patch，返回应该包装的函数。

        返回值是调用者的 wrapper 应当委托的"上游"函数：
        - 如果 func_id 尚无任何 patch：返回 original_func
        - 如果已有其他模块 patch：返回链中最新的 patched_func（即上一个 patch）

        Args:
            func_id: 函数唯一标识，如 ``"replace_user_references"``。
            module_name: 当前模块名，如 ``"regex_precompile"``。
            original_func: 模块看到的"原始"函数引用。
            patched_func: 当前模块准备写入的 patched 版本。

        Returns:
            调用者应当包装（wrap/delegate）的目标函数。
        """
        with self._rlock:
            if func_id not in self._chains:
                self._chains[func_id] = []

            chain = self._chains[func_id]

            # 检测冲突
            for existing_module, _, _ in chain:
                if existing_module != module_name:
                    logger.warning(
                        f"[PatchChain] ⚠️ 检测到冲突: {func_id} 已被 "
                        f"{existing_module} patch，现在 {module_name} 也将 patch 此函数"
                    )

            # 获取当前应该包装的函数（链中最后一个 patch 的结果，或原始函数）
            wrapper_target = original_func
            if chain:
                _, _, last_patched = chain[-1]
                wrapper_target = last_patched

            chain.append((module_name, original_func, patched_func))
            logger.debug(
                f"[PatchChain] 注册: {func_id} <- {module_name} "
                f"(链长度={len(chain)})"
            )
            return wrapper_target

    def unregister_patch(self, func_id: str, module_name: str) -> None:
        """取消注册一个 patch。

        Args:
            func_id: 函数唯一标识。
            module_name: 要移除的模块名。
        """
        with self._rlock:
            if func_id not in self._chains:
                return
            before = len(self._chains[func_id])
            self._chains[func_id] = [
                (m, o, p)
                for m, o, p in self._chains[func_id]
                if m != module_name
            ]
            after = len(self._chains[func_id])
            if before != after:
                logger.debug(
                    f"[PatchChain] 取消注册: {func_id} x {module_name} "
                    f"(链长度={before}->{after})"
                )

    def get_chain(self, func_id: str) -> List[Tuple[str, Any, Any]]:
        """获取某个函数的完整 patch 链（只读副本）。"""
        with self._rlock:
            return list(self._chains.get(func_id, []))

    def summary(self) -> Dict[str, List[str]]:
        """返回所有已注册的 patch 摘要：{func_id: [module_name, ...]}。"""
        with self._rlock:
            return {
                fid: [m for m, _, _ in chain]
                for fid, chain in self._chains.items()
                if chain
            }


def get_patch_chain() -> PatchChain:
    """获取全局 PatchChain 单例的便捷函数。"""
    return PatchChain()
