"""
核心工具模块 - 统计类和辅助函数
"""

import threading
from typing import Any, Dict


class ModuleStats:
    """单个模块的统计。

    设计说明：
    - 命中/未命中（hit/miss）：只统计"可缓存"且进入缓存判定的请求。
    - 跳过（skipped）：不可缓存/主动跳过缓存的请求（例如高基数范围查询、limit 过大等）。
    - 过滤（filtered）：命中但因为一致性校验等原因未采用缓存结果。
    - 未选择（unselected）：由于策略/分流等原因，本次请求没有走该模块的缓存路径。

    这样报告中的"命中率"不会被大量跳过请求拉低，同时能区分"跳过"与"未选择"。

    统计字段约定：
    - t_*：累计值（total）
    - i_*：本统计间隔内值（interval），通过 reset_interval() 获取并清零
    """

    def __init__(self, name: str):
        self.name = str(name)
        self.lock = threading.Lock()

        # 累计
        self.t_hit = 0
        self.t_miss = 0
        self.t_filtered = 0
        self.t_skipped = 0
        self.t_unselected = 0
        self.t_fast = 0
        self.t_slow = 0
        self.t_fast_time = 0.0
        self.t_slow_time = 0.0

        # 间隔
        self.i_hit = 0
        self.i_miss = 0
        self.i_filtered = 0
        self.i_skipped = 0
        self.i_unselected = 0
        self.i_fast = 0
        self.i_slow = 0
        self.i_fast_time = 0.0
        self.i_slow_time = 0.0

    def hit(self) -> None:
        """记录缓存命中。"""
        with self.lock:
            self.t_hit += 1
            self.i_hit += 1

    def miss(self, elapsed: float) -> None:
        """记录缓存未命中，并按耗时归类快/慢。"""
        with self.lock:
            self.t_miss += 1
            self.i_miss += 1

            if float(elapsed) > 0.1:
                self.t_slow += 1
                self.i_slow += 1
                self.t_slow_time += float(elapsed)
                self.i_slow_time += float(elapsed)
            else:
                self.t_fast += 1
                self.i_fast += 1
                self.t_fast_time += float(elapsed)
                self.i_fast_time += float(elapsed)

    def skipped(self) -> None:
        """记录不可缓存/被跳过的情况（不进入命中率分母）。"""
        with self.lock:
            self.t_skipped += 1
            self.i_skipped += 1

    def filtered(self) -> None:
        """记录命中但缓存结果未采用的情况（如一致性校验失败）。"""
        with self.lock:
            self.t_filtered += 1
            self.i_filtered += 1

    def unselected(self) -> None:
        """记录未选择该模块缓存路径的情况。

        典型场景：
        - 同一功能存在多级缓存/分流策略（例如优先走 hotset，未走 query-cache）。
        - 满足 cacheable，但由于策略（例如 limit 太大）选择不写入缓存。
        """
        with self.lock:
            self.t_unselected += 1
            self.i_unselected += 1

    def reset_interval(self) -> Dict[str, Any]:
        """获取并清空间隔统计。"""
        with self.lock:
            r: Dict[str, Any] = {
                "i_hit": self.i_hit,
                "i_miss": self.i_miss,
                "i_filtered": self.i_filtered,
                "i_skipped": self.i_skipped,
                "i_unselected": self.i_unselected,
                "i_fast": self.i_fast,
                "i_slow": self.i_slow,
                "i_fast_time": self.i_fast_time,
                "i_slow_time": self.i_slow_time,
            }

            self.i_hit = 0
            self.i_miss = 0
            self.i_filtered = 0
            self.i_skipped = 0
            self.i_unselected = 0
            self.i_fast = 0
            self.i_slow = 0
            self.i_fast_time = 0.0
            self.i_slow_time = 0.0
            return r

    def total(self) -> Dict[str, Any]:
        """获取累计统计。"""
        with self.lock:
            return {
                "t_hit": self.t_hit,
                "t_miss": self.t_miss,
                "t_filtered": self.t_filtered,
                "t_skipped": self.t_skipped,
                "t_unselected": self.t_unselected,
                "t_fast": self.t_fast,
                "t_slow": self.t_slow,
                "t_fast_time": self.t_fast_time,
                "t_slow_time": self.t_slow_time,
            }


def rate(hit, miss, filtered=0):
    t = hit + miss + filtered
    return (hit / t * 100) if t > 0 else 0


class ChatVersionTracker:
    """写入侧递增版本号。

    用于让"按 chat_id 拉最近上下文"的缓存键稳定命中，且在新消息写入后自动失效。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._ver: Dict[str, int] = {}

    def get(self, chat_id: str) -> int:
        with self._lock:
            return int(self._ver.get(chat_id, 0))

    def bump(self, chat_id: str) -> int:
        with self._lock:
            v = int(self._ver.get(chat_id, 0)) + 1
            self._ver[chat_id] = v
            return v