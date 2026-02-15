"""
CM 性能优化插件 - 停止事件处理器

确保插件停止时正确回滚补丁
"""

from typing import Optional, Tuple

try:
    from src.plugin_system.base.base_events_handler import BaseEventHandler
    from src.plugin_system.base.component_types import EventType
    from src.common.logger import get_logger
except ImportError:
    # Fallback for standalone testing
    class BaseEventHandler:
        def __init__(self):
            pass
    
    class EventType:
        ON_STOP = "on_stop"
    
    def get_logger(name):
        import logging
        return logging.getLogger(name)

logger = get_logger("CM_perf_opt")


class PerfOptStopHandler(BaseEventHandler):
    """性能优化停止事件处理器 - 确保插件停止时正确回滚补丁"""

    # 事件处理器配置 - 使用 ON_STOP 事件
    event_type = EventType.ON_STOP
    handler_name = "perf_opt_stop_handler"
    handler_description = "插件停止时回滚所有补丁"
    weight = 10  # 高优先级，确保在其他处理器之前执行
    intercept_message = False
    
    def __init__(self, plugin_instance=None):
        super().__init__()
        self.plugin_instance = plugin_instance
    
    async def execute(self, message):
        """
        处理停止事件 - 调用 Optimizer.stop() 回滚补丁
        
        Args:
            message: MaiMessages 对象（ON_STOP 时为 None）
            
        Returns:
            Tuple[bool, bool, Optional[str], Optional[CustomEventHandlerResult], Optional[MaiMessages]]
        """
        # 优先尝试获取注入的实例（动态加载模块后由插件主类写入）
        plugin_instance = self.plugin_instance
        if plugin_instance is None:
            plugin_instance = globals().get("_plugin_instance")

        # 回退逻辑：尝试从插件模块导入（在 spec_from_file_location 场景下可能失效）
        if plugin_instance is None:
            try:
                from ..plugin import _plugin_instance as imported_plugin_instance

                plugin_instance = imported_plugin_instance
            except ImportError:
                plugin_instance = None

        self.plugin_instance = plugin_instance

        if plugin_instance is not None and hasattr(plugin_instance, "_opt"):
            opt = plugin_instance._opt
            if opt:
                try:
                    opt.stop()
                    logger.info("[PerfOpt] ✓ 插件已停止，补丁已回滚")
                except Exception as e:
                    logger.error(f"[PerfOpt] 停止失败: {e}")
        return (True, True, None, None, None)
