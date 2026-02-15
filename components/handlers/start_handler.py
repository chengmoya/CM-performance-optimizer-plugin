"""
CM 性能优化插件 - 启动事件处理器

在插件启动时应用补丁
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
        ON_START = "on_start"
    
    def get_logger(name):
        import logging
        return logging.getLogger(name)

logger = get_logger("CM_perf_opt")


class PerfOptStartHandler(BaseEventHandler):
    """性能优化启动事件处理器 - 在插件启动时应用补丁"""

    # 事件处理器配置 - 使用 ON_START 事件
    event_type = EventType.ON_START
    handler_name = "perf_opt_start_handler"
    handler_description = "插件启动时应用补丁"
    weight = 10  # 高优先级，确保在其他处理器之前执行
    intercept_message = False
    
    def __init__(self, plugin_instance=None):
        super().__init__()
        self.plugin_instance = plugin_instance
    
    async def execute(self, message):
        """
        处理启动事件 - 应用补丁并启动优化器
        
        Args:
            message: MaiMessages 对象（ON_START 时为 None）
            
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

        if plugin_instance is not None:
            try:
                await plugin_instance._apply_patches_and_start()
            except Exception as e:
                logger.error(f"[PerfOpt] 启动失败: {e}")
                # 不阻止插件加载，记录错误并继续
                logger.warning("[PerfOpt] 插件将以降级模式运行")
        else:
            logger.warning("[PerfOpt] 无法获取插件实例，跳过启动")
        return (True, True, None, None, None)
