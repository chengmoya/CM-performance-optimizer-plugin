"""
CM 性能优化插件

功能模块：
1. 消息缓存 (message_cache) - 缓存 find_messages 查询结果
2. 人物信息缓存 (person_cache) - 缓存人物信息查询
3. 表达式缓存 (expression_cache) - 双缓冲+缓慢加载+原子切换
4. 黑话缓存 (slang_cache) - 双缓冲+缓慢加载+原子切换+内容索引
5. 知识库图谱缓存 (kg_cache) - 双缓冲+缓慢加载+原子切换

安装：将目录放入 MaiBot/plugins/ 下，重启 MaiBot
依赖：无额外依赖
"""

from .plugin import CMPerformanceOptimizerPlugin

__all__ = ["CMPerformanceOptimizerPlugin"]
