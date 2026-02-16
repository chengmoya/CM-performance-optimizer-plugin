# CM-performance-optimizer-plugin 全模块启用测试计划

## 版本信息
- **插件版本**: v5.0.0
- **测试日期**: 2026-02-15
- **适用范围**: 全模块启用模式

---

## 1. 修复验证清单

### Bug 1: 停用时统一回滚 monkey-patch ✅
**位置**: [`plugin.py`](CM-performance-optimizer-plugin/plugin.py:878) `_PerformanceOptimizer.stop()`

**修复内容**:
```python
# BUG FIX: 统一回滚所有模块的 monkey-patch
for name, cache in list(self.cache_manager.caches.items()):
    try:
        if hasattr(cache, "remove_patch") and callable(getattr(cache, "remove_patch")):
            cache.remove_patch()
            self.logger.debug(f"[PerfOpt] 已回滚 {name} 的补丁")
    except Exception as e:
        self.logger.warning(f"[PerfOpt] 回滚 {name} 补丁失败: {e}")
```

**验证步骤**:
1. 启动 MaiBot，确认插件正常加载
2. 检查日志输出，确认所有模块补丁已应用
3. 停用插件（或重启 MaiBot）
4. 验证以下函数已恢复原实现：
   - `message_repository.find_messages`
   - `message_repository.count_messages`
   - `person_info.levenshtein_distance`
   - `MessageStorage.store_message`
5. 检查日志中是否有 `[PerfOpt] 已回滚 xxx 的补丁` 记录

**预期结果**: 所有被 patch 的函数在停用后应恢复为原始实现，无残留补丁。

---

### Bug 2: 全量缓存删除字段错误 ✅
**位置**: [`full_message_cache.py`](CM-performance-optimizer-plugin/full_message_cache.py:547) `FullMessageCache.remove_message()`

**修复内容**:
```python
# BUG FIX: 兼容 message_id 和 id 两种字段名
def _get_msg_id(m):
    """获取消息ID，兼容 message_id 和 id 两种字段名"""
    return getattr(m, "message_id", None) or getattr(m, "id", None)

messages[:] = [m for m in messages if _get_msg_id(m) != message_id]
```

**验证步骤**:
1. 启用全量消息缓存（如使用）
2. 发送若干条消息到聊天
3. 调用 `remove_message(chat_id, message_id)` 删除特定消息
4. 立即查询该 chat 的消息列表
5. 验证被删除的消息已从缓存中消失

**预期结果**: 删除操作应正确移除缓存中的对应条目，无论消息对象使用 `message_id` 还是 `id` 字段。

---

### Bug 3: 增量加载统计错误 ✅
**位置**: [`full_message_cache.py`](CM-performance-optimizer-plugin/full_message_cache.py:648) `FullMessageCache._incremental_load_chat()`

**修复内容**:
```python
# BUG FIX: 检查 chat 是否已存在，避免重复统计
chat_exists = chat_id in self.buffer_a
old_message_count = len(self.buffer_a.get(chat_id, {}).get("messages", []))

# ... 更新缓存 ...

# 更新统计（仅当 chat 不存在时才增加 _total_chats）
with self._stats_lock:
    # 减去旧消息数量，加上新消息数量
    self._total_messages += len(messages) - old_message_count
    if not chat_exists:
        self._total_chats += 1
```

**验证步骤**:
1. 清空缓存并重新加载
2. 对同一 chat_id 多次触发增量加载（模拟缓存未命中）
3. 观察统计信息中的 `_total_messages` 和 `_total_chats`

**预期结果**: 
- `_total_chats` 不应随同一 chat 的多次加载而增加
- `_total_messages` 应反映实际消息数变化，而非线性累加

---

### Bug 4: 缓存 key 边界条件 ✅
**位置**: [`components/modules/message_cache.py`](CM-performance-optimizer-plugin/components/modules/message_cache.py:876) `MessageCacheModule._make_cache_key()`

**修复内容**:
```python
# BUG FIX: 添加边界检查，处理 time_cond 为 {"$lt": None} 的情况
if ts_val is None:
    cacheable = False
    ts_float = 0.0
else:
    now = time.time()
    ts_float = float(ts_val)
```

**验证步骤**:
1. 构造包含 `time_cond = {"$lt": None}` 的查询
2. 调用 `_make_cache_key()` 生成缓存键
3. 验证函数不抛出异常且返回 `cacheable=False`

**预期结果**: 当 `time_cond` 包含 `None` 值时，函数应安全处理并标记为不可缓存，不触发 UnboundLocalError。

---

## 2. 功能测试用例

### 2.1 模块启动/停止测试

| 测试项 | 步骤 | 预期结果 |
|--------|------|----------|
| 正常启动 | 1. 配置全模块启用<br>2. 启动 MaiBot | 所有模块成功初始化，日志显示 ✓ |
| 热重载 | 1. 修改配置<br>2. 触发配置重载 | 配置变更生效，无异常 |
| 优雅停止 | 1. 运行中停止插件<br>2. 检查函数状态 | 所有 patch 已回滚，原函数恢复 |

### 2.2 缓存功能测试

| 测试项 | 步骤 | 预期结果 |
|--------|------|----------|
| 消息缓存命中 | 1. 首次查询（miss）<br>2. 相同条件再次查询 | 第二次命中缓存，响应更快 |
| 热集缓存 | 1. 激活聊天流<br>2. 滑动窗口查询 | 优先命中热集，不走 query-cache |
| 写入失效 | 1. 查询并缓存结果<br>2. 发送新消息<br>3. 再次查询 | 缓存自动失效，返回最新数据 |
| LRU淘汰 | 1. 填充缓存至上限<br>2. 继续添加新 chat | 最久未使用的 chat 被淘汰 |

### 2.3 性能测试

| 测试项 | 方法 | 指标 |
|--------|------|------|
| 查询延迟 | 对比 patch 前后的 find_messages 耗时 | P99 延迟降低 >= 30% |
| 内存占用 | 监控各缓存模块内存使用 | 不超过配置的阈值 |
| 命中率 | 统计 query-cache 和 hotset 命中率 | query-cache >= 60%, hotset >= 40% |

---

## 3. 回归测试

### 3.1 兼容性测试
- [ ] 与旧版本配置兼容
- [ ] 与其他插件共存
- [ ] 降级模式（rapidfuzz 不可用等）

### 3.2 边界条件测试
- [ ] 空数据库启动
- [ ] 超大消息量（10万+）
- [ ] 高频写入场景
- [ ] 网络中断恢复

---

## 4. 测试执行建议

### 4.1 环境准备
```bash
# 安装可选依赖
pip install aiofiles orjson psutil rapidfuzz

# 备份现有配置
cp config.toml config.toml.backup
```

### 4.2 自动化测试脚本
```python
# test_fixes.py 已包含基础验证
python CM-performance-optimizer-plugin/test_fixes.py
```

### 4.3 手动验证命令
```python
# 在 MaiBot 控制台执行
from CM_performance_optimizer_plugin.plugin import _PerformanceOptimizer
opt = _PerformanceOptimizer()

# 查看统计
print(opt.get_stats())

# 查看内存使用
print(opt.get_memory_usage())

# 检查是否降级
print(opt.plugin_instance.is_degraded())
```

---

## 5. 问题排查指南

### 5.1 常见日志分析
| 日志关键词 | 含义 | 处理建议 |
|-----------|------|----------|
| `✓ 补丁应用成功` | 模块正常加载 | 无需处理 |
| `⚠️ 链式 patch` | 多模块 patch 同一函数 | 检查模块加载顺序 |
| `降级模式` | 部分功能不可用 | 安装缺失依赖 |
| `回滚 xxx 补丁失败` | 停止时清理异常 | 检查模块实现 |

### 5.2 调试开关
```toml
[plugin]
log_level = "DEBUG"  # 开启详细日志
```

---

## 6. 验收标准

- [x] 所有 4 个 Bug 修复通过验证
- [ ] 全模块启用无异常启动
- [ ] 停用后所有 patch 正确回滚
- [ ] 缓存命中率符合预期
- [ ] 内存使用在合理范围
- [ ] 与 MaiBot 核心功能无冲突

---

**测试负责人**: _____________  
**测试日期**: _____________  
**测试结果**: ⬜ 通过 / ⬜ 有条件通过 / ⬜ 失败
