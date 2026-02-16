# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [5.2.0] - 2026-02-16

### Changed
- 统一版本号到 5.2.0（包括配置系统版本号）

## [5.1.0] - 2026-02-15

### Changed
- 统一版本号到 5.1.0
- 清理遗留注释和调试代码
- 优化代码结构和可读性
- 完善类型注解和文档

## [5.0.0] - 2026-02-15

### Added
- 表达式缓存模块（expression_cache）
- 拼写错误生成器缓存（typo_generator_cache）
- Levenshtein 距离加速模块（levenshtein_fast）
- 轻量级性能分析器（lightweight_profiler）
- 异步循环池（asyncio_loop_pool）
- 数据库调优模块（db_tuning）
- 正则预编译模块（regex_precompile）
- 用户引用批量解析（user_reference_batch_resolve）
- 消息仓库快速路径（message_repository_fastpath）
- 图片描述批量查询（image_desc_bulk_lookup）

### Changed
- 重构核心架构，模块化设计
- 优化配置系统，支持热更新
- 完善类型注解和错误处理

### Dependencies
- 添加 json-repair>=0.7.0（可选）

## [4.7.0] - 2026-02-01

### Changed
- 黑话缓存支持 Aho-Corasick 算法优化
- 几万条黑话也能毫秒级响应
- 性能提升 100 倍以上

## [4.5.0] - 2026-02-03

### Added
- 全量消息缓存模式

## [4.3.1] - 2026-01-31

### Changed
- 优化内存统计功能
- 更新文档

## [4.2.0] - 2026-01-31

### Added
- 知识库图谱缓存功能（kg_cache）

## [3.0.0] - 2026-01-31

### Added
- 消息缓存模块（message_cache）
- 人物信息缓存（person_cache）
- 黑话缓存（jargon_cache）

---

[5.0.0]: https://github.com/chengmoya/CM-performance-optimizer-plugin/compare/v4.7.0...v5.0.0
[4.7.0]: https://github.com/chengmoya/CM-performance-optimizer-plugin/compare/v4.5.0...v4.7.0
[4.5.0]: https://github.com/chengmoya/CM-performance-optimizer-plugin/compare/v4.3.1...v4.5.0
[4.3.1]: https://github.com/chengmoya/CM-performance-optimizer-plugin/compare/v4.2.0...v4.3.1
[4.2.0]: https://github.com/chengmoya/CM-performance-optimizer-plugin/compare/v3.0.0...v4.2.0
[3.0.0]: https://github.com/chengmoya/CM-performance-optimizer-plugin/releases/tag/v3.0.0
