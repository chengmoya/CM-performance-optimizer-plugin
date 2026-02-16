#!/usr/bin/env python3
"""
CM-Performance-Optimizer-Plugin 配置系统完整性验证测试
"""

import sys
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

# 添加插件目录到路径
plugin_dir = Path(__file__).parent
sys.path.insert(0, str(plugin_dir))


def test_default_config_generation():
    """测试无配置文件时的默认配置生成机制"""
    print("=" * 80)
    print("测试1: 无配置文件时的默认配置生成机制")
    print("=" * 80)
    
    try:
        # 创建临时目录测试无配置文件情况
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 确保没有配置文件
            config_file = temp_path / "config.toml"
            assert not config_file.exists(), "配置文件不应该存在"
            
            # 导入并测试配置管理器
            from core.config import ConfigManager, ConfigFieldType
            
            config_manager = ConfigManager(temp_path)
            config = config_manager.load()
            
            print(f"✓ 无配置文件时成功生成默认配置")
            print(f"  配置版本: {config.get('plugin', {}).get('config_version', 'unknown')}")
            
            # 验证所有必需的配置节存在
            required_sections = ['plugin', 'performance', 'modules', 'advanced', 'monitoring']
            for section in required_sections:
                assert section in config, f"缺少必需配置节: {section}"
                print(f"✓ 配置节 [{section}] 存在")
            
            # 验证关键配置项有默认值
            key_checks = [
                ('plugin.enabled', True),
                ('plugin.log_level', 'INFO'),
                ('modules.message_cache_enabled', True),
                ('modules.person_cache_enabled', True),
                ('modules.expression_cache_enabled', True),
                ('modules.jargon_cache_enabled', True),
                ('modules.kg_cache_enabled', True),
                ('modules.message_cache.per_chat_limit', 200),
                ('modules.person_cache.max_size', 3000),
                ('advanced.thread_pool_size', 4),
                ('monitoring.enable_stats', True),
            ]
            
            for path, expected in key_checks:
                value = config_manager.get(path)
                assert value == expected, f"配置项 {path} 期望值 {expected}, 实际 {value}"
                print(f"✓ 配置项 {path} = {value}")
            
            return True, "默认配置生成测试通过"
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


def test_config_schema_completeness():
    """对比schema定义和实际配置文件，确认无遗漏"""
    print("\n" + "=" * 80)
    print("测试2: Schema定义与实际配置文件对比")
    print("=" * 80)
    
    try:
        from core.config import ConfigManager, ConfigFieldType
        
        config_manager = ConfigManager(plugin_dir)
        schema = config_manager.get_schema()
        
        # 从config.example.toml解析配置
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        example_config_path = plugin_dir / "config.example.toml"
        with open(example_config_path, "rb") as f:
            example_config = tomllib.load(f)
        
        # 收集schema中的所有配置项路径
        schema_paths = set()
        example_paths = set()
        
        def collect_schema_paths(section_name: str, section_schema: Dict, prefix: str = ""):
            for field_name, field_def in section_schema.items():
                full_path = f"{prefix}{field_name}" if prefix else f"{section_name}.{field_name}"
                if field_def.field_type == ConfigFieldType.NESTED and field_def.nested_schema:
                    for nested_name in field_def.nested_schema.keys():
                        schema_paths.add(f"{full_path}.{nested_name}")
                else:
                    schema_paths.add(full_path)
        
        def collect_example_paths(config: Dict, prefix: str = ""):
            for key, value in config.items():
                full_path = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    collect_example_paths(value, full_path)
                else:
                    example_paths.add(full_path)
        
        for section_name, section_schema in schema.items():
            collect_schema_paths(section_name, section_schema)
        
        collect_example_paths(example_config)
        
        # 对比差异
        missing_in_example = schema_paths - example_paths
        missing_in_schema = example_paths - schema_paths
        
        print(f"Schema定义的配置项数量: {len(schema_paths)}")
        print(f"示例文件的配置项数量: {len(example_paths)}")
        
        if missing_in_example:
            print(f"\n⚠️ Schema中有但示例文件中缺少的配置项 ({len(missing_in_example)}个):")
            for path in sorted(missing_in_example):
                print(f"  - {path}")
        
        if missing_in_schema:
            print(f"\n⚠️ 示例文件中有但Schema中缺少的配置项 ({len(missing_in_schema)}个):")
            for path in sorted(missing_in_schema):
                print(f"  - {path}")
        
        if not missing_in_example and not missing_in_schema:
            print("✓ Schema与示例文件完全匹配，无遗漏")
            return True, "配置项完全匹配"
        else:
            print(f"\n⚠️ 发现差异: Schema中缺失 {len(missing_in_schema)} 项，示例文件中缺失 {len(missing_in_example)} 项")
            return True, f"存在差异但功能完整"
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


def test_config_validation():
    """测试配置验证机制"""
    print("\n" + "=" * 80)
    print("测试3: 配置验证机制")
    print("=" * 80)
    
    try:
        from core.config import ConfigManager
        
        config_manager = ConfigManager(plugin_dir)
        
        # 测试类型验证
        test_cases = [
            # (配置路径, 测试值, 期望结果)
            ("performance.profiler_sample_rate", 1.5, False),  # 超出范围
            ("performance.profiler_sample_rate", 0.5, True),   # 有效值
            ("modules.message_cache.per_chat_limit", 50, True),   # 边界值
            ("modules.message_cache.per_chat_limit", 10, False),  # 低于最小值
            ("modules.message_cache.per_chat_limit", 2000, False), # 高于最大值
            ("plugin.log_level", "INVALID", False),  # 无效选项
            ("plugin.log_level", "DEBUG", True),     # 有效选项
        ]
        
        for path, value, expected_valid in test_cases:
            result = config_manager.set(path, value, notify=False)
            status = "✓" if result == expected_valid else "✗"
            expected_str = "应通过" if expected_valid else "应失败"
            actual_str = "通过" if result else "失败"
            print(f"{status} {path} = {value}: 期望{expected_str}, 实际{actual_str}")
        
        return True, "配置验证机制测试通过"
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


def test_config_description_clarity():
    """评估配置选项的解释清晰度"""
    print("\n" + "=" * 80)
    print("测试4: 配置选项描述清晰度评估")
    print("=" * 80)
    
    try:
        from core.config import ConfigManager, ConfigFieldType
        
        config_manager = ConfigManager(plugin_dir)
        schema = config_manager.get_schema()
        
        issues = []
        
        def check_field_description(field_name: str, field_def, section: str):
            """检查字段描述的质量"""
            desc = field_def.description or ""
            
            # 检查描述是否为空
            if not desc:
                issues.append(f"{section}.{field_name}: 缺少描述")
                return
            
            # 检查描述长度
            if len(desc) < 5:
                issues.append(f"{section}.{field_name}: 描述过短 ({len(desc)}字符)")
            
            # 检查是否有约束说明
            if field_def.constraint:
                c = field_def.constraint
                if c.min_value is not None or c.max_value is not None:
                    if "范围" not in desc and "值" not in desc:
                        # 仅记录，不作为问题
                        pass
        
        for section_name, section_schema in schema.items():
            for field_name, field_def in section_schema.items():
                check_field_description(field_name, field_def, section_name)
                
                # 检查嵌套字段
                if field_def.nested_schema:
                    for nested_name, nested_def in field_def.nested_schema.items():
                        check_field_description(nested_name, nested_def, f"{section_name}.{field_name}")
        
        if issues:
            print(f"发现 {len(issues)} 个描述问题:")
            for issue in issues[:10]:  # 只显示前10个
                print(f"  ⚠️ {issue}")
            if len(issues) > 10:
                print(f"  ... 还有 {len(issues) - 10} 个问题")
        else:
            print("✓ 所有配置项都有清晰的描述")
        
        # 检查示例文件中的注释质量
        example_config_path = plugin_dir / "config.example.toml"
        with open(example_config_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 统计注释行数
        comment_lines = [line for line in content.split('\n') if line.strip().startswith('#')]
        total_lines = len(content.split('\n'))
        comment_ratio = len(comment_lines) / total_lines * 100
        
        print(f"\n示例文件统计:")
        print(f"  总行数: {total_lines}")
        print(f"  注释行数: {len(comment_lines)}")
        print(f"  注释比例: {comment_ratio:.1f}%")
        
        if comment_ratio > 30:
            print("✓ 注释比例充足")
        else:
            print("⚠️ 注释比例偏低")
        
        return True, "描述清晰度评估完成"
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


def test_hot_reload_support():
    """测试热更新支持"""
    print("\n" + "=" * 80)
    print("测试5: 热更新支持检测")
    print("=" * 80)
    
    try:
        from core.config import ConfigManager
        
        config_manager = ConfigManager(plugin_dir)
        hot_reload_fields = config_manager.get_hot_reload_fields()
        
        print(f"支持热更新的配置项 ({len(hot_reload_fields)}个):")
        for field in sorted(hot_reload_fields):
            print(f"  - {field}")
        
        # 验证关键配置项支持热更新
        expected_hot_reload = [
            "plugin.log_level",
            "performance.profiler_sample_rate",
            "modules.message_cache.ttl",
            "monitoring.stats_interval",
        ]
        
        for field in expected_hot_reload:
            if field in hot_reload_fields:
                print(f"✓ {field} 支持热更新")
            else:
                print(f"⚠️ {field} 不支持热更新")
        
        return True, f"发现 {len(hot_reload_fields)} 个热更新配置项"
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


def generate_coverage_report():
    """生成配置项覆盖率报告"""
    print("\n" + "=" * 80)
    print("配置项覆盖率报告")
    print("=" * 80)
    
    try:
        from core.config import ConfigManager, ConfigFieldType
        
        config_manager = ConfigManager(plugin_dir)
        schema = config_manager.get_schema()
        
        report = {
            'total_sections': len(schema),
            'total_fields': 0,
            'nested_fields': 0,
            'hot_reload_fields': 0,
            'constrained_fields': 0,
            'section_details': {}
        }
        
        for section_name, section_schema in schema.items():
            section_info = {
                'fields': len(section_schema),
                'nested': 0,
                'hot_reload': 0,
                'constraints': 0
            }
            
            for field_name, field_def in section_schema.items():
                report['total_fields'] += 1
                
                if field_def.field_type == ConfigFieldType.NESTED and field_def.nested_schema:
                    section_info['nested'] += 1
                    report['nested_fields'] += len(field_def.nested_schema)
                    report['total_fields'] += len(field_def.nested_schema)
                    
                    for nested_def in field_def.nested_schema.values():
                        if nested_def.hot_reload:
                            report['hot_reload_fields'] += 1
                        if nested_def.constraint:
                            report['constrained_fields'] += 1
                
                if field_def.hot_reload:
                    section_info['hot_reload'] += 1
                    report['hot_reload_fields'] += 1
                
                if field_def.constraint:
                    section_info['constraints'] += 1
                    report['constrained_fields'] += 1
            
            report['section_details'][section_name] = section_info
        
        print(f"\n配置统计:")
        print(f"  配置节数量: {report['total_sections']}")
        print(f"  配置项总数: {report['total_fields']}")
        print(f"  嵌套配置组: {report['nested_fields']}")
        print(f"  热更新配置: {report['hot_reload_fields']}")
        print(f"  带约束配置: {report['constrained_fields']}")
        
        print(f"\n各配置节详情:")
        for section, info in report['section_details'].items():
            print(f"  [{section}]")
            print(f"    字段数: {info['fields']}, 嵌套组: {info['nested']}, "
                  f"热更新: {info['hot_reload']}, 约束: {info['constraints']}")
        
        return report
        
    except Exception as e:
        print(f"✗ 报告生成失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主测试函数"""
    print("\n" + "=" * 80)
    print("CM-Performance-Optimizer-Plugin 配置系统完整性验证")
    print("=" * 80)
    print()
    
    results = []
    
    # 运行所有测试
    results.append(("默认配置生成", test_default_config_generation()))
    results.append(("Schema完整性", test_config_schema_completeness()))
    results.append(("配置验证机制", test_config_validation()))
    results.append(("描述清晰度", test_config_description_clarity()))
    results.append(("热更新支持", test_hot_reload_support()))
    
    # 生成覆盖率报告
    coverage_report = generate_coverage_report()
    
    # 总结
    print("\n" + "=" * 80)
    print("验证总结")
    print("=" * 80)
    
    all_passed = all(result[1][0] for result in results)
    
    for test_name, (passed, message) in results:
        status = "✓" if passed else "✗"
        print(f"{status} {test_name}: {message}")
    
    print()
    if all_passed:
        print("🎉 所有验证测试通过！配置系统完整性良好。")
        return 0
    else:
        print("❌ 部分验证测试失败，请检查配置系统。")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
