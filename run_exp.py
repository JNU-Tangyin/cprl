#!/usr/bin/env python3
"""
CPRL 实验运行脚本
修复了模块导入路径问题
"""
import sys
import os

# 添加项目根目录到 Python 路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 添加 src 目录到路径
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# 添加 exp/exp 目录到路径
EXP_DIR = os.path.join(PROJECT_ROOT, "exp", "exp")
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)

# 添加 time_series_library 到路径
TSL_DIR = os.path.join(PROJECT_ROOT, "time_series_library")
if TSL_DIR not in sys.path:
    sys.path.insert(0, TSL_DIR)

print(f"[Setup] Project root: {PROJECT_ROOT}")
print(f"[Setup] Python path configured")

# 导入并运行实验
from exp.exp_conformal import ExpConformal, get_args

if __name__ == "__main__":
    args = get_args()
    exp = ExpConformal(args)
    exp.run()
    print("Experiment finished successfully.")
