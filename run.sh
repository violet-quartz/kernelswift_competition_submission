#!/usr/bin/env bash
# KernelSwift 提交作品 —— 一键运行脚本
#
# 用法:
#   bash run.sh                          # 自动探测芯片，按官方参数跑全部 task
#   bash run.sh --chip metax-c500        # 指定输出目录名
#   bash run.sh --only hc_split_sinkhorn # 只跑某一题
#   bash run.sh --warmup 20 --repeat 50  # 开发期快速迭代（不可用于最终性能报告）
#
# 输出:
#   results/<chip>/results.json   完整结果（含每题的原始 auto_bench 输出）
#   results/<chip>/RESULTS.md     性能测试结果表
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "================================================================"
echo " KernelSwift — Triton 算子优化"
echo "================================================================"
echo

# 计时和对拍全部交给官方的 bench/auto_bench.py（与上游逐字一致，未做任何改动），
# run_all.py 只负责批量拉起子进程和汇总，不介入任何计时或比对逻辑。
exec python3 bench/run_all.py "$@"
