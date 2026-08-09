#!/usr/bin/env bash
# 快照当前机器的真实环境到 env/<chip>/env.lock.txt，并做后端连通性自检。
#
# 用法: bash env/capture.sh metax-c500
#       bash env/capture.sh ascend-a2
set -uo pipefail

CHIP="${1:-}"
if [[ -z "$CHIP" ]]; then
  echo "用法: bash env/capture.sh <chip>    # 例如 metax-c500 / ascend-a2" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/env/$CHIP/env.lock.txt"
mkdir -p "$(dirname "$OUT")"

{
  echo "# 环境快照 — $CHIP"
  echo "# 生成时间: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "# 由 env/capture.sh 自动生成，请勿手改"
  echo
  echo "## 系统"
  echo "hostname: $(hostname 2>/dev/null || echo '?')"
  echo "kernel:   $(uname -srm 2>/dev/null || echo '?')"
  [[ -f /etc/os-release ]] && echo "os:       $(. /etc/os-release && echo "$PRETTY_NAME")"
  echo
  echo "## Python 栈"
  echo "python:   $(python3 -V 2>&1)"
  echo "which:    $(command -v python3)"
  echo
  echo "## 设备"
  for cmd in mx-smi npu-smi cnmon rocm-smi nvidia-smi; do
    if command -v "$cmd" >/dev/null 2>&1; then
      echo "--- $cmd ---"
      case "$cmd" in
        npu-smi) "$cmd" info 2>&1 | head -30 ;;
        *)       "$cmd"      2>&1 | head -30 ;;
      esac
      echo
    fi
  done
  echo "## 已安装包（torch / triton 相关）"
  python3 -m pip list 2>/dev/null | grep -iE 'torch|triton|numpy|maca|ascend|npu|mlu' || echo "(pip list 不可用)"
  echo
  echo "## torch / triton 自检"
  # 自检代码放在 env/selftest.py 这个真实文件里，不能写成 heredoc 从 stdin 喂进来：
  # @triton.jit 构造时要用 inspect.getsourcelines() 抠 kernel 源码做 AST 解析，
  # 而 <stdin> 在磁盘上没有对应文件，会直接抛 OSError: could not get source code。
  python3 "$ROOT/env/selftest.py" 2>&1
  SELFTEST_RC=$?
  echo
  if [[ $SELFTEST_RC -eq 0 ]]; then
    echo "## 自检结论: 通过 (rc=0) —— 后端可用、Triton 工具链可用"
  else
    echo "## 自检结论: 失败 (rc=$SELFTEST_RC) —— 在修好之前，性能数字都不可信"
  fi
} > "$OUT" 2>&1

echo "已写入 $OUT"
echo "----------------------------------------"
tail -40 "$OUT"
grep -q "自检结论: 通过" "$OUT" || { echo; echo "!! 环境自检未通过，详见上面的输出"; exit 1; }
