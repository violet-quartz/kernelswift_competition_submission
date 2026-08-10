#!/usr/bin/env bash
# 沐曦 C500 环境准备
#
# 前提：使用沐曦官方提供的 MACA 容器镜像 / 机器环境，其中已预装
#       MACA 驱动 + toolkit + 适配 MACA 的 PyTorch 和 Triton。
#       我们不去重装这些，只做校验和补齐纯 python 依赖。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "==> [1/3] 校验沐曦驱动与设备"
if command -v mx-smi >/dev/null 2>&1; then
  mx-smi
else
  echo "!! 找不到 mx-smi。请确认已进入沐曦官方镜像/已加载 MACA 驱动。" >&2
  exit 1
fi

echo
echo "==> [2/3] 安装纯 python 依赖"
# 注意：torch 和 triton 刻意不在 requirements.txt 里 pin，
# 它们由沐曦镜像提供且与 MACA 版本强绑定，pip 覆盖装会直接把环境搞坏。
python3 -m pip install -r "$ROOT/env/metax-c500/requirements.txt"

echo
echo "==> [3/3] 环境快照 + 后端自检"
bash "$ROOT/env/capture.sh" metax-c500

cat <<'EOF'

沐曦上的注意事项
----------------
1. C500 走 torch.cuda 命名空间，所以 auto_bench.py 的 _iter_accelerators()
   会把它识别为 "cuda"。device="cuda" 的字面量在这里是可用的
   —— 但本仓库仍统一用 CPU 张量，以便同一份代码能直接搬到昇腾上。
2. Triton 首次编译每个 kernel 会有秒级耗时，auto_bench 默认 warmup=200
   足以覆盖，不必额外预热。
3. 若 Triton 缓存出现异常（改了 kernel 但行为没变），清掉 ~/.triton/cache。
EOF
