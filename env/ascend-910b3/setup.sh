#!/usr/bin/env bash
# 昇腾 Atlas A2 (910B) 环境准备
#
# 前提：使用昇腾官方镜像 / 已装好 NPU 驱动 + CANN toolkit + 配套的
#       torch、torch_npu、triton-ascend。我们不去重装这些，只做校验和补齐
#       纯 python 依赖。
#
# 与沐曦最大的不同：昇腾**必须先 source CANN 的环境变量**，否则
# torch_npu 导入即失败、Triton 找不到毕昇编译器。这一步没有等价物，
# 沐曦镜像里驱动路径是直接进 PATH 的。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "==> [1/5] 加载 CANN 环境变量"
# 常见安装路径按优先级探测。root 默认装 ascend-toolkit，
# 非 root 装在 ~/Ascend，推理场景可能只有 nnae。
CANN_ENV=""
for p in \
  "${ASCEND_HOME_PATH:-}/../set_env.sh" \
  /usr/local/Ascend/ascend-toolkit/set_env.sh \
  /usr/local/Ascend/nnae/set_env.sh \
  "$HOME/Ascend/ascend-toolkit/set_env.sh" ; do
  if [ -n "$p" ] && [ -f "$p" ]; then CANN_ENV="$p"; break; fi
done

if [ -z "$CANN_ENV" ]; then
  echo "!! 找不到 CANN 的 set_env.sh。请确认已安装 ascend-toolkit，" >&2
  echo "   或手动 export ASCEND_HOME_PATH 后重试。" >&2
  exit 1
fi

# set_env.sh 内部会引用未定义变量，和 set -u 冲突，这里临时关掉。
set +u
# shellcheck disable=SC1090
source "$CANN_ENV"
set -u
echo "已加载: $CANN_ENV"
echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-<未设置>}"

# 毕昇编译器随 toolkit 一起装，Triton-Ascend 靠它生成 NPU 代码。
# 找不到通常意味着 toolkit 装的是精简包，或 set_env.sh 没生效。
if command -v bishengir-compile >/dev/null 2>&1; then
  echo "毕昇编译器: $(command -v bishengir-compile)"
else
  echo "!! PATH 里没有 bishengir-compile —— Triton kernel 编译会失败。" >&2
  echo "   检查 toolkit 是否为完整包，或 set_env.sh 是否真的生效。" >&2
fi

echo
echo "==> [2/5] 校验驱动与设备"
if command -v npu-smi >/dev/null 2>&1; then
  npu-smi info
else
  echo "!! 找不到 npu-smi。请确认已进入昇腾官方镜像 / 已加载 NPU 驱动。" >&2
  echo "   容器场景还需确认启动参数带了 --device=/dev/davinci* 等设备映射。" >&2
  exit 1
fi

echo
echo "==> [3/5] 校验 torch / torch_npu / triton-ascend 配套"
# 昇腾的版本配套比沐曦更严：驱动 <-> CANN <-> torch_npu <-> triton-ascend
# 四者互相绑定，任一错配都会在导入或首次编译时炸，且报错信息通常很难懂。
# torch_npu 的主版本号必须和 torch 完全一致（如 torch 2.7.1 配 torch_npu 2.7.1）。
python3 - <<'PY'
import sys

try:
    import torch
except Exception as e:
    print("!! torch 导入失败:", e); sys.exit(1)
print("torch:", torch.__version__)

try:
    import torch_npu
except Exception as e:
    print("!! torch_npu 导入失败:", e)
    print("   最常见原因: 没 source set_env.sh，或 torch_npu 与 torch 版本不配套")
    sys.exit(1)
print("torch_npu:", torch_npu.__version__)

# 主版本号比对：torch 2.7.1 <-> torch_npu 2.7.1.xxx
t = torch.__version__.split('+')[0].split('.')[:3]
n = str(torch_npu.__version__).split('+')[0].split('.')[:3]
if t != n:
    print(f"!! 版本不配套: torch {'.'.join(t)} vs torch_npu {'.'.join(n)}")
    print("   参见昇腾社区的配套表，二者主版本号必须一致")
    sys.exit(1)
print("torch/torch_npu 版本配套: OK")

if not torch.npu.is_available():
    print("!! torch.npu.is_available() == False —— 驱动或设备映射有问题")
    sys.exit(1)
print(f"npu 设备数: {torch.npu.device_count()}  name={torch.npu.get_device_name(0)}")

try:
    import triton
except Exception as e:
    print("!! triton 导入失败:", e); sys.exit(1)
print("triton:", triton.__version__)

# 判别装的是不是昇腾后端。**不要用版本号字符串匹配** —— 厂商构建未必带
# local version 后缀（沐曦就是干净的 3.0.0）。后端注册表才是可靠信号。
try:
    from triton.backends import backends as _b
    names = sorted(_b.keys())
    print("triton backends:", names)
    if names in (['nvidia'], ['amd', 'nvidia']):
        print("!! 这是上游 pytorch-triton，不是 triton-ascend。")
        print("   kernel 会去编译 NVIDIA 目标然后失败，报错会指向 ptxas/cuda。")
        print("   pip install triton-ascend （版本需与 CANN、torch_npu 配套）")
        sys.exit(1)
except Exception as e:
    print("triton backends: 不可探测 -", e)
PY

echo
echo "==> [4/5] 安装纯 python 依赖"
# 注意：torch、torch_npu、triton-ascend 刻意不在 requirements.txt 里 pin，
# 它们与驱动/CANN 版本强绑定，pip 覆盖装会直接把环境搞坏。
python3 -m pip install -r "$ROOT/env/ascend-a2/requirements.txt"

echo
echo "==> [5/5] 环境快照 + 后端自检"
bash "$ROOT/env/capture.sh" ascend-a2

cat <<'EOF'

昇腾上的注意事项
----------------
1. **每个新 shell 都要重新 source set_env.sh。** 这是昇腾最容易踩的坑：
   直接开个新终端跑脚本，torch_npu 导入就失败。建议写进 ~/.bashrc，
   或所有入口脚本都像本文件这样先加载一次。

2. 昇腾走 torch.npu 命名空间（不是 torch.cuda），且**必须先 import torch_npu**
   才会注册进去。auto_bench.py 的 _iter_accelerators() 会识别为 "npu"。
   本仓库统一用 CPU 张量再 .to(device)，正是为了同一份代码能在
   沐曦(cuda) / 昇腾(npu) / 寒武纪(mlu) 之间直接搬。

3. 设备可见性用 ASCEND_RT_VISIBLE_DEVICES（不是 CUDA_VISIBLE_DEVICES）。

4. Triton 首次编译每个 kernel 有秒级耗时，auto_bench 默认 warmup=200
   足以覆盖。但**若 kernel 里用了 tl.static_range 且展开次数多，冷编译
   可能到分钟级**，warmup 计时会被污染 —— 这种情况先单独跑一次预热。

5. 若 Triton 缓存出现异常（改了 kernel 但行为没变），清掉 ~/.triton/cache。
   昇腾侧还可能有 CANN 的算子编译缓存，位置见 ASCEND_CACHE_PATH。

6. triton-ascend 相比上游 Triton 有功能缺口（部分 atomic 操作等）。
   移植 kernel 时先跑 env/selftest.py 的特性冒烟，别等真 kernel 报错。
   已知在沐曦 C500 上，「scf.for 的 loop-carried 是 2D tensor 且循环体内
   对它做规约」会让编译器段错误；昇腾是否有同类限制需实测，
   selftest 的 ⑥a 探针就是干这个的。

7. 调试崩溃：昇腾的错误常在 CANN 的原生代码里，python 的 try/except 拦不住。
   用 python3 -X faulthandler，并在每次 kernel 调用后 torch.npu.synchronize()
   把异步错误的暴露点收拢到紧邻位置。
EOF