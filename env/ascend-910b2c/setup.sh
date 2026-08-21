#!/usr/bin/env bash
# 昇腾 Atlas A2 环境准备
#
# 实测机器报 Ascend910B2C（申请时写的是 910B3，以 npu-smi / get_device_name 为准）。
#
# 前提：昇腾官方镜像已装好 NPU 驱动 + CANN toolkit + 配套的 torch / torch_npu。
#       **但镜像通常不带 triton-ascend** —— 实测那台系统 python 里是上游
#       triton 3.6.0（backends=['amd','nvidia']），拿它编 kernel 会去打 NVIDIA
#       目标然后失败。所以本脚本会把 triton-ascend 装进一个**独立 venv**。
#
# 为什么是 venv 而不是直接 pip 装进系统环境：
#   * torch / torch_npu 与驱动强绑定，绝不能被 pip 覆盖（见下面第 4 步的说明）；
#   * triton-ascend 会带自己的 triton 3.2.0 **替换掉**现有的 3.6.0；
#   * venv 加 --system-site-packages 既能复用系统的 torch/torch_npu，又能
#     随时 rm -rf 完全回退，不留痕迹。
# 编排工具 chips/*.toml 里的 remote_python 就指向这个 venv。
#
# 与沐曦最大的不同：昇腾**必须先 source CANN 的环境变量**，否则
# torch_npu 导入即失败、Triton 找不到毕昇编译器。这一步没有等价物，
# 沐曦镜像里驱动路径是直接进 PATH 的。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "==> [1/6] 加载 CANN 环境变量"
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
echo "==> [2/6] 校验驱动与设备"
if command -v npu-smi >/dev/null 2>&1; then
  npu-smi info
else
  echo "!! 找不到 npu-smi。请确认已进入昇腾官方镜像 / 已加载 NPU 驱动。" >&2
  echo "   容器场景还需确认启动参数带了 --device=/dev/davinci* 等设备映射。" >&2
  exit 1
fi

echo
echo "==> [3/6] 校验 torch / torch_npu 配套（系统 python）"
# 昇腾的版本配套比沐曦更严：驱动 <-> CANN <-> torch_npu <-> triton-ascend
# 四者互相绑定，任一错配都会在导入或首次编译时炸，且报错信息通常很难懂。
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

# [KS-PORT] 只比 major.minor，且要容忍 rc/post 后缀 ——
# 实测那台是 torch 2.9.0+cpu 配 torch_npu 2.9.0rc1，
# 按 major.minor.patch 逐段比会把 '0' 和 '0rc1' 判成不配套，误报。
t = torch.__version__.split('+')[0].split('.')[:2]
n = str(torch_npu.__version__).split('+')[0].split('.')[:2]
if t != n:
    print(f"!! 版本不配套: torch {'.'.join(t)}.x vs torch_npu {'.'.join(n)}.x")
    print("   参见昇腾社区的配套表")
    sys.exit(1)
print("torch/torch_npu 版本配套: OK")

if not torch.npu.is_available():
    print("!! torch.npu.is_available() == False —— 驱动或设备映射有问题")
    sys.exit(1)
print(f"npu 设备数: {torch.npu.device_count()}  name={torch.npu.get_device_name(0)}")
PY

echo
echo "==> [4/6] 建独立 venv 并安装 triton-ascend"
# 为什么不装进系统环境，见文件抬头。--system-site-packages 是关键：
# 复用系统那份与驱动绑定的 torch / torch_npu，只把 triton 换掉。
VENV="${KS_VENV:-$HOME/ks-venv}"
if [ ! -f "$VENV/bin/python" ]; then
  echo "创建 venv: $VENV"
  python3 -m venv --system-site-packages "$VENV"
else
  echo "venv 已存在: $VENV"
fi

# 已经是厂商版就跳过。判据用后端注册表，**不要用版本号字符串** ——
# 厂商构建未必带 local version 后缀（昇腾那份也自称 3.2.0，和上游撞号）。
if "$VENV/bin/python" - <<'PY'
import sys
try:
    import triton
    sys.exit(0 if "ascend" in triton.backends.backends else 1)
except Exception:
    sys.exit(1)
PY
then
  echo "venv 里已是 triton-ascend，跳过安装"
else
  echo "安装 triton-ascend（版本需与 CANN、torch_npu 配套）"
  # pypi 上目前只有 3.2.0；更新的版本在昇腾社区/gitee，不在 pypi。
  "$VENV/bin/pip" install "${KS_TRITON_ASCEND:-triton-ascend==3.2.0}"
fi

# 纯 python 依赖。pybind11 是 triton-ascend 漏声明的依赖，见 requirements.txt。
"$VENV/bin/pip" install -r "$ROOT/env/ascend-910b2c/requirements.txt"

echo
echo "==> [5/6] 在 venv 里验证 triton 后端"
"$VENV/bin/python" - <<'PY'
import sys
try:
    import triton
except Exception as e:
    print("!! triton 导入失败:", e)
    print("   若报 ModuleNotFoundError: pybind11 —— triton-ascend 漏声明了该依赖，")
    print("   pip install pybind11 补上即可。")
    sys.exit(1)
print("triton:", triton.__version__, "  file:", triton.__file__)

# 判别装的是不是昇腾后端。**唯一可靠的信号是后端注册表** ——
# __version__ 和 __file__ 都区分不出来。
names = sorted(triton.backends.backends)
print("triton backends:", names)
if "ascend" not in names:
    print("!! venv 里的 triton 不是昇腾后端（大概率是上游 pytorch-triton）。")
    print("   kernel 会去编译 NVIDIA 目标然后失败，报错会指向 ptxas/cuda。")
    sys.exit(1)
print("triton-ascend: OK")

import torch, torch_npu   # noqa: F401  确认 venv 里也看得到系统的 torch/torch_npu
print("venv 内 torch:", torch.__version__, " npu 可用:", torch.npu.is_available())
PY

echo
echo "venv 就绪: $VENV/bin/python"
echo "编排工具用法: chips/<chip>.toml 里 remote_python 指向它"

echo
echo "==> [6/6] 环境快照 + 后端自检"
# [KS-PORT] 必须把 venv 放进 PATH ——  capture.sh / selftest.py 内部调的是
# `python3`，不带这一步会解析到系统 python3，那里是上游 triton 3.6.0，
# 报 `RuntimeError: 0 active drivers ([])`（它在找 NVIDIA/AMD 驱动）。
PATH="$VENV/bin:$PATH" bash "$ROOT/env/capture.sh" ascend-910b2c

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

6. 调试崩溃：昇腾的错误常在 CANN 的原生代码里，python 的 try/except 拦不住。
   用 python3 -X faulthandler，并在每次 kernel 调用后 torch.npu.synchronize()
   把异步错误的暴露点收拢到紧邻位置。

7. **triton-ascend 3.2.0 已实测出的四条限制**（都做过最小复现，不是从报错猜的）：

   a. `tl.reshape` 之后不能再做**带 index** 的规约。拍平成一维后接
      `tl.argmax`/`tl.argmin` 报
          'hfusion.reduce_with_index' op currently ReduceWithIndexOp
          only supports one reduction dimension
      对照过四种写法：`reshape → tl.max` 正常、天然一维 argmax 正常、
      二维 `argmax(axis=1)` 正常，**只有 reshape → argmax 挂**。
      也就是 reshape 本身没问题，后端是在 index 那条路径上没折掉它。

   b. **不接受末维非连续的 load**。把大跨步放在最内层
      （`offs_d[:, None] + offs_n[None, :] * stride`，即"转置读"）编译失败。
      这与 `tl.dot` 无关 —— 转置载入哪怕只 store 出去也一样挂；
      "行跨步、末维连续"的 load 完全正常。改成自然布局读入再 `tl.trans`。
      ⚠ 沐曦的取向**相反**（对布局转换脆，宁可转置读），必须分芯片。

   c. **UB（片上 Unified Buffer）只有 192KB**，超了编译失败：
          ub overflow, requires 3932160 bits while 1572864 bits available
      UB 只认 **tile 形状**，不认生命周期 —— 内联 `tl.trans`、推迟 load、
      合并中间量，要的 UB 一个 bit 都不会变。想降只能真的缩小 tile。

   d. `root alloc` 报错有**两个**来源：b 的直接症状，以及 c 的伴生噪声。
      只看它会诊断错方向，必须往下翻有没有 `ub overflow` 那行。

   另外，沐曦上那个「scf.for 的 loop-carried 是 2D tensor 且循环体内对它做规约」
   导致编译器段错误的限制，昇腾上**没有**触发过（本仓库的 kernel 为绕开沐曦
   本来就不写这种循环，所以也没有反向验证）。

8. **性能上的两条**，选题和调参前先看：

   a. **一次 Triton kernel 启动固定 ~18us**（空 kernel、grid=1 和 grid=128 都一样），
      而一个 torch 逐元素算子只要 3.4us。所以估算收益时先算：
      v0 的 torch 算子数 × 3.4us 若和 18us 同量级，这题在昇腾上就没有摊薄空间。
      `head_compute_mix_bwd` 实测天花板 0.86x 就是这么来的。

   b. **别在 kernel 里现算常量掩码。** `offs_n[None, :] <= offs_m[:, None]` 这种
      "两个 1D arange 广播成 2D 再比较"极慢：`flex_attention` 单测 0.1375ms，
      同一 kernel 去掉因果只要 0.0278ms —— 掩码一项占 80%。改成 host 预算好、
      kernel 里直接 load，回到 0.0274ms，端到端 0.73x → 1.73x。

9. `num_warps` / `num_stages` 那条调参轴在昇腾上是**废的**，triton-ascend 会主动
   警告 `Please DO NOT tune args [...]`。为别的卡写的
   `autotune(configs=[num_warps=1/2/4/8])` 在这里是 benchmark 四个等价配置，
   选出来的是噪声。

10. 传文件：`rsync` 没有也装不了。`scp` 二进制在，但远端 sshd 声明的
    `Subsystem sftp` 指向一个不存在的文件，OpenSSH 9+ 的 scp 默认走 SFTP
    会报 `Connection closed`（看着像网络问题）。用 `scp -O` 回退到传统协议，
    或者 `ssh <host> 'cat > 路径' < 本地文件`；打包传用 tar 管道最稳。

11. 非交互 SSH **不加载 .bashrc**，于是 set_env.sh 没生效、torch_npu 导入即失败。
    自动化里要么显式 `bash -lc`，要么每条命令前先 source。

12. `pip list` 在 venv 里会**同时列出 triton 3.6.0 和 triton-ascend 3.2.0** ——
    前者是 --system-site-packages 透过来的系统包，不是装重了。实际 `import triton`
    解析到 venv 里那份（`triton.__file__` 指向 venv、`backends == ['ascend']`），
    env.lock.txt 的"triton 来源"那行就是用来确认这点的。

EOF