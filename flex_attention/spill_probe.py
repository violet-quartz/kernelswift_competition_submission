#!/usr/bin/env python3
"""spill 探针：warmup 编译一次 v1 kernel，不执行。供 bench/check_spill.py 通用驱动调用。

约定：模块级函数 warmup(dev) -> triton kernel（kernel.warmup(...) 的原始返回值，
不要在这里对**返回的那个** kernel 调 _init_handles()，通用驱动会调）。

为什么这道题需要这个探针
------------------------
这是本仓库寄存器压力最大的一个 kernel。grid 只有 (NUM_HEADS,) = 8 个 program，
每个 program 一趟吃下整个注意力头，同时活跃的 tile（按 32-bit 寄存器槽计，
fp16 两个打包进一个槽）：

    S / P   [128, 128] fp32  -> 16384      <- 大头
    acc     [128,  64] fp32  ->  8192
    Q       [128,  64] fp16  ->  4096
    K_t     [ 64, 128] fp16  ->  4096
    V       [128,  64] fp16  ->  4096
                        合计  ≈ 37000

C500 实测 regsPerBlock=131072、maxThreadsPerBlock=1024、warpSize=64，
所以每线程可用寄存器 = 131072 / 线程数，且受架构上限约 255 卡住：

    num_warps=4  ->  256 线程 -> 145 个/线程   <- 逼近上限，最可能 spill
    num_warps=8  ->  512 线程 ->  72 个/线程
    num_warps=16 -> 1024 线程 ->  36 个/线程

**溢出不会报错，只会悄悄变慢** —— 装不下就换出到 local memory（其实还是显存）。
所以要在跑 run.sh 之前先查一次。

真溢出的话的退路
----------------
把 Q 切块：grid 从 (NUM_HEADS,) 改成 (NUM_HEADS, cdiv(SEQ_LEN, BLOCK_M))，
BLOCK_M=32 时 S 只有 [32, 128]，压力降到约 1/4。
**注意这样切依然不引入循环**（每个 program 处理一个 Q 块、K/V 一次读全），
沐曦上"2D tile 作为 loop-carried 变量 + 循环体内规约"触发 make_ttgir 段错误
那个雷仍然是躲开的 —— 见 hc_split_sinkhorn/docs/。

不要改成经典 flash-attention 那种沿 K/V 分块的循环写法，那个正好踩雷。
"""
import importlib.util
import sys
from pathlib import Path

import torch
import triton

OP_DIR = Path(__file__).resolve().parent

# 必须和 v1/flex_attention.py 里 @triton.autotune 的 configs 保持一致
NUM_WARPS_CANDIDATES = (4, 8, 16)


def _load_v1():
    path = OP_DIR / "v1" / "flex_attention.py"
    spec = importlib.util.spec_from_file_location("_ks_v1_flexattn_spillprobe", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _unwrap_autotuner(k):
    """剥掉 @triton.autotune 的包装，拿到底下的 JITFunction。

    Autotuner 有 .configs 而 JITFunction 没有，用这个做判据 —— 不能用
    `hasattr(k, "fn")` 循环剥，因为 JITFunction.fn 是原始的 Python 函数，
    会一路剥过头。也不要直接调 Autotuner.warmup()：部分 Triton 版本上它会
    把所有 config 都编一遍并返回 list，跟本探针的单 kernel 约定对不上。
    """
    if hasattr(k, "configs") and hasattr(k, "fn"):
        return k.fn
    return k


def _build_args(mod, dev):
    """照 forward() 的口径把宿主侧参数摆好，形状/grid/constexpr 都跟真实调用一致。"""
    model = mod.ModelNew(*mod.get_init_inputs()).to(dev)
    query, key, value = (t.to(dev) for t in mod.get_inputs())

    num_tokens = query.shape[0]
    out = torch.empty(num_tokens, model.num_heads * model.head_size,
                      device=dev, dtype=query.dtype)
    block = triton.next_power_of_2(num_tokens)          # 83 -> 128

    args = (query, key, value, out)
    kwargs = dict(
        # constexpr 一律用关键字传：warmup() 对 constexpr 位置参数的绑定在不同
        # Triton 版本间行为不一，其他算子的探针里这套写法在沐曦 3.0.0 和
        # 昇腾 3.2.0 上都跑通过。
        SCALE=model.scale,
        SEQ_LEN=num_tokens,
        NUM_HEADS=model.num_heads,
        HEAD_SIZE=model.head_size,
        BLOCK_M=block,
        BLOCK_N=block,
        IS_CAUSAL=True,
        grid=(model.num_heads,),
    )
    return args, kwargs


def warmup(dev):
    mod = _load_v1()
    jit_fn = _unwrap_autotuner(mod._flash_attention_kernel)
    args, kwargs = _build_args(mod, dev)

    # autotune 会在三个 num_warps 里挑，而 warmup() 必须显式指定一个。
    # 先把三个都编一遍打张表 —— 光看一个 config 会误判：num_warps=4 溢出
    # 但 8/16 不溢出时，autotune 本来就会避开 4（它按耗时选，溢出的必然慢）。
    # **只有三个全溢出才是真问题**，那时候才需要回去切 Q 块。
    rows = []
    for nw in NUM_WARPS_CANDIDATES:
        k = jit_fn.warmup(*args, num_warps=nw, **kwargs)
        regs = spills = None
        try:
            k._init_handles()               # 这些是探针自己的副本，不是返回给驱动的那个
            regs, spills = k.n_regs, k.n_spills
        except Exception:
            pass                            # 非 CUDA 系后端读不到，交给驱动去说明
        rows.append((nw, regs, spills))

    print("各 num_warps 候选的静态资源（本探针自测，判定以驱动输出为准）：")
    print(f"    {'num_warps':>9}  {'线程数':>6}  {'n_regs':>7}  {'n_spills':>8}")
    for nw, regs, spills in rows:
        print(f"    {nw:>9}  {nw * 64:>6}  {str(regs):>7}  {str(spills):>8}")
    print()

    # 返回"最好"的那个给驱动判定：spill 最少，并列时取寄存器最少的。
    # 读不到指标（昇腾等）就退回第一个候选，让驱动去打"本平台测不了"。
    usable = [r for r in rows if isinstance(r[2], int)]
    best_nw = min(usable, key=lambda r: (r[2], r[1] or 0))[0] if usable else NUM_WARPS_CANDIDATES[0]

    # 重新 warmup 一个干净的对象返回 —— 上面那些已经被 _init_handles() 过了，
    # 而驱动会再调一次。编译结果有缓存，这次是命中，不额外花时间。
    return jit_fn.warmup(*args, num_warps=best_nw, **kwargs)
