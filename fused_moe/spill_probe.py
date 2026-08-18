#!/usr/bin/env python3
"""spill 探针：warmup 编译一次 v1 kernel，不执行。供 bench/check_spill.py 通用驱动调用。

约定：模块级函数 warmup(dev) -> triton kernel（kernel.warmup(...) 的原始返回值，
不要在这里调 _init_handles()，通用驱动会调）。

为什么这道题需要这个探针
------------------------
本 kernel 同时活着的临时量比前几道题都多 —— 一个 [BLOCK_T, H] 的 fp32 累加器、
三块权重 tile、两个 [BLOCK_T, I] 的 fp32 GEMM 结果，外加输入 x。按 32-bit
寄存器槽粗估（H=128, I=64）：

    权重 tile（与 BLOCK_T 无关的地板）
        w1g [128,64] fp16  4096 槽
        w1u [128,64] fp16  4096 槽
        w2t [64,128] fp16  4096 槽                        小计 12288 槽

    随 BLOCK_T 线性增长的部分（x / acc / gate / up / y / gate_w）
        BLOCK_T=16   ~ 7300 槽      合计 ~19.6k
        BLOCK_T=32   ~14600 槽      合计 ~26.9k
        BLOCK_T=64   ~29200 槽      合计 ~41.5k
        BLOCK_T=128  ~58400 槽      合计 ~70.7k

摊到线程上（沐曦 C500 实测 warpSize=64，所以 num_warps 要乘 64 而不是 32；
每线程架构上限约 255）：

        BLOCK_T=128, num_warps=4  → 256 线程 → ~276 个/线程  ← **超上限，必 spill**
        BLOCK_T=128, num_warps=16 → 1024 线程 → ~69 个/线程
        BLOCK_T=64,  num_warps=4  → 256 线程 → ~162 个/线程   ← 逼近，危险
        BLOCK_T=32,  num_warps=4  → 256 线程 → ~105 个/线程
        BLOCK_T=16,  num_warps=4  → 256 线程 →  ~77 个/线程

注意那 12288 槽的**权重地板不随 BLOCK_T 缩小**：BLOCK_T 从 32 砍到 16，总量只
从 26.9k 降到 19.6k（-27%），不是减半。所以靠调小 BLOCK_T 来救 spill 是有天花板的，
num_warps 才是这里真正的"寄存器扩容"手段 —— 这也是 [KS-TUNE] 里把 num_warps
候选给到 16 的原因。

**溢出不会报错，只会悄悄变慢**，所以要在跑 run.sh 之前先查一次。

探针查哪个 config
-----------------
v1 的 kernel 挂了 @triton.autotune，有十几个 config。本探针**故意挑最容易溢出的
那个**来编译 —— 它是这批 config 的上界，它不溢出则全都不溢出。

判据不是"BLOCK_T 最大"，而是上面那个估算式本身：

    每线程寄存器数 ∝ (12288 + 456 * BLOCK_T) / num_warps

分子里那个 12288 的常数地板（三块权重 tile）意味着 BLOCK_T 和 num_warps 不是
对称的。按当前 config 表实测排下来最危险的是 BLOCK_T=64 / num_warps=4（~162 个
每线程），而不是 BLOCK_T=128 / num_warps=16（~69 个）—— 后者虽然 tile 最大，
但线程也开到了 1024。

反过来，如果它溢出了**不代表成绩会差**：autotune 会在实测中把它淘汰掉。
真正要做的是看 n_regs 的绝对值，判断 [KS-TUNE] 的 config 表是不是整体偏到了
危险区、需要把重心往"大 num_warps / 小 BLOCK_T"挪。

要挨个看全部 config，用 warmup_all(dev)（本模块提供，通用驱动不会调）。
"""
import importlib.util
import os
import sys
from pathlib import Path

import torch
import triton

OP_DIR = Path(__file__).resolve().parent


# --- 两个静态资源估算式 ---
#
# shared memory（**本题真正的瓶颈**，见 v1 里的 [KS-SMEM]）：
#     smem = BLOCK_T·H·2 + BLOCK_T·I·2 + 3·H·I·2·num_stages
# 保守上界，C500 上校准过 7 个点：ns=2 时逐字节相符（BLOCK_T=64 → 122880）；
# ns=1 时估算比实测多一个 act 项（BLOCK_T=32：估 61440 / 实测 57344），
# 因为权重不双缓冲时 act 能复用已死的 w1g 空间。多出来的部分当安全余量。
_SMEM_LIMIT = 65536              # C500 实测硬件上限

# 寄存器槽（次要约束）：常数项 = 三块权重 tile，一次项 = x/acc/gate/up/y/gate_w
_SLOTS_FLOOR = 4096 * 3
_SLOTS_PER_BLOCK_T = 64 + 128 + 64 + 64 + 128 + 8
_WARP_SIZE = 64                  # 沐曦 C500 实测值


def _smem_bytes(block_t, num_stages, H=128, I=64):
    return block_t * H * 2 + block_t * I * 2 + 3 * H * I * 2 * num_stages


def _regs_per_thread(block_t, num_warps):
    return (_SLOTS_FLOOR + _SLOTS_PER_BLOCK_T * block_t) / (num_warps * _WARP_SIZE)


# 哪个版本：默认 v1，用环境变量切到 v2 做对照
#     KS_FUSED_MOE_VER=v2 python3 bench/check_spill.py fused_moe
_VER = os.environ.get("KS_FUSED_MOE_VER", "v1")


def _load_v1():
    path = OP_DIR / _VER / "fused_moe.py"
    spec = importlib.util.spec_from_file_location(f"_ks_{_VER}_fused_moe_spillprobe", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _host_side(mod, dev):
    """照 forward() 的口径把宿主侧参数摆好。返回 (jit_fn, kwargs, T, configs)。

    kernel.warmup() 只编译、不执行，所以传的张量不需要有意义的数值。
    """
    # v1 是单 kernel（_fused_moe_kernel）；v2 拆成两个，主 kernel 是
    # _moe_expert_kernel（规约那个没有 tl.dot，不占 shared memory，不必查）。
    kernel = getattr(mod, "_moe_expert_kernel", None) or mod._fused_moe_kernel
    # 挂了 autotune 之后 kernel 是 Autotuner 而不是 JITFunction，
    # .fn 才是能接受显式 BLOCK_T / num_warps 的那一层。
    jit_fn = getattr(kernel, "fn", kernel)
    configs = list(getattr(kernel, "configs", []))

    # 走 ModelNew 构造再 .to(dev)，跟 auto_bench.py L526 的 model_new.to(...) 对齐：
    # w1/w2 是 nn.Parameter，要靠这一步才会上卡。
    model = mod.ModelNew(*mod.get_init_inputs()).to(dev)
    hidden_states, router_logits = (t.to(dev) for t in mod.get_inputs())

    T, H = hidden_states.shape
    x = hidden_states.contiguous()
    logits = router_logits.contiguous()
    # 预转置 + 预降精度的权重，跟 forward 里走同一条路径（见 [KS-CACHE]）
    w1t, w2t = model._prepared_weights(x.dtype)

    if _VER == "v1":
        sink = torch.empty((T, H), dtype=x.dtype, device=dev)                  # out
    else:
        sink = torch.empty((model.num_experts, T, H), dtype=torch.float32,
                           device=dev)                                          # partial

    kwargs = dict(
        E=model.num_experts,
        H=model.hidden_size,
        I=model.intermediate_size,
        TOP_K=model.top_k,
        RENORM=model.renormalize,
    )
    return jit_fn, (x, logits, w1t, w2t, sink, T), kwargs, T, configs


def _compile_one(jit_fn, args, kwargs, T, block_t, num_warps, num_stages):
    return jit_fn.warmup(
        *args,
        # constexpr 一律用关键字传：warmup() 对 constexpr 位置参数的绑定在不同
        # Triton 版本间行为不一，mhc_post/spill_probe.py 里这套写法在沐曦 3.0.0
        # 和昇腾 3.2.0 上都跑通过。
        **kwargs,
        BLOCK_T=block_t,
        num_warps=num_warps,
        num_stages=num_stages,
        grid=((triton.cdiv(T, block_t),) if _VER == "v1"
              else (kwargs["E"], triton.cdiv(T, block_t))),
    )


def warmup(dev):
    """通用驱动的入口：只编译最容易溢出的那个 config（见模块 docstring）。"""
    mod = _load_v1()
    jit_fn, args, kwargs, T, configs = _host_side(mod, dev)

    if configs:
        # 按 shared memory 排，不是按寄存器 —— C500 上先撞上的是 smem
        worst = max(configs, key=lambda c: _smem_bytes(c.kwargs["BLOCK_T"], c.num_stages))
        block_t = worst.kwargs["BLOCK_T"]
        num_warps, num_stages = worst.num_warps, worst.num_stages
    else:
        # v1 万一哪天摘掉了 autotune，退回一组保守默认值，探针仍然可用
        block_t, num_warps, num_stages = 32, 4, 2

    print(f"[spill_probe] 版本={_VER}  编译最危险的 config: "
          f"BLOCK_T={block_t} num_warps={num_warps} num_stages={num_stages} "
          f"grid=({triton.cdiv(T, block_t)},)")
    smem = _smem_bytes(block_t, num_stages)
    print(f"[spill_probe] 估算 smem={smem} B (上限 {_SMEM_LIMIT} B, "
          f"{'装得下' if smem <= _SMEM_LIMIT else '**装不下**'})，"
          f"~{_regs_per_thread(block_t, num_warps):.0f} 寄存器/线程")
    return _compile_one(jit_fn, args, kwargs, T, block_t, num_warps, num_stages)


def warmup_all(dev):
    """手动用：把 autotune 的每个 config 都编一遍，打印各自的 n_regs / n_spills。

        python3 -c "import sys; sys.path.insert(0,'fused_moe'); \
import torch, spill_probe as p; p.warmup_all(torch.device('cuda'))"
    """
    mod = _load_v1()
    jit_fn, args, kwargs, T, configs = _host_side(mod, dev)
    if not configs:
        raise SystemExit("v1 没有挂 autotune，没有 config 可枚举")

    rows = []
    for c in sorted(configs, key=lambda c: (c.kwargs["BLOCK_T"], c.num_warps, c.num_stages)):
        bt, nw, ns = c.kwargs["BLOCK_T"], c.num_warps, c.num_stages
        # 单个 config 装不下要继续跑完剩下的，不能整个中断 —— 这个函数的用途就是
        # 在 run.sh 之前把整张表探一遍，看哪些能编译。
        try:
            k = _compile_one(jit_fn, args, kwargs, T, bt, nw, ns)
            k._init_handles()
            n_regs, n_spills = k.n_regs, k.n_spills
            smem = getattr(getattr(k, "metadata", None), "shared", None)
        except Exception as exc:
            tag = "装不下" if "out of resource" in str(exc).lower() else type(exc).__name__
            n_regs = n_spills = smem = f"<{tag}>"
        rows.append((bt, nw, ns, n_regs, n_spills, smem,
                     _smem_bytes(bt, ns), _regs_per_thread(bt, nw)))

    print(f"{'BLOCK_T':>8} {'warps':>6} {'stages':>7} {'n_regs':>8} {'n_spills':>9} "
          f"{'smem':>8} {'smem估算':>9} {'regs估算':>9}")
    for r in rows:
        print(f"{r[0]:>8} {r[1]:>6} {r[2]:>7} {str(r[3]):>8} {str(r[4]):>9} "
              f"{str(r[5]):>8} {r[6]:>9} {r[7]:>9.0f}")
    return rows
