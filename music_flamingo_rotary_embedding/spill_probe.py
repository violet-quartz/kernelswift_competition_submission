#!/usr/bin/env python3
"""spill 探针：warmup 编译一次 v1 kernel，不执行。供 bench/check_spill.py 通用驱动调用。

约定：模块级函数 warmup(dev) -> triton kernel（kernel.warmup(...) 的原始返回值，
不要在这里调 _init_handles()，通用驱动会调）。

为什么这道题需要这个探针
------------------------
grid 只有 (batch_size,) = 4 个 program，每个 program 要一口气吃下整个
(SEQ_LEN, DIM) = (32, 64) 的平面：`time_freqs` 本身就是 2048 个 float，加上
`batch_freqs`、两次三角函数的中间量、以及 sin/cos 各自的结果，同时活跃的
临时量是这个 kernel 唯一的真实风险。

**溢出不会报错，只会悄悄变慢** —— 寄存器装不下就换出到 local memory，
指标上看不出来，只有耗时不对。所以要在跑 run.sh 之前先查一次。

真溢出的话有两条退路，方向相反，都要实测过才知道哪条划算：
  * 把 grid 从 (batch_size,) 改成 (batch_size * seq_len,)，每个 program 只管
    一个 (b, t) 位置的 2*DIM 个通道 —— tile 从二维退化成一维，寄存器压力骤降，
    而且 angle 变成标量、连广播都不需要。
  * 保持 grid 不变，在 program 内部按 SEQ_LEN 分块循环，用更小的 BLOCK_S。
    注意这条会引入循环携带变量，沐曦的 make_ttgir 段错误对这个敏感。
"""
import importlib.util
import sys
from pathlib import Path

import torch

OP_DIR = Path(__file__).resolve().parent


def _load_v1():
    path = OP_DIR / "v1" / "music_flamingo_rotary_embedding.py"
    spec = importlib.util.spec_from_file_location("_ks_v1_mfre_spillprobe", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def warmup(dev):
    mod = _load_v1()
    kernel = mod._music_flamingo_rotary_embedding_forward_kernel

    # 照 forward() 的口径把宿主侧参数摆好：形状、grid、constexpr 都跟真实调用
    # 一致，只是不真的执行 —— warmup() 只编译、不落数据。
    #
    # 模型走 ModelNew(*get_init_inputs()) 构造再搬到 dev 上，跟 auto_bench.py
    # L526 的 `model_new.to(target_device)` 对齐：inv_freq / position_angles 是
    # register_buffer，要靠这一步才会跟着上卡。
    model = mod.ModelNew(*mod.get_init_inputs()).to(dev)
    timestamps, seq_len = mod.get_inputs()
    timestamps = timestamps.to(dev)

    batch_size = timestamps.shape[0]
    dim = model.inv_freq.shape[0] * 2

    freqs_sin = torch.empty(batch_size, seq_len, dim * 2,
                            device=dev, dtype=model.inv_freq.dtype)
    freqs_cos = torch.empty(batch_size, seq_len, dim * 2,
                            device=dev, dtype=model.inv_freq.dtype)

    return kernel.warmup(
        timestamps,
        model.inv_freq,
        model.position_angles,
        freqs_sin,
        freqs_cos,
        # 三个 constexpr 用关键字传：warmup() 对 constexpr 位置参数的绑定在不同
        # Triton 版本间行为不一，mhc_post/spill_probe.py 里这套写法在沐曦 3.0.0
        # 和昇腾 3.2.0 上都跑通过。
        SEQ_LEN=seq_len,
        DIM=dim,
        MAX_SEQ_LEN=model.max_seq_len,
        grid=(batch_size,),
    )
