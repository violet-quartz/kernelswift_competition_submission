#!/usr/bin/env python3
"""spill 探针：warmup 编译一次 v1 kernel，不执行。供 bench/check_spill.py 通用驱动调用。

约定：模块级函数 warmup(dev) -> triton kernel（kernel.warmup(...) 的原始返回值，
不要在这里调 _init_handles()，通用驱动会调）。

H=1280 比 hc_split_sinkhorn 的 HC=4 大得多，BLOCK_H 又按 2 的幂上取整到 2048，
这里最值得关注的就是有没有 spill。
"""
import importlib.util
import sys
from pathlib import Path

import torch
import triton

OP_DIR = Path(__file__).resolve().parent


def _load_v1():
    path = OP_DIR / "v1" / "mhc_post.py"
    spec = importlib.util.spec_from_file_location("_ks_v1_mhc_post_spillprobe", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def warmup(dev):
    mod = _load_v1()
    kernel = mod._mhc_post_kernel

    x, residual, post_layer_mix, comb_res_mix = (
        t.to(dev) for t in mod.get_inputs()
    )

    # 照 forward() 里的口径把宿主侧参数摆好：形状、grid、BLOCK_H 都跟真实调用
    # 一致，只是不真的执行 —— warmup() 只编译，不落数据，可以放心传空张量。
    n0, n1, mhc_mult, h = residual.shape
    if not x.is_contiguous():
        x = x.contiguous()
    if not residual.is_contiguous():
        residual = residual.contiguous()
    if not post_layer_mix.is_contiguous():
        post_layer_mix = post_layer_mix.contiguous()
    if not comb_res_mix.is_contiguous():
        comb_res_mix = comb_res_mix.contiguous()

    output = torch.empty((n0, n1, mhc_mult, h), dtype=torch.bfloat16, device=dev)
    block_h = triton.next_power_of_2(h)

    return kernel.warmup(
        x, residual, post_layer_mix, comb_res_mix,
        output,
        H=h, BLOCK_H=block_h, M=mhc_mult,
        grid=(1,),
    )
