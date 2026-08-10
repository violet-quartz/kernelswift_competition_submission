#!/usr/bin/env python3
"""spill 探针：warmup 编译一次 v1 kernel，不执行。供 bench/check_spill.py 通用驱动调用。

约定：模块级函数 warmup(dev) -> triton kernel（kernel.warmup(...) 的原始返回值，
不要在这里调 _init_handles()，通用驱动会调）。
"""
import importlib.util
import sys
from pathlib import Path

import torch

OP_DIR = Path(__file__).resolve().parent


def _load_v1():
    path = OP_DIR / "v1" / "hc_split_sinkhorn.py"
    spec = importlib.util.spec_from_file_location("_ks_v1_hc_split_sinkhorn_spillprobe", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def warmup(dev):
    mod = _load_v1()
    kernel = mod._hc_split_sinkhorn_kernel

    model = mod.ModelNew(*(mod.get_init_inputs() or [])).to(dev).eval()
    mixes, hc_scale, hc_base = (x.to(dev) for x in mod.get_inputs())

    # 照 forward() 里的口径把宿主侧参数摆好：形状、hc、grid 都跟真实调用一致，
    # 只是不真的执行 —— warmup() 只编译，不落数据，可以放心传空张量。
    b, s, mix_hc = mixes.shape
    hc = model.hc_mult
    n = b * s
    x = mixes if mixes.dtype == torch.float32 else mixes.float()
    if not x.is_contiguous():
        x = x.contiguous()
    base = hc_base if hc_base.dtype == torch.float32 else hc_base.float()
    pre = torch.empty((n, hc), device=dev, dtype=torch.float32)
    post = torch.empty((n, hc), device=dev, dtype=torch.float32)
    comb = torch.empty((n, hc, hc), device=dev, dtype=torch.float32)

    return kernel.warmup(
        x, hc_scale, base,
        pre, post, comb,
        EPS=model.eps,
        LOOP_ITERS=model.loop_iters,
        HC=hc,
        MIX=mix_hc,
        grid=(1,),
    )
