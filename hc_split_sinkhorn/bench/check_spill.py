#!/usr/bin/env python3
"""检查 v1 kernel 是否寄存器溢出。

v1 文件头声称"整个 HC×HC 矩阵全程留在寄存器里"，但那是设计意图，不是实测。
这里用 kernel.warmup() 只编译、不执行，读出编译器实际分配的寄存器数：

    n_regs   : 每个 program 用了多少寄存器
    n_spills : 溢出到 local memory（其实还是显存）的寄存器数 —— > 0 就是性能灾难
    smem     : shared memory 占用（字节）

跟 auto_bench.py 的计时口径无关，纯一次性诊断，不产出 results/ 下任何文件。

用法:
    python3 bench/check_spill.py hc_split_sinkhorn
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def pick_device():
    for n in ("gcu", "cuda", "npu", "mlu"):
        m = getattr(torch, n, None)
        if m is None:
            continue
        try:
            if m.is_available():
                return n, m
        except Exception:
            pass
    raise SystemExit("no accelerator")


def load_v1(name: str):
    path = ROOT / "v1" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_ks_v1_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("task")
    args = ap.parse_args()

    dev_name, _ = pick_device()
    dev = torch.device(dev_name)

    mod = load_v1(args.task)
    kernel_name = f"_{args.task}_kernel"
    kernel = getattr(mod, kernel_name, None)
    if kernel is None:
        raise SystemExit(
            f"{mod.__file__} 里找不到 {kernel_name}，按实际 kernel 名调整脚本里的 kernel_name"
        )

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

    k = kernel.warmup(
        x, hc_scale, base,
        pre, post, comb,
        EPS=model.eps,
        LOOP_ITERS=model.loop_iters,
        HC=hc,
        MIX=mix_hc,
        grid=(1,),
    )
    k._init_handles()

    print(f"task={args.task}  device={dev_name}")
    print(f"n_regs={k.n_regs}  n_spills={k.n_spills}  smem={k.metadata.shared}")
    if k.n_spills:
        print(
            "\n⚠ n_spills > 0 —— 寄存器装不下，溢出到 local memory，是性能灾难，"
            "需要回去改 kernel（减小同时活跃的临时变量、或拆分计算）。"
        )
    else:
        print("\n✅ 无寄存器溢出，\"全程留在寄存器里\" 的设计意图得到实测确认。")


if __name__ == "__main__":
    main()
