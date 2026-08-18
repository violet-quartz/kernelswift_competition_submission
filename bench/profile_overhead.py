#!/usr/bin/env python3
"""拆解 v1 每次 forward 的固定开销都花在哪。

为什么需要这个
--------------
hc_split_sinkhorn 在沐曦 C500 上实测 v1 = 0.109 ms，而 probe_loop_variants
测出 kernel 本体只要 0.023 ms —— 也就是说 **~86 µs 是 kernel 之外的开销**。

这个数字是所有 task 共用的分母：

    加速比 ≈ (v0 算子数 × 单算子派发成本) / (kernel 时间 + 固定开销)

分子由题目决定，改不了；分母里的固定开销是我们自己的，能改。算子数越少的题，
v0 总耗时越接近这个固定开销，加速比就越受它压制 —— 所以先测清楚这 86 µs
的构成，再决定优化哪一层。

口径
----
与 auto_bench.py L429-445 严格一致：每次调用后 sync，取 median，
warmup 200 / repeat 500。这样测出来的数字可以直接和 run.sh 的结果对齐。

用法:
    python3 bench/profile_overhead.py hc_split_sinkhorn
    python3 bench/profile_overhead.py hc_split_sinkhorn --repeat 200
    python3 bench/profile_overhead.py fused_moe --ver v2      # 对照另一个实现版本
"""
import argparse
import importlib.util
import statistics
import sys
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parent.parent


@triton.jit
def _noop_kernel(p):
    """什么都不做的 kernel，用来单独量出 Triton Python launcher 的成本。"""
    pass


@triton.autotune(configs=[triton.Config({}, num_warps=4)], key=[])
@triton.jit
def _noop_autotuned_kernel(p):
    """同上，但挂了 @triton.autotune（单 config，调优本身在 warmup 里就结束）。

    与 _noop_kernel 的差值 = Autotuner 包装层每次调用的净成本：构造 key、查 cache、
    再委派给 JITFunction。key=[] 时它不会重新 benchmark，但那几步 Python 仍然每次都走。
    如果这一项可观，"调完之后把赢家写死、摘掉 autotune" 就是一笔白捡的收益。
    """
    pass


def load_v1(name: str, ver: str = "v1"):
    # 每个算子一个文件夹：ROOT/<name>/<ver>/<name>.py
    # （早先仓库是扁平布局 ROOT/v1/<name>.py，这里跟着改过了，别改回去。）
    path = ROOT / name / ver / f"{name}.py"
    if not path.is_file():
        raise SystemExit(f"{path} 不存在")
    spec = importlib.util.spec_from_file_location(f"_ks_{ver}_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


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


def bench(fn, sync, warmup, repeat):
    """与 auto_bench.time_forward 同口径：每次调用后 sync，取 median（毫秒）。"""
    for _ in range(warmup):
        fn()
    sync()
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task")
    ap.add_argument("--ver", default="v1",
                    help="要profile 的实现版本目录名，默认 v1；对照用 --ver v2")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--repeat", type=int, default=500)
    args = ap.parse_args()

    dev_name, dev_mod = pick_device()
    dev = torch.device(dev_name)
    sync = dev_mod.synchronize

    mod = load_v1(args.task, args.ver)
    model = mod.ModelNew(*(mod.get_init_inputs() or []))
    if hasattr(model, "to"):
        model = model.to(dev)
    model.eval()
    inputs = [x.to(dev) if isinstance(x, torch.Tensor) else x
              for x in (mod.get_inputs() or [])]

    print(f"task={args.task}  ver={args.ver}  device={dev_name}  warmup={args.warmup} repeat={args.repeat}")
    print("口径与 auto_bench.py L429-445 一致（每次调用后 sync，取 median）\n")

    rows = []

    # ① 测量框架自身的成本：一个空 python 函数 + sync。
    #    这是任何实现都逃不掉的地板，v1 再快也不可能低于它。
    rows.append(("① 空函数 + sync（测量框架地板）", bench(lambda: None, sync, args.warmup, args.repeat)))

    # ② Triton Python launcher 的成本：一个什么都不干的 kernel。
    #    ②-① 就是 launcher 的净开销。这一项若占大头，优化方向是绕开
    #    Triton 的 Python 启动路径（缓存 CompiledKernel、复用 launch metadata）。
    dummy = torch.empty(1, device=dev)
    n_prog = 16
    rows.append(("② + 空 kernel launch",
                 bench(lambda: _noop_kernel[(n_prog,)](dummy), sync, args.warmup, args.repeat)))

    # ②b 两次连续 launch。②b-② 才是"多一个 kernel"的**边际**成本 ——
    #     launch 是异步的，第二次的 Python 开销可能与第一次的执行重叠，
    #     所以边际成本未必等于 ②-①。要不要把两个 kernel 合成一个，看的是这个差值。
    def two_launch():
        _noop_kernel[(n_prog,)](dummy)
        _noop_kernel[(n_prog,)](dummy)

    rows.append(("②b + 两次空 kernel launch", bench(two_launch, sync, args.warmup, args.repeat)))

    # ②c 挂了 autotune 的空 kernel。②c-② = Autotuner 包装层的每次调用成本。
    rows.append(("②c + 空 kernel launch（带 autotune）",
                 bench(lambda: _noop_autotuned_kernel[(n_prog,)](dummy),
                       sync, args.warmup, args.repeat)))

    # ③ 输出张量分配的成本。照 forward 里的实际形状分配。
    #    ③-① 就是 torch.empty 的净开销。这一项若占大头，优化方向是复用
    #    workspace（但要先想清楚跨调用复用输出缓冲在语义上是否可接受）。
    with torch.no_grad():
        outs = model.forward(*inputs)
    out_list = list(outs) if isinstance(outs, (tuple, list)) else [outs]
    shapes = [(tuple(o.shape), o.dtype) for o in out_list]
    print("输出张量:", ", ".join(f"{s}{str(d).replace('torch.','')}" for s, d in shapes), "\n")

    def alloc_all():
        for shp, dt in shapes:
            torch.empty(shp, device=dev, dtype=dt)

    rows.append((f"③ + {len(shapes)} 次 torch.empty",
                 bench(alloc_all, sync, args.warmup, args.repeat)))

    # ④ 完整 forward。④-③-② 大致就是真实 kernel 的执行时间 + 其余 python 开销。
    def full():
        with torch.no_grad():
            model.forward(*inputs)

    rows.append(("④ 完整 forward（= run.sh 里的 v1）", bench(full, sync, args.warmup, args.repeat)))

    w = max(len(r[0]) for r in rows) + 2
    print(f"{'层次':<{w}} {'耗时(ms)':>10} {'净增(µs)':>10}")
    print("-" * (w + 22))
    base = rows[0][1]
    prev = 0.0
    for i, (label, ms) in enumerate(rows):
        delta = (ms - (base if i else 0.0)) * 1e3 if i else ms * 1e3
        print(f"{label:<{w}} {ms:>10.4f} {delta:>10.1f}")
        prev = ms
    print("-" * (w + 22))

    floor, noop_k, alloc, full_ms = (r[1] for r in rows)
    print(f"\n解读:")
    print(f"  框架地板（空函数+sync）      : {floor * 1e3:>7.1f} µs   —— 不可压缩")
    print(f"  Triton launcher 净开销       : {(noop_k - floor) * 1e3:>7.1f} µs")
    print(f"  输出分配净开销               : {(alloc - floor) * 1e3:>7.1f} µs")
    print(f"  kernel 本体 + 其余 python    : "
          f"{(full_ms - noop_k - (alloc - floor)) * 1e3:>7.1f} µs   （粗略差值，仅供定位）")
    print(f"  完整 forward                 : {full_ms * 1e3:>7.1f} µs")
    print(f"\n  哪一项最大，就先优化哪一项。若 Triton launcher 占大头，")
    print(f"  这是所有 6 道题共用的分母，优化一次全盘受益。")


if __name__ == "__main__":
    main()
