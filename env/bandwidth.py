#!/usr/bin/env python3
"""量这台机器的**可达访存带宽** —— 给"算子是不是撞到 roofline 了"提供参照物。

为什么需要这个
--------------
优化 memory-bound 算子时，唯一能回答"还有没有空间"的问题是：**我现在跑到了
可达带宽的百分之几？** 没有这个分母，speedup 数字再好看也不知道是撞墙了还是
只用了三成。mhc_post 就卡在这里过：昇腾上有厂商 aclnnAdd 的 691 GB/s 可以对照，
一眼看出 231 GB/s 还有 3x 空间；沐曦上没有任何参照值，382 GB/s 就悬着没法判读。

为什么不查 datasheet
--------------------
1. 标称 HBM 带宽是理论峰值，纯访存 kernel 一般只能吃到 70-85%，直接拿来当分母
   会系统性低估自己的算子。
2. **本项目的 C500 是切片卡**（mx-smi 显示 Compute 25% / Vram Quota 16000 MiB），
   整卡的标称值跟这个切片实际能用的带宽根本不是一回事。
3. 容器化环境里到底分到了什么卡、什么驱动，只有实测知道。

所以口径是：在**跑 benchmark 的同一台机器、同一个切片上**，用最简单的纯访存
torch 算子量一个上界。这个数字跟着 env.lock.txt 一起归档，换机器会自动重测。

测什么
------
    copy   fp32   c.copy_(a)             1读1写   可能走 DMA，见下
    scale  fp32   torch.mul(a,2.,out=c)  1读1写   确定走 kernel
    add    fp32   torch.add(a,b,out=c)   2读1写   跟昇腾 aclnnAdd 口径一致
    scale  bf16   同上                   1读1写
    add    bf16   同上                   2读1写

**读写比要跟被测算子对齐。** GPU 上写比读贵（write-allocate、ECC 的
read-modify-write），读写比不同，可达带宽能差 10-20%。mhc_post 的实际比例是
读 105.5 MB : 写 83.9 MB = 1.26:1，**离 copy/scale 的 1:1 远比离 add 的 2:1 近**，
所以对照 mhc_post 时应该看 scale 那一行，不是 add。脚本会自动按 --rw-ratio
标出最贴题的行。

**为什么 copy 和 scale 都要测。** 同 device、同 dtype、连续的 `copy_`，PyTorch
通常直接调 cudaMemcpyAsync(D2D)，可能由 DMA 拷贝引擎执行而不是 SM 发射的
load/store。你的算子走的是后者，所以 scale 才是口径一致的那个。两者的差值
正好告诉你 DMA 路径有没有额外优势。

**bf16 单独量**是因为字节数一样不代表快慢一样；如果某后端的 bf16 向量路径更慢，
只测 fp32 会得出偏乐观的分母。

口径细节
--------
  * **平台期检查**：同一个用例在多个张量尺寸上测，带宽应随尺寸增大趋于平坦。
    还在单调上升说明没跑到饱和（被 cache 兜住、或 launch 开销占比过大），
    这时候报出来的是**下界**而不是可达带宽。切片卡上尤其需要这一步 ——
    L2 是否也被切分没有先验答案，多大算"远大于 cache"只能靠平台期判断。
  * 计时包住整个 repeat 循环再除以次数，把 launch 开销摊掉；跑 rounds 轮取**最快**
    的一轮（要的是"能达到多少"的上界，不是平均表现）。
  * GB/s 一律按 10^9 字节算，跟厂商标称和 profiling 工具的口径一致（不是 GiB/s）。

用法
----
    python3 env/bandwidth.py                    # 默认带平台期扫描
    python3 env/bandwidth.py --no-sweep         # 只测最大尺寸，快
    python3 env/bandwidth.py --sizes 128,256,512,1024
    python3 env/bandwidth.py --rw-ratio 1.26    # 按被测算子的读写比标注贴题行

也被 env/selftest.py 的步骤 ⑦ 调用，结果随 env/capture.sh 落进 env.lock.txt。
"""
import argparse
import importlib
import sys
import time

import torch

# 报告里给一条换算参考：搬这么多字节的算子，在峰值带宽下的耗时下限是多少。
# 189.4 MB 是 mhc_post 每次 forward 的固定访存量（读 x 20.97 + residual 83.89
# + 两个小张量 0.66 = 105.52，写 output 83.89），拿它当例子是因为它是本仓库
# 目前最 memory-bound 的算子。
_EXAMPLE_TASK = "mhc_post"
_EXAMPLE_READ_MB = 105.52
_EXAMPLE_WRITE_MB = 83.89
_EXAMPLE_BYTES = (_EXAMPLE_READ_MB + _EXAMPLE_WRITE_MB) * 1e6
_EXAMPLE_RW = _EXAMPLE_READ_MB / _EXAMPLE_WRITE_MB      # 1.26

# 平台期判据：相邻两档带宽相对差小于这个值，就认为已经饱和
_PLATEAU_TOL = 0.03


def bootstrap():
    """昇腾/寒武纪要先 import 扩展，torch.npu / torch.mlu 才会存在。沐曦走 torch.cuda。"""
    for m in ("torch_npu", "torch_mlu"):
        try:
            importlib.import_module(m)
        except ImportError:
            pass


def pick_device():
    """跟 bench/profile_chip.py 的 pick_device() 同一套探测顺序。"""
    for name in ("gcu", "cuda", "npu", "mlu"):
        mod = getattr(torch, name, None)
        if mod is None:
            continue
        try:
            if mod.is_available():
                return name, mod
        except Exception:
            pass
    return None, None


# (标签, 读数, 写数, 需要几个输入张量, 是否确定走 kernel, thunk 工厂)
#   * 读写系数分开记，才能按读写比匹配被测算子；总系数 = reads + writes
#   * kernel_path=False 表示可能被 runtime 转成 DMA（copy_ 就是），
#     选对照行时会被降级 —— 你的算子走的是 SM 发射的 load/store，口径要一致
CASES = (
    ("copy",  1, 1, 1, False, lambda a, b, c: (lambda: c.copy_(a))),
    ("scale", 1, 1, 1, True,  lambda a, b, c: (lambda: torch.mul(a, 2.0, out=c))),
    ("add",   2, 1, 2, True,  lambda a, b, c: (lambda: torch.add(a, b, out=c))),
)


def _best_seconds(fn, mod, repeat: int, rounds: int) -> float:
    """跑 rounds 轮、每轮 repeat 次，返回最快一轮的单次耗时（秒）。

    同步放在循环外：这些 kernel 单次只有零点几毫秒，每次都 synchronize 的话
    同步开销本身就能占掉可观比例，量出来的带宽会偏低。
    """
    best = float("inf")
    for _ in range(rounds):
        mod.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeat):
            fn()
        mod.synchronize()
        best = min(best, (time.perf_counter() - t0) / repeat)
    return best


def _alloc(n, dtype, dev, need_b):
    """分配张量；OOM 时返回 None（调用方负责降档）。

    失败路径上显式置 None：原版在 a 分配成功、b 失败时会把 a 泄漏到下一轮，
    切片卡显存本来就紧，重试几次就真的不够了。
    """
    a = b = c = None
    try:
        a = torch.randn(n, dtype=dtype, device=dev)
        b = torch.randn(n, dtype=dtype, device=dev) if need_b else None
        c = torch.empty(n, dtype=dtype, device=dev)
        return a, b, c
    except RuntimeError as exc:
        del a, b, c
        if "out of memory" not in str(exc).lower():
            raise
        return None


def measure_one(dev_name, mod, case, dtype, mib, warmup, repeat, rounds):
    """测单个 (用例, dtype, 尺寸)，返回 (moved_bytes, secs, gbps) 或 None。"""
    label, reads, writes, need_b, _kernel_path, make_fn = case
    dev = torch.device(dev_name)
    itemsize = torch.empty((), dtype=dtype).element_size()
    n = (mib * 1024 * 1024) // itemsize

    bufs = _alloc(n, dtype, dev, need_b)
    if bufs is None:
        return None
    a, b, c = bufs

    try:
        fn = make_fn(a, b, c)
        for _ in range(warmup):
            fn()
        secs = _best_seconds(fn, mod, repeat, rounds)
        moved = (reads + writes) * n * itemsize
        return moved, secs, moved / secs / 1e9
    finally:
        del a, b, c


def sweep(dev_name, mod, sizes, dtypes, warmup, repeat, rounds):
    """在多个尺寸上扫每个用例，返回 {(label, dtype_name): [(mib, gbps), ...]}。"""
    out = {}
    for case in CASES:
        for dtype in dtypes:
            key = (case[0], str(dtype).replace("torch.", ""))
            series = []
            for mib in sizes:
                r = measure_one(dev_name, mod, case, dtype, mib, warmup, repeat, rounds)
                if r is None:
                    print(f"  {key[0]}/{key[1]} @ {mib} MiB: 显存不足，跳过")
                    continue
                series.append((mib, r[1], r[2]))
            if series:
                out[key] = series
    return out


def check_plateau(series):
    """判断带宽是否已进入平台期。返回 (是否饱和, 最后两档的相对差)。"""
    if len(series) < 2:
        return None, None
    g1, g2 = series[-2][2], series[-1][2]
    rel = (g2 - g1) / g1
    return abs(rel) < _PLATEAU_TOL, rel


def report(results, rw_ratio, ref_dtype="bfloat16"):
    """打印成 env.lock.txt 里可读、可 grep 的样子；返回 (峰值 GB/s, 贴题值 GB/s)。"""
    if not results:
        print("  (没量到任何一行)")
        return 0.0, 0.0

    # 表头用 ASCII：CJK 在 len() 里算 1 但终端里占 2 格，中文表头对齐会整列歪掉
    print()
    print(f"  {'op':<7} {'dtype':<10} {'r:w':>5} {'size':>9} "
          f"{'per-call':>12} {'bandwidth':>13}  {'plateau':>8}")

    peak = 0.0
    candidates = []                        # (排序键, key, gbps, ratio, kernel_path)
    warnings = []

    for (label, dtype), series in sorted(results.items()):
        case = next(c for c in CASES if c[0] == label)
        reads, writes, kernel_path = case[1], case[2], case[4]
        ratio = reads / writes
        saturated, rel = check_plateau(series)

        for i, (mib, secs, gbps) in enumerate(series):
            last = (i == len(series) - 1)
            if last and saturated is not None:
                tag = "yes" if saturated else f"{rel * 100:+.0f}%"
            else:
                tag = ""
            print(f"  {label if i == 0 else '':<7} {dtype if i == 0 else '':<10} "
                  f"{f'{reads}:{writes}' if i == 0 else '':>5} {mib:>6} MiB "
                  f"{secs * 1e3:>9.4f} ms {gbps:>8.1f} GB/s  {tag:>8}")

        gbps = series[-1][2]
        peak = max(peak, gbps)
        if saturated is False:
            # rel 是正是负都算没饱和，但成因不同：还在涨说明尺寸不够大；
            # 还在跌通常是小尺寸那档被 cache 兜住、虚高，正在往真值收敛。
            trend = "仍在上升" if rel > 0 else "仍在下降"
            warnings.append(f"{label}/{dtype} {trend} ({rel * 100:+.0f}%)")

        # 挑对照行的优先级：dtype 对上 > 确定走 kernel > 读写比接近。
        # dtype 排第一位是因为 bf16 和 fp32 的向量路径可能快慢不同；
        # kernel_path 排第二位是因为 copy_ 可能走 DMA，跟你的算子不是一条路。
        # 元组只放排序键 / 行标识 / 带宽三项，其余信息在打印时按 label 回查 CASES。
        candidates.append((
            (0 if dtype == ref_dtype else 1,
             0 if kernel_path else 1,
             abs(ratio - rw_ratio)),
            (label, dtype), gbps,
        ))

    print()
    print(f"  峰值可达带宽: {peak:.1f} GB/s")

    best_match = min(candidates, key=lambda t: t[0]) if candidates else None
    if best_match:
        _, (label, dtype), gbps = best_match
        # 说明必须**据实生成**，不能写死：min() 挑的是现有档里最好的那个，不保证
        # 三个条件都满足（比如 --ref-dtype 传了根本没测的 dtype，或只有 copy 可选）。
        # 这段会被 capture.sh 归档进 env.lock.txt，在那里留一句假话比不留更糟。
        case = next(c for c in CASES if c[0] == label)
        ratio = case[1] / case[2]
        notes = [
            f"dtype 匹配 {ref_dtype}" if dtype == ref_dtype
            else f"⚠ dtype 是 {dtype}，被测算子是 {ref_dtype}",
            "走 kernel 路径" if case[4]
            else "⚠ 可能走 DMA，跟 SM 发射的 load/store 不是一条路",
            f"读写比 {ratio:.2f}:1 对算子的 {rw_ratio:.2f}:1",
        ]
        sat, _rel = check_plateau(results[(label, dtype)])
        if sat is False:
            notes.append("⚠ 未进入平台期，这是下界")
        elif sat is None:
            notes.append("⚠ 只测了一档，无法判断是否饱和")

        print(f"  对照行: {label}/{dtype} = {gbps:.1f} GB/s")
        for n in notes:
            print(f"          {n}")
        print(f"          **算利用率用这一行做分母，不是峰值**")

    if warnings:
        print()
        print("  ⚠ 以下用例尚未进入平台期，报出的是**下界**不是可达带宽：")
        for w in warnings:
            print(f"      {w}")
        print("    加大 --sizes 的最大档位重测，或在结论里注明未饱和。")

    ref = best_match[2] if best_match else peak
    print()
    print(f"  换算参考: {_EXAMPLE_TASK} 每次 forward 搬 {_EXAMPLE_BYTES / 1e6:.1f} MB "
          f"(读 {_EXAMPLE_READ_MB:.1f} / 写 {_EXAMPLE_WRITE_MB:.1f})，")
    print(f"            按 {ref:.1f} GB/s 算，耗时下限约 "
          f"{_EXAMPLE_BYTES / ref / 1e9 * 1e3:.4f} ms")
    return peak, ref


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="64,128,256,512",
                    help="张量尺寸档位（MiB，逗号分隔）。必须扫到带宽趋平为止")
    ap.add_argument("--no-sweep", action="store_true",
                    help="只测最大那一档，快但无法判断是否饱和")
    ap.add_argument("--rw-ratio", type=float, default=_EXAMPLE_RW,
                    help=f"被测算子的读:写比，用来挑最贴题的对照行（默认 "
                         f"{_EXAMPLE_RW:.2f}，即 {_EXAMPLE_TASK}）")
    ap.add_argument("--dtypes", default="float32,bfloat16")
    ap.add_argument("--ref-dtype", default="bfloat16",
                    help="被测算子用的 dtype，挑对照行时优先匹配它"
                         f"（默认 bfloat16，即 {_EXAMPLE_TASK}）")
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    if args.no_sweep:
        sizes = sizes[-1:]
    dtypes = [getattr(torch, d.strip()) for d in args.dtypes.split(",") if d.strip()]

    bootstrap()
    name, mod = pick_device()
    if name is None:
        print("探测不到加速器，跳过带宽测量")
        return 0

    dev_desc = mod.get_device_name(0) if hasattr(mod, "get_device_name") else "?"
    print(f"设备: {name} ({dev_desc})")
    print(f"尺寸档位: {sizes} MiB   dtype: {[str(d).replace('torch.', '') for d in dtypes]}")

    results = sweep(name, mod, sizes, dtypes, args.warmup, args.repeat, args.rounds)
    report(results, args.rw_ratio, args.ref_dtype)
    return 0


if __name__ == "__main__":
    sys.exit(main())