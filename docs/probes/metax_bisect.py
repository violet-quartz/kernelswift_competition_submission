#!/usr/bin/env python3
"""沐曦 Triton 后端崩溃特性二分定位

背景：selftest 的 _smoke2d 在 metax/compiler.py 的 make_ttgir 里段错误。
那个 kernel 一次性用了 6~7 个特性，需要逐个拆开确认是哪个触发的。

用法：
    python3 metax_bisect.py                 # 跑全部用例
    python3 metax_bisect.py --case 07       # 单独跑某个用例（调试用）
    python3 metax_bisect.py --list          # 只列出用例

关键设计：每个用例在**独立子进程**里跑。段错误会直接杀进程，
放在同一进程里第一个崩溃就没有后续结果了。

另：@triton.jit 的代码必须在真实文件里（inspect 要取源码），
所以这里用「同一个文件 + --case 自分发」而不是动态 exec。
"""
import argparse
import subprocess
import sys
import os
import triton
import triton.language as tl

HC = 16          # tile 边长，2 的幂
ITERS = 4        # 循环次数


# =============================================================================
# 用例定义：每个只比前一个多一个特性
# =============================================================================
def case_01_load_store_2d():
    """2D tile：广播构造下标 + load + store（无任何规约）"""
    import torch, triton, triton.language as tl

    @triton.jit
    def k(p, o, HC: tl.constexpr):
        i = tl.arange(0, HC)[:, None]
        j = tl.arange(0, HC)[None, :]
        off = i * HC + j
        tl.store(o + off, tl.load(p + off))
    return k, ('plain',)


def case_02_exp_2d():
    """2D tile + tl.exp"""
    import torch, triton, triton.language as tl

    @triton.jit
    def k(p, o, HC: tl.constexpr):
        i = tl.arange(0, HC)[:, None]
        j = tl.arange(0, HC)[None, :]
        off = i * HC + j
        tl.store(o + off, tl.exp(tl.load(p + off)))
    return k, ('plain',)


def case_03_reduce_axis1():
    """沿 axis=1 规约，结果写成 1D（不广播回去）"""
    import torch, triton, triton.language as tl

    @triton.jit
    def k(p, o, HC: tl.constexpr):
        i = tl.arange(0, HC)[:, None]
        j = tl.arange(0, HC)[None, :]
        t = tl.load(p + i * HC + j)
        tl.store(o + tl.arange(0, HC), tl.sum(t, axis=1))
    return k, ('vec',)


def case_04_reduce_axis0():
    """沿 axis=0 规约（列方向，通常比 axis=1 更容易出问题）"""
    import torch, triton, triton.language as tl

    @triton.jit
    def k(p, o, HC: tl.constexpr):
        i = tl.arange(0, HC)[:, None]
        j = tl.arange(0, HC)[None, :]
        t = tl.load(p + i * HC + j)
        tl.store(o + tl.arange(0, HC), tl.sum(t, axis=0))
    return k, ('vec',)


def case_05_reduce_bcast_axis1():
    """axis=1 规约后 [:, None] 广播回 2D"""
    import torch, triton, triton.language as tl

    @triton.jit
    def k(p, o, HC: tl.constexpr):
        i = tl.arange(0, HC)[:, None]
        j = tl.arange(0, HC)[None, :]
        off = i * HC + j
        t = tl.load(p + off)
        tl.store(o + off, t / (tl.sum(t, axis=1)[:, None] + 1e-6))
    return k, ('plain',)


def case_06_reduce_bcast_axis0():
    """axis=0 规约后 [None, :] 广播回 2D"""
    import torch, triton, triton.language as tl

    @triton.jit
    def k(p, o, HC: tl.constexpr):
        i = tl.arange(0, HC)[:, None]
        j = tl.arange(0, HC)[None, :]
        off = i * HC + j
        t = tl.load(p + off)
        tl.store(o + off, t / (tl.sum(t, axis=0)[None, :] + 1e-6))
    return k, ('plain',)


def case_07_range_loop_carried_2d():
    """range 循环，2D tile 作为 loop-carried 变量  ← 头号嫌疑"""
    import torch, triton, triton.language as tl

    @triton.jit
    def k(p, o, ITERS: tl.constexpr, HC: tl.constexpr):
        i = tl.arange(0, HC)[:, None]
        j = tl.arange(0, HC)[None, :]
        off = i * HC + j
        t = tl.load(p + off)
        for _ in range(ITERS):
            t = t * 1.001
        tl.store(o + off, t)
    return k, ('iters',)


def case_08_static_range_loop_carried_2d():
    """同上，但用 tl.static_range（完全展开，不生成 scf.for）"""
    import torch, triton, triton.language as tl

    @triton.jit
    def k(p, o, ITERS: tl.constexpr, HC: tl.constexpr):
        i = tl.arange(0, HC)[:, None]
        j = tl.arange(0, HC)[None, :]
        off = i * HC + j
        t = tl.load(p + off)
        for _ in tl.static_range(ITERS):
            t = t * 1.001
        tl.store(o + off, t)
    return k, ('iters',)


def case_09_range_loop_with_reduction():
    """range 循环 + 循环体内双向规约（= _smoke2d 的核心结构）"""
    import torch, triton, triton.language as tl

    @triton.jit
    def k(p, o, ITERS: tl.constexpr, HC: tl.constexpr):
        i = tl.arange(0, HC)[:, None]
        j = tl.arange(0, HC)[None, :]
        off = i * HC + j
        t = tl.load(p + off)
        for _ in range(ITERS):
            t = t / (tl.sum(t, axis=1)[:, None] + 1e-6)
            t = t / (tl.sum(t, axis=0)[None, :] + 1e-6)
        tl.store(o + off, t)
    return k, ('iters',)


def case_10_static_range_with_reduction():
    """同上，改用 tl.static_range  ← 头号解决方案候选"""
    import torch, triton, triton.language as tl

    @triton.jit
    def k(p, o, ITERS: tl.constexpr, HC: tl.constexpr):
        i = tl.arange(0, HC)[:, None]
        j = tl.arange(0, HC)[None, :]
        off = i * HC + j
        t = tl.load(p + off)
        for _ in tl.static_range(ITERS):
            t = t / (tl.sum(t, axis=1)[:, None] + 1e-6)
            t = t / (tl.sum(t, axis=0)[None, :] + 1e-6)
        tl.store(o + off, t)
    return k, ('iters',)


def case_11_max_axis1_bcast():
    """tl.max(axis=1) 后广播（_smoke2d 里 softmax 那一步）"""
    import torch, triton, triton.language as tl

    @triton.jit
    def k(p, o, HC: tl.constexpr):
        i = tl.arange(0, HC)[:, None]
        j = tl.arange(0, HC)[None, :]
        off = i * HC + j
        t = tl.load(p + off)
        tl.store(o + off, tl.exp(t - tl.max(t, axis=1)[:, None]))
    return k, ('plain',)


def case_12_full_smoke2d():
    """完整复现 selftest 里的 _smoke2d"""
    import torch, triton, triton.language as tl

    @triton.jit
    def k(p, o, ITERS: tl.constexpr, HC: tl.constexpr):
        i = tl.arange(0, HC)[:, None]
        j = tl.arange(0, HC)[None, :]
        off = i * HC + j
        t = tl.load(p + off)
        t = tl.exp(t - tl.max(t, axis=1)[:, None])
        for _ in range(ITERS):
            t = t / (tl.sum(t, axis=1)[:, None] + 1e-6)
            t = t / (tl.sum(t, axis=0)[None, :] + 1e-6)
        tl.store(o + off, t)
    return k, ('iters',)


CASES = {
    '01': case_01_load_store_2d,
    '02': case_02_exp_2d,
    '03': case_03_reduce_axis1,
    '04': case_04_reduce_axis0,
    '05': case_05_reduce_bcast_axis1,
    '06': case_06_reduce_bcast_axis0,
    '07': case_07_range_loop_carried_2d,
    '08': case_08_static_range_loop_carried_2d,
    '09': case_09_range_loop_with_reduction,
    '10': case_10_static_range_with_reduction,
    '11': case_11_max_axis1_bcast,
    '12': case_12_full_smoke2d,
}


# =============================================================================
# 单个用例的执行（在子进程里被调用）
# =============================================================================
def run_one(name):
    import faulthandler
    faulthandler.enable()
    import torch

    # 找加速器，逻辑同 selftest
    dev_name = None
    for n in ("gcu", "cuda", "npu", "mlu"):
        m = getattr(torch, n, None)
        if m is not None:
            try:
                if m.is_available():
                    dev_name = n
                    break
            except Exception:
                pass
    if dev_name is None:
        print("no accelerator")
        return 2

    dev = torch.device(dev_name)
    kernel, sig = CASES[name]()

    p = torch.randn(HC, HC, device=dev)
    if sig[0] == 'vec':
        o = torch.empty(HC, device=dev)
    else:
        o = torch.empty(HC, HC, device=dev)

    if sig[0] == 'iters':
        kernel[(1,)](p, o, ITERS=ITERS, HC=HC)
    else:
        kernel[(1,)](p, o, HC=HC)
    getattr(torch, dev_name).synchronize()

    if not torch.isfinite(o).all():
        print("compiled+ran, but output has nan/inf")
        return 3
    return 0


# =============================================================================
# 驱动：把每个用例丢进独立子进程
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case')
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()

    if args.list:
        for n, f in sorted(CASES.items()):
            print(f"  {n}  {f.__doc__.splitlines()[0]}")
        return 0

    if args.case:
        return run_one(args.case)

    print(f"{'用例':<6} {'结果':<12} 说明")
    print("-" * 74)
    first_fail = None
    for n, f in sorted(CASES.items()):
        desc = f.__doc__.splitlines()[0]
        r = subprocess.run([sys.executable, '-X', 'faulthandler', os.path.abspath(__file__),
                            '--case', n],
                           capture_output=True, text=True, timeout=600)
        if r.returncode == 0:
            status = "通过"
        elif r.returncode < 0 or r.returncode == 139:
            status = "段错误 ✗"
            first_fail = first_fail or n
        elif r.returncode == 3:
            status = "数值异常 ✗"
            first_fail = first_fail or n
        elif r.returncode == 2:
            status = "无加速器"
        else:
            status = "异常 ✗"
            first_fail = first_fail or n
        print(f"{n:<6} {status:<12} {desc}")
        if status.endswith("✗"):
            tail = (r.stderr or r.stdout).strip().splitlines()
            for line in tail[-6:]:
                print(f"       | {line}")

    print("-" * 74)
    if first_fail:
        print(f"最早失败的用例: {first_fail}")
        print("对比它和前一个通过的用例，两者的差异就是触发编译器 bug 的特性。")
        print("若 09 失败而 10 通过 -> 换 tl.static_range 即可绕过。")
    else:
        print("全部通过 —— 说明触发条件比这些用例更特殊，尝试调大 HC 或 ITERS。")
    return 0


if __name__ == "__main__":
    sys.exit(main())