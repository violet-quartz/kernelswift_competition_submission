#!/usr/bin/env python3
"""确认 tl.static_range 方案在真实规模下是否可用

背景：metax_bisect 定位出「range 循环体内做规约 + 2D loop-carried」会让
沐曦后端的 make_ttgir 段错误，改用 tl.static_range 可绕过（用例 10 通过）。
但 static_range 会完全展开循环，代价随 ITERS 线性上升。

这个脚本在真实的 (HC, ITERS) 组合上测：能不能编译、编译要多久、数值对不对。

用法:
    python3 static_range_scale.py                  # 跑全部组合
    python3 static_range_scale.py --case 16x50     # 单跑一个（调试用）
"""
import argparse
import subprocess
import sys
import os
import time
import triton
import triton.language as tl

# 按你的真实场景改这两行
HC_LIST = [8, 16, 32, 64]
ITERS_LIST = [10, 20, 50, 100]


def build_and_run(HC, ITERS):
    import faulthandler
    faulthandler.enable()
    import torch, triton, triton.language as tl

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

    @triton.jit
    def sinkhorn(p_ptr, o_ptr, ITERS: tl.constexpr, HC: tl.constexpr):
        i = tl.arange(0, HC)[:, None]
        j = tl.arange(0, HC)[None, :]
        off = i * HC + j
        t = tl.load(p_ptr + off)
        t = tl.exp(t - tl.max(t, axis=1)[:, None])
        for _ in tl.static_range(ITERS):                 # ← 关键：static_range
            t = t / (tl.sum(t, axis=1)[:, None] + 1e-6)
            t = t / (tl.sum(t, axis=0)[None, :] + 1e-6)
        tl.store(o_ptr + off, t)

    p = torch.randn(HC, HC, device=dev)
    o = torch.empty_like(p)

    t0 = time.perf_counter()
    sinkhorn[(1,)](p, o, ITERS=ITERS, HC=HC)             # 首次调用含编译
    getattr(torch, dev_name).synchronize()
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(20):                                   # 第二次起走缓存
        sinkhorn[(1,)](p, o, ITERS=ITERS, HC=HC)
    getattr(torch, dev_name).synchronize()
    run_ms = (time.perf_counter() - t0) / 20 * 1e3

    if not torch.isfinite(o).all():
        print(f"COMPILE_S={compile_s:.2f} RUN_MS={run_ms:.3f} NAN")
        return 3
    # Sinkhorn 收敛后行和列和都该接近 1
    col = o.sum(dim=0)
    row = o.sum(dim=1)
    cerr = (col - 1).abs().max().item()
    rerr = (row - 1).abs().max().item()
    print(f"COMPILE_S={compile_s:.2f} RUN_MS={run_ms:.3f} COLERR={cerr:.2e} ROWERR={rerr:.2e}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case')
    args = ap.parse_args()

    if args.case:
        hc, it = args.case.split('x')
        return build_and_run(int(hc), int(it))

    print(f"{'HC':>5} {'ITERS':>7} {'结果':<10} {'编译(s)':>9} {'执行(ms)':>10}  收敛误差")
    print("-" * 72)
    for HC in HC_LIST:
        for ITERS in ITERS_LIST:
            r = subprocess.run(
                [sys.executable, '-X', 'faulthandler', os.path.abspath(__file__),
                 '--case', f'{HC}x{ITERS}'],
                capture_output=True, text=True, timeout=1800)
            out = r.stdout.strip()
            if r.returncode == 0 and 'COMPILE_S' in out:
                kv = dict(p.split('=') for p in out.split() if '=' in p)
                print(f"{HC:>5} {ITERS:>7} {'通过':<10} {float(kv['COMPILE_S']):>9.2f} "
                      f"{float(kv['RUN_MS']):>10.3f}  列 {kv['COLERR']} 行 {kv['ROWERR']}")
            elif r.returncode < 0 or r.returncode == 139:
                print(f"{HC:>5} {ITERS:>7} {'段错误 ✗':<10}  <- static_range 在这个规模也顶不住")
            elif r.returncode == 3:
                print(f"{HC:>5} {ITERS:>7} {'数值异常 ✗':<10}")
            elif r.returncode == 2:
                print("无加速器")
                return 2
            else:
                tail = (r.stderr or '').strip().splitlines()[-2:]
                print(f"{HC:>5} {ITERS:>7} {'失败 ✗':<10}  {' | '.join(tail)}")
    print("-" * 72)
    print("看编译时间随 ITERS 的增长：接近线性属正常，突然爆炸说明展开到了极限。")
    print("若大 ITERS 撑不住，改用 u/v 向量形式的 Sinkhorn（循环只携带 1D，不携带 2D tile）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())