#!/usr/bin/env python3
"""在真实配置 (HC=4, ITERS=19) 上比较 Sinkhorn 循环的几种写法。

背景
----
metax_bisect 定位出：沐曦 Triton 3.0.0 的 make_ttgir 在「range 循环（scf.for）
体内做规约」时段错误（用例 09 崩、10 用 tl.static_range 通过）。
static_range 完全展开可绕过，但 HC=8/ITERS=10 冷编译就要 209 秒。

这个脚本要回答的问题是：**编译器崩的到底是哪一个组合？**

  (a) 「循环体内有任何规约」    -> 只能靠 static_range，认了
  (b) 「2D tile 作为 loop-carried 变量 + 规约」-> 换成只携带 1D 就能绕过

如果是 (b)，就能用 u/v 形式：把
    C <- C / (rowsum(C) + eps);  C <- C / (colsum(C) + eps)
改写成对角缩放 C_k = diag(u) · C0 · diag(v)，其中

    R_i = sum_j C0[i,j] * v_j        u <- u / (u * R + eps)
    S_j = sum_i C0[i,j] * u_i        v <- v / (v * S + eps)

这是**恒等变换而非近似**（推导见下），C0 在循环外算一次，循环只携带两个
长度 HC 的 1D 向量，编译量与 ITERS 无关。

推导:
  设 C = diag(u)·C0·diag(v)，则
    rowsum_i(C) = sum_j u_i·C0[i,j]·v_j = u_i · R_i,  R_i = sum_j C0[i,j]·v_j
    C/(rowsum+eps) 的 [i,j] 元 = u_i·C0[i,j]·v_j / (u_i·R_i + eps)
                              = (u_i/(u_i·R_i+eps)) · C0[i,j] · v_j
    => u <- u/(u·R + eps)，结构保持
  列方向同理（注意 S 要用更新后的 u）。

用法:
    python3 env/metax-c500/probe_loop_variants.py             # 跑全部变体
    python3 env/metax-c500/probe_loop_variants.py --case uv_range
"""
import argparse
import os
import subprocess
import sys
import time
import triton
import triton.language as tl

# v1/hc_split_sinkhorn.py 的真实配置：hc_mult=4，循环体跑 sinkhorn_iters-1 = 19 轮
HC = 4
ITERS = 19
EPS = 1e-6

VARIANTS = {
    "2d_static":      "现方案: 2D loop-carried + tl.static_range（已知可行，测真实配置的编译代价）",
    "2d_range_1axis": "诊断: 2D loop-carried + range，体内只做单向规约",
    "uv_range":       "候选: 1D loop-carried (u,v) + range          <- 最想要的",
    "uv_static":      "对照: 1D loop-carried (u,v) + tl.static_range",
}


def build_and_run(case):
    import faulthandler
    faulthandler.enable()

    import torch
    import triton
    import triton.language as tl

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

    # ---- 四个变体 ----------------------------------------------------------
    @triton.jit
    def k_2d_static(c_ptr, o_ptr, ITERS: tl.constexpr, HC: tl.constexpr, EPS: tl.constexpr):
        i = tl.arange(0, HC)[:, None]
        j = tl.arange(0, HC)[None, :]
        off = i * HC + j
        t = tl.load(c_ptr + off)
        for _ in tl.static_range(ITERS):
            t = t / (tl.sum(t, axis=1)[:, None] + EPS)
            t = t / (tl.sum(t, axis=0)[None, :] + EPS)
        tl.store(o_ptr + off, t)

    @triton.jit
    def k_2d_range_1axis(c_ptr, o_ptr, ITERS: tl.constexpr, HC: tl.constexpr, EPS: tl.constexpr):
        # 诊断用：只做行规约。它若也崩，说明问题是"循环体内有任何规约"；
        # 它若不崩而双向崩，说明是两个方向的 layout 冲突。数值不参与校验。
        i = tl.arange(0, HC)[:, None]
        j = tl.arange(0, HC)[None, :]
        off = i * HC + j
        t = tl.load(c_ptr + off)
        for _ in range(ITERS):
            t = t / (tl.sum(t, axis=1)[:, None] + EPS)
        tl.store(o_ptr + off, t)

    @triton.jit
    def k_uv_range(c_ptr, o_ptr, ITERS: tl.constexpr, HC: tl.constexpr, EPS: tl.constexpr):
        i = tl.arange(0, HC)[:, None]
        j = tl.arange(0, HC)[None, :]
        off = i * HC + j
        c0 = tl.load(c_ptr + off)                    # loop-invariant，循环外只读
        u = tl.zeros([HC], dtype=tl.float32) + 1.0
        v = tl.zeros([HC], dtype=tl.float32) + 1.0
        for _ in range(ITERS):                       # loop-carried 只有 u、v 两个 1D
            r = tl.sum(c0 * v[None, :], axis=1)
            u = u / (u * r + EPS)
            s = tl.sum(c0 * u[:, None], axis=0)
            v = v / (v * s + EPS)
        tl.store(o_ptr + off, u[:, None] * c0 * v[None, :])

    @triton.jit
    def k_uv_static(c_ptr, o_ptr, ITERS: tl.constexpr, HC: tl.constexpr, EPS: tl.constexpr):
        i = tl.arange(0, HC)[:, None]
        j = tl.arange(0, HC)[None, :]
        off = i * HC + j
        c0 = tl.load(c_ptr + off)
        u = tl.zeros([HC], dtype=tl.float32) + 1.0
        v = tl.zeros([HC], dtype=tl.float32) + 1.0
        for _ in tl.static_range(ITERS):
            r = tl.sum(c0 * v[None, :], axis=1)
            u = u / (u * r + EPS)
            s = tl.sum(c0 * u[:, None], axis=0)
            v = v / (v * s + EPS)
        tl.store(o_ptr + off, u[:, None] * c0 * v[None, :])

    kernels = {
        "2d_static": k_2d_static,
        "2d_range_1axis": k_2d_range_1axis,
        "uv_range": k_uv_range,
        "uv_static": k_uv_static,
    }
    kern = kernels[case]

    # 构造一个和真实场景同分布的 C0：softmax 后再做一次行/列归一
    torch.manual_seed(0)
    c0 = torch.randn(HC, HC, device=dev)
    c0 = torch.softmax(c0, dim=-1) + EPS
    c0 = c0 / (c0.sum(dim=-2, keepdim=True) + EPS)
    out = torch.empty_like(c0)

    t0 = time.perf_counter()
    kern[(1,)](c0, out, ITERS=ITERS, HC=HC, EPS=EPS)     # 首次调用含编译
    getattr(torch, dev_name).synchronize()
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(50):
        kern[(1,)](c0, out, ITERS=ITERS, HC=HC, EPS=EPS)
    getattr(torch, dev_name).synchronize()
    run_ms = (time.perf_counter() - t0) / 50 * 1e3

    if not torch.isfinite(out).all():
        print(f"COMPILE_S={compile_s:.2f} RUN_MS={run_ms:.4f} NAN")
        return 3

    # 与 torch 参考对拍（单向规约的诊断变体不参与，它本来就算的是别的东西）
    if case == "2d_range_1axis":
        print(f"COMPILE_S={compile_s:.2f} RUN_MS={run_ms:.4f} DIFF=n/a")
        return 0

    ref = c0.clone()
    for _ in range(ITERS):
        ref = ref / (ref.sum(dim=-1, keepdim=True) + EPS)
        ref = ref / (ref.sum(dim=-2, keepdim=True) + EPS)
    diff = (out - ref).abs().max().item()
    print(f"COMPILE_S={compile_s:.2f} RUN_MS={run_ms:.4f} DIFF={diff:.3e}")
    return 0 if diff < 1e-4 else 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=sorted(VARIANTS))
    args = ap.parse_args()

    if args.case:
        return build_and_run(args.case)

    print(f"真实配置: HC={HC}  ITERS={ITERS}  EPS={EPS}")
    print("提示: 每个变体在独立子进程里跑，段错误不会带走整个探针\n")
    print(f"{'变体':<16} {'结果':<12} {'编译(s)':>9} {'执行(ms)':>10} {'与torch差':>11}")
    print("-" * 78)
    verdict = {}
    TIMEOUTS = {"uv_range": 300, "2d_range_1axis": 300, "uv_static": 1800, "2d_static": 1800}
    for case in ("uv_range", "2d_range_1axis", "uv_static", "2d_static"):
        r = subprocess.run(
            [sys.executable, "-X", "faulthandler", os.path.abspath(__file__), "--case", case],
            capture_output=True, text=True, timeout=TIMEOUTS[case])
        out = (r.stdout or "").strip()
        if r.returncode == 0 and "COMPILE_S" in out:
            kv = dict(p.split("=") for p in out.split() if "=" in p)
            print(f"{case:<16} {'通过':<12} {float(kv['COMPILE_S']):>9.2f} "
                  f"{float(kv['RUN_MS']):>10.4f} {kv['DIFF']:>11}")
            verdict[case] = "pass"
        elif r.returncode < 0 or r.returncode == 139:
            print(f"{case:<16} {'段错误 ✗':<12}")
            verdict[case] = "segv"
        elif r.returncode == 4:
            kv = dict(p.split("=") for p in out.split() if "=" in p)
            print(f"{case:<16} {'数值不符 ✗':<12} {float(kv['COMPILE_S']):>9.2f} "
                  f"{float(kv['RUN_MS']):>10.4f} {kv['DIFF']:>11}")
            verdict[case] = "wrong"
        elif r.returncode == 2:
            print("无加速器")
            return 2
        else:
            tail = (r.stderr or "").strip().splitlines()[-2:]
            print(f"{case:<16} {'失败 ✗':<12}  {' | '.join(tail)}")
            verdict[case] = "fail"

    print("-" * 78)
    print("\n怎么读这张表:")
    if verdict.get("uv_range") == "pass":
        print("  * uv_range 通过 -> 编译器崩的是【2D loop-carried + 规约】的组合。")
        print("    改用 u/v 形式即可，循环保留真正的 scf.for，编译量与 ITERS 无关，")
        print("    不必付 static_range 的展开代价。这是首选方案。")
    elif verdict.get("uv_range") == "segv":
        print("  * uv_range 也崩 -> 编译器崩的是【循环体内有任何规约】，与 loop-carried")
        print("    是 1D 还是 2D 无关。只能用 static_range，接受编译代价。")
        if verdict.get("uv_static") == "pass":
            print("    但 uv_static 通过，且它的展开体比 2d_static 小（循环体里是 1D 运算），")
            print("    编译时间应明显更短 —— 对比上面两行的编译(s)列。")
    if verdict.get("2d_range_1axis") == "pass" and verdict.get("uv_range") == "segv":
        print("  * 单向规约不崩、双向崩 -> 是两个方向的 layout 转换在打架，")
        print("    可考虑把行归一和列归一拆成两个连续的 scf.for 循环再试。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
