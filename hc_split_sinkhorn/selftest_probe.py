#!/usr/bin/env python3
"""selftest 探针：v1/hc_split_sinkhorn.py 实际用到的 Triton 特性冒烟测试。

由 env/selftest.py 自动发现并调用（约定见 env/selftest.py 的同名章节）：
本文件必须提供模块级函数 probe(dev) -> (bool, str)。

背景见 docs/debug-hc_split_sinkhorn.md：把 2D 的 comb tile 直接作为循环携带
变量迭代，会让沐曦 Triton 3.0.0 的 make_ttgir 段错误（core dumped，不是抛
异常）。v1 实际用的是 u/v 对角缩放改写——循环只携带两个 1D 向量，2D 矩阵
是循环不变量，正面消除了触发条件。这里验的正是这个实际写法，不是已知会
崩溃、且已经绕开的朴素写法。

【为什么 triton / tl 要写在模块级 import，不能作为参数传进 probe()】
@triton.jit 编译 kernel 时，kernel 体里的自由变量（tl.arange、tl.sum ...）
是通过 fn.__globals__ 解析的——也就是定义该函数的**模块**（这个文件）的全局
命名空间，不是 probe() 的函数局部作用域。把 tl 作为参数传进来，kernel 体
真正编译时会找不到 tl。这条坑 env/selftest.py 顶部也记了一遍，这里再踩一次
没必要——直接在模块级 import 就从根上避开。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _sinkhorn_uv(p_ptr, o_ptr, ITERS: tl.constexpr, HC: tl.constexpr):
    i = tl.arange(0, HC)[:, None]
    j = tl.arange(0, HC)[None, :]
    off = i * HC + j
    x = tl.load(p_ptr + off)
    c0 = tl.exp(x - tl.max(x, axis=1)[:, None])      # 循环外算一次，之后只读

    u = tl.zeros([HC], dtype=tl.float32) + 1.0
    v = tl.zeros([HC], dtype=tl.float32) + 1.0
    for _ in range(ITERS):                            # 普通 range，不需要 static_range
        r = tl.sum(c0 * v[None, :], axis=1)
        u = u / (u * r + 1e-6)
        s = tl.sum(c0 * u[:, None], axis=0)
        v = v / (v * s + 1e-6)

    tl.store(o_ptr + off, u[:, None] * c0 * v[None, :])


def probe(dev):
    """编译 + 执行 + 数值校验。返回 (是否通过, 附加信息)。"""
    p = torch.randn(4, 4, device=dev)
    o = torch.empty_like(p)
    _sinkhorn_uv[(1,)](p, o, ITERS=20, HC=4)
    getattr(torch, dev.type).synchronize()
    col = o.sum(dim=0)
    ok = torch.allclose(col, torch.ones_like(col), atol=1e-3, rtol=1e-3)
    return ok, f"列和={col.tolist()}"
