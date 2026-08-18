#!/usr/bin/env python3
"""报告 autotune 选中了哪个 config，并（单 kernel 版本时）逐个 config 单独计时。

    python3 fused_moe/scratch/which_config.py          # 默认 v1
    python3 fused_moe/scratch/which_config.py v2       # v2

为什么需要它：@triton.autotune 的 benchmark 结果是局部变量，跑完就丢了，
Autotuner 上只留下 .best_config / .cache（赢家），看不到「赢了多少」。
而判断要不要继续优化，需要的恰恰是全表的分布 —— 如果第一名和最后一名差 5%，
说明旋钮已经调到头了，瓶颈在别处；差 3 倍就还有得挖。

v1 是单 kernel，能做完整的逐 config 计时；v2 拆成 expert + reduce 两个 kernel，
config 是叉乘关系，这里只报告各自的赢家和端到端耗时。
"""
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.testing

OP_DIR = Path(__file__).resolve().parent.parent


def _load(ver):
    path = OP_DIR / ver / "fused_moe.py"
    spec = importlib.util.spec_from_file_location(f"_ks_{ver}_fused_moe_which", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _autotuners(mod):
    """模块里所有挂了 @triton.autotune 的 kernel，按定义顺序。"""
    found = []
    for name, obj in vars(mod).items():
        if hasattr(obj, "configs") and hasattr(obj, "best_config") and hasattr(obj, "fn"):
            found.append((name, obj))
    return found


def main():
    ver = sys.argv[1] if len(sys.argv) > 1 else "v1"
    dev = torch.device("cuda")
    mod = _load(ver)

    model = mod.ModelNew(*mod.get_init_inputs()).to(dev)
    hidden_states, router_logits = (t.to(dev) for t in mod.get_inputs())

    # --- 1. 真实跑一次，让 autotune 做出选择 ---
    with torch.no_grad():
        out = model(hidden_states, router_logits)
    torch.cuda.synchronize()

    tuners = _autotuners(mod)
    print("=" * 72)
    print(f"版本 {ver} —— {len(tuners)} 个挂了 autotune 的 kernel")
    for name, k in tuners:
        print(f"  {name}: {k.best_config}")

    # --- 2. 端到端计时 ---
    with torch.no_grad():
        ms = triton.testing.do_bench(lambda: model(hidden_states, router_logits))
    print(f"\n端到端 forward: {ms * 1000:.1f} us")

    if len(tuners) != 1:
        print("\n（多 kernel 版本，config 是叉乘关系，跳过逐 config 计时。)")
        return

    # --- 3. 单 kernel 版本：逐 config 单独 launch 计时，绕开 autotune 包装层 ---
    _, kernel = tuners[0]
    jit_fn = kernel.fn
    T, H = hidden_states.shape
    x = hidden_states.contiguous()
    logits = router_logits.contiguous()
    w1t, w2t = model._prepared_weights(x.dtype)
    buf = torch.empty((T, H), dtype=x.dtype, device=dev)
    ck = dict(E=model.num_experts, H=model.hidden_size, I=model.intermediate_size,
              TOP_K=model.top_k, RENORM=model.renormalize)

    rows = []
    for c in sorted(kernel.configs, key=lambda c: (c.kwargs["BLOCK_T"], c.num_warps)):
        bt, nw, ns = c.kwargs["BLOCK_T"], c.num_warps, c.num_stages
        grid = (triton.cdiv(T, bt),)

        def call(bt=bt, nw=nw, ns=ns, grid=grid):
            jit_fn[grid](x, logits, w1t, w2t, buf, T, **ck,
                         BLOCK_T=bt, num_warps=nw, num_stages=ns)

        try:
            call()
            torch.cuda.synchronize()
            us = triton.testing.do_bench(call) * 1000
            ok = torch.allclose(buf.float(), out.float(), atol=1e-2, rtol=1e-2)
        except Exception as exc:
            us, ok = float("nan"), f"<{type(exc).__name__}>"
        rows.append((bt, nw, ns, us, ok))

    best = min((r[3] for r in rows if r[3] == r[3]), default=float("nan"))
    print(f"\n{'BLOCK_T':>8} {'warps':>6} {'stages':>7} {'kernel us':>10} {'相对最快':>9}  数值")
    for bt, nw, ns, us, ok in rows:
        rel = f"{us / best:.2f}x" if us == us else "—"
        print(f"{bt:>8} {nw:>6} {ns:>7} {us:>10.1f} {rel:>9}  {ok}")
    print(f"\n最快 kernel: {best:.1f} us   端到端: {ms * 1000:.1f} us"
          f"   → 差值 {ms * 1000 - best:.1f} us 是 host 侧 + launch 开销")


if __name__ == "__main__":
    main()
