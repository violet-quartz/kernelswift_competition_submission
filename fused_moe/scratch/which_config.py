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
        # v2：config 是叉乘关系，逐 config 扫没意义；但**分 kernel 计时**很有意义 ——
        # 决定 partial[E,T,H] 那 340 KB 的往返值不值得想办法省掉。
        _per_kernel_v2(mod, model, hidden_states, router_logits, ms)
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


def _per_kernel_v2(mod, model, hidden_states, router_logits, end_to_end_ms):
    """按 autotune 选中的 config 分别 launch 两个 kernel 计时。"""
    dev = hidden_states.device
    T, H = hidden_states.shape
    E = model.num_experts
    x = hidden_states.contiguous()
    logits = router_logits.contiguous()
    w1t, w2t = model._prepared_weights(x.dtype)
    partial = torch.empty((E, T, H), dtype=torch.float32, device=dev)
    out = torch.empty((T, H), dtype=x.dtype, device=dev)

    ek, rk = mod._moe_expert_kernel, mod._moe_reduce_kernel
    ebt = ek.best_config.kwargs["BLOCK_T"]
    rbt = rk.best_config.kwargs["BLOCK_T"]

    def run_expert():
        ek.fn[(E, triton.cdiv(T, ebt))](
            x, logits, w1t, w2t, partial, T,
            E=E, H=model.hidden_size, I=model.intermediate_size,
            TOP_K=model.top_k, RENORM=model.renormalize,
            BLOCK_T=ebt, num_warps=ek.best_config.num_warps, num_stages=1)

    def run_reduce():
        rk.fn[(triton.cdiv(T, rbt),)](
            partial, out, T, E=E, H=model.hidden_size,
            BLOCK_T=rbt, num_warps=rk.best_config.num_warps, num_stages=1)

    run_expert(); run_reduce(); torch.cuda.synchronize()
    e_us = triton.testing.do_bench(run_expert) * 1000
    r_us = triton.testing.do_bench(run_reduce) * 1000
    tot = end_to_end_ms * 1000

    print(f"\n{'kernel':>22} {'us':>8} {'占端到端':>9}")
    print(f"{'_moe_expert_kernel':>22} {e_us:>8.1f} {e_us / tot * 100:>8.1f}%")
    print(f"{'_moe_reduce_kernel':>22} {r_us:>8.1f} {r_us / tot * 100:>8.1f}%")
    print(f"{'两者之和':>22} {e_us + r_us:>8.1f} {(e_us + r_us) / tot * 100:>8.1f}%")
    print(f"{'端到端':>22} {tot:>8.1f}")
    print(f"\n差值 {tot - e_us - r_us:.1f} us = host 侧 + 两次 launch + partial 分配")
    print(f"参考：reduce 要搬 {E * T * H * 4 / 1024:.0f} KB fp32 读 + "
          f"{T * H * 2 / 1024:.0f} KB fp16 写")


if __name__ == "__main__":
    main()
