#!/usr/bin/env python3
"""把 forward 逐步拆开计时，看 MLM head 那一行的每一步各花多少。

    python3 SPLADE_sparse_pooler/scratch/breakdown.py

为什么需要它：bench/profile_overhead.py 只能把"kernel 本体 + 其余 python"报成
一个数（本题 ~373 µs），看不出里面 dense / GELU / LayerNorm / decoder / pool
各占多少。要决定"融合哪一步"就得先知道这个分布。

口径与 auto_bench.py L429-445 一致（每次调用后 sync，取 median），
所以这里的数字可以直接和 run.sh 的结果对齐。

每一行同时给出该步的 roofline 下界（按沐曦实测 382 GB/s），
净增值离下界越远，说明该步越有优化空间。
"""
import importlib.util
import sys
from pathlib import Path

import torch
import triton

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "bench"))
from profile_overhead import bench, pick_device          # noqa: E402

BW = 382e9      # 沐曦 C500 实测有效带宽，来自 env/bandwidth.py

# 第一步的计时里裹着**每次 forward 都要付一遍**的固定开销：框架地板 + 首次
# launch + 输出分配。不减掉的话第一步会背上全部这笔账，看起来效率奇低。
# 数值来自 bench/profile_overhead.py 在同一台机器上的 ①②③ 三行。
_FIXED_US = 8.5 + 48.0 + 10.3


def _load_v1():
    path = ROOT / "SPLADE_sparse_pooler" / "v1" / "SPLADE_sparse_pooler.py"
    spec = importlib.util.spec_from_file_location("_ks_v1_splade_breakdown", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    dev_name, dev_mod = pick_device()
    dev = torch.device(dev_name)
    sync = dev_mod.synchronize
    warmup, repeat = 200, 500

    mod = _load_v1()
    m = mod.ModelNew(*mod.get_init_inputs()).to(dev).eval()
    hs, seq_lens = (t.to(dev) for t in mod.get_inputs())
    T, H = hs.shape
    V = m.decoder.weight.shape[0]
    S = seq_lens.shape[0]

    f32 = 4
    # ⚠ 这几步必须**逐字复刻 forward 的实际链路**，否则累计值不单调、净增出负数。
    #   踩过一次：forward 已经把 decoder 换成 fp16 的 F.linear，而这里还在调
    #   m.decoder(...)（fp32 的 nn.Linear），于是 ④ 比 ⑤ 还慢，⑤ 的净增是 -123.8。
    use_fp16 = getattr(mod, "_USE_FP16_DECODER", False)
    gdt = torch.float16 if use_fp16 else torch.float32
    gb = 2 if use_fp16 else 4              # decoder 这一路每元素几字节
    dw, db = m._decoder_weights(gdt)

    def head():                            # dense + GELU + LayerNorm，恒为 fp32
        return m.layer_norm(m.act(m.dense(hs)))

    with torch.no_grad():
        steps = [
            ("① dense",
             lambda: m.dense(hs),
             H * H * f32 + T * H * f32),
            ("② + GELU",
             lambda: m.act(m.dense(hs)),
             T * H * f32),
            ("③ + LayerNorm",
             head,
             T * H * f32),
            (f"④ + decoder（{'fp16' if use_fp16 else 'fp32'}，= 目标那一行）",
             lambda: torch.nn.functional.linear(head().to(gdt), dw, db),
             V * H * gb + V * gb + T * V * gb),
            ("⑤ + pool kernel（= 完整 forward）",
             lambda: m(hs, seq_lens),
             T * V * gb + S * V * f32),
        ]
        rows = []
        for label, fn, delta_bytes in steps:
            rows.append((label, bench(fn, sync, warmup, repeat) * 1e3, delta_bytes))

    print(f"task=SPLADE_sparse_pooler  device={dev_name}  warmup={warmup} repeat={repeat}")
    print(f"口径与 auto_bench.py L429-445 一致；roofline 按 {BW/1e9:.0f} GB/s")
    print(f"decoder 路径 dtype = {gdt}（v1 的 _USE_FP16_DECODER = {use_fp16}）\n")
    w = max(len(r[0]) for r in rows) + 2
    print(f"{'步骤':<{w}}{'累计(µs)':>10}{'净增(µs)':>10}{'该步访存':>11}{'roofline':>10}{'效率':>8}")
    print("-" * (w + 49))
    prev = _FIXED_US        # 第一步减掉固定开销，见 _FIXED_US 的说明
    for label, us, b in rows:
        net = us - prev
        floor = b / BW * 1e6
        eff = f"{floor / net * 100:.0f}%" if net > 0.5 else "—"
        print(f"{label:<{w}}{us:>10.1f}{net:>10.1f}{b/1024**2:>9.2f}MB{floor:>10.1f}{eff:>8}")
        prev = us
    print("-" * (w + 49))
    print(f"（① 的净增已扣掉 {_FIXED_US:.1f} µs 固定开销：框架地板 + 首次 launch + 输出分配）")

    tot_b = sum(r[2] for r in rows)
    print(f"\n合计访存 {tot_b/1024**2:.1f} MB -> roofline {tot_b/BW*1e6:.0f} µs，实测 {rows[-1][1]:.1f} µs"
          f"（含 ~{66:.0f} µs 固定开销，见 bench/profile_overhead.py）")
    print("\n怎么读：")
    print("  * 净增 >> roofline  -> 那一步有优化空间（多半是启动开销或实现低效）。")
    print("  * 净增 ~= roofline  -> 已经贴着带宽跑，只能靠**减少字节数**再进一步")
    print("                        （降精度 / 不落地中间张量）。")
    print("  * 效率 > 100%       -> 该步的数据命中了 cache，没真的走 HBM。")
    print("                        这类步骤的『融合掉中间张量』收益会低于 roofline 估计，")
    print("                        因为那次往返本来就没花 HBM 的钱。")


if __name__ == "__main__":
    main()
