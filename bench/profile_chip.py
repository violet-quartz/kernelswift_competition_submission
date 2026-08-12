#!/usr/bin/env python3
"""对任意算子的 v0 / v1 做芯片侧 profiling —— 通用驱动，不需要每个算子写探针。

为什么这个能完全通用（不像 bench/check_spill.py 要 spill_probe.py）
------------------------------------------------------------------
check_spill 要调 kernel.warmup()，必须知道那个 kernel 的完整签名，所以每个算子
得自己写一份。这里只需要 model.forward(*inputs) —— 而 auto_bench.py 的契约已经
保证了每个 v0/v1 都提供 Model/ModelNew + get_init_inputs + get_inputs，所以驱动
可以完全通用。

口径与 auto_bench.py 对齐的地方
-------------------------------
  * 模型分别用各自的 get_init_inputs() 构造（auto_bench L378-393 就是这么做的）
  * **输入统一用 v0 的**，v1 的 get_inputs() 只用来校验参数个数 —— 对应
    auto_bench L530 那句 `v1_inputs = clone_value(v0_inputs)`。这样 v0/v1 的
    profiling 结果才是同一份数据上的公平对比。

看什么
------
昇腾 DaVinci 的 AI Core 内部是几条独立流水线，profiler 给出各自占比：

    mte2_ratio   HBM -> UB（搬入）    memory-bound 算子这项该高
    mte3_ratio   UB -> HBM（搬出）    同上
    vec_ratio    向量计算单元
    mac_ratio    Cube（矩阵）单元     没用矩乘的话应该接近 0
    scalar_ratio 标量单元（地址计算等）

判读：mte2+mte3 高 => 真的在拼命搬数据，瓶颈是带宽；**所有 ratio 都低** =>
AI Core 大部分时间在空等，瓶颈是调度/launch 开销，不是访存也不是计算。

还要重点看 kernel_details.csv 的 `Block Dim` 列：kernel 实际被分到几个核上跑。
如果 grid 给了 8192 而 Block Dim 远小于它，说明在核内串行循环了很多轮。

用法
----
    python3 bench/profile_chip.py mhc_post              # v0 和 v1 都跑
    python3 bench/profile_chip.py mhc_post --which v1   # 只跑 v1
    python3 bench/profile_chip.py mhc_post --summary-only ./prof_out/...  # 只解析已有结果

产物写到 <算子文件夹>/scratch/prof_<v0|v1>/，属于调研产物，不进 results/。
"""
import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


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


def load_module(task: str, sub: str):
    """加载 <task>/<sub>/<task>.py。sub 取 'v0' 或 'v1'。"""
    path = REPO_ROOT / task / sub / f"{task}.py"
    if not path.is_file():
        raise SystemExit(f"找不到 {path}")
    spec = importlib.util.spec_from_file_location(f"_prof_{sub}_{task}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def to_dev(value, dev):
    if isinstance(value, torch.Tensor):
        return value.to(dev)
    if isinstance(value, (list, tuple)):
        return type(value)(to_dev(v, dev) for v in value)
    return value


def build(task: str, sub: str, dev):
    """返回 (model, 该模块自己的 get_inputs 结果)。类名按 auto_bench 的契约取。"""
    mod = load_module(task, sub)
    cls_name = "Model" if sub == "v0" else "ModelNew"
    cls = getattr(mod, cls_name, None)
    if cls is None:
        raise SystemExit(f"{task}/{sub}/{task}.py 里找不到 {cls_name}")
    model = cls(*(mod.get_init_inputs() or []))
    if hasattr(model, "to"):
        model = model.to(dev)
    if hasattr(model, "eval"):
        model.eval()
    return model, mod


def profile_npu(model, inputs, outdir: Path, warmup: int, active: int):
    """昇腾：torch_npu.profiler，带 AI Core 流水线占用率。"""
    import torch_npu

    # _ExperimentalConfig 的字段名在不同 CANN / torch_npu 版本间变动过，
    # 拿不到就退化成不带 aic_metrics 的基础 profiling，别让整个脚本挂掉。
    exp_cfg = None
    try:
        exp_cfg = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        )
    except Exception as exc:
        print(f"  ⚠ 拿不到 _ExperimentalConfig（{type(exc).__name__}: {exc}）")
        print("    退化成基础 profiling —— 不会有 aic_*_ratio 那几列。")
        print("    可以 dir(torch_npu.profiler) 看看这个版本实际提供了什么。")

    kwargs = dict(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=0, warmup=warmup, active=active, repeat=1
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(outdir)),
    )
    if exp_cfg is not None:
        kwargs["experimental_config"] = exp_cfg

    with torch_npu.profiler.profile(**kwargs) as prof:
        for _ in range(warmup + active):
            with torch.no_grad():
                model.forward(*inputs)
            prof.step()          # 必须调，否则 schedule 不推进


def profile_cuda(model, inputs, outdir: Path, warmup: int, active: int):
    """CUDA 系（含沐曦 MACA）：标准 torch.profiler，用来跟昇腾横向对比。

    注意没有 AI Core 流水线那几个指标 —— 那是 DaVinci 专属。这里能看的是
    每个 kernel 的耗时占比和调用次数。
    """
    outdir.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=warmup, active=active, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(outdir)),
        record_shapes=True,
    ) as prof:
        for _ in range(warmup + active):
            with torch.no_grad():
                model.forward(*inputs)
            prof.step()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))


# ---------------------------------------------------------------------------
# 解析昇腾的 kernel_details.csv
# ---------------------------------------------------------------------------
def _find_col(fieldnames, *keywords):
    """列名在不同 CANN 版本间会变，按关键字模糊匹配。"""
    for f in fieldnames:
        low = f.lower()
        if all(k in low for k in keywords):
            return f
    return None


def summarize_npu(outdir: Path):
    """把 kernel_details.csv 里最关键的几列汇总出来，省得手动翻 CSV。"""
    csvs = list(outdir.rglob("ASCEND_PROFILER_OUTPUT/kernel_details.csv"))
    if not csvs:
        print(f"  （{outdir} 下没找到 kernel_details.csv，"
              f"可能 profiling 还没落盘，或这个版本产物结构不同——手动翻一下目录）")
        return
    path = csvs[0]
    print(f"\n  解析 {path}")

    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("  （文件是空的）")
        return

    fields = rows[0].keys()
    c_name = _find_col(fields, "name") or list(fields)[0]
    c_dur = _find_col(fields, "duration")
    c_block = _find_col(fields, "block", "dim")
    # 注意不能写成 `"ratio" in f.lower()` —— "du(ratio)n" 里就有这个子串，
    # Duration(us) 会被误判成占用率列。aic 指标实际叫 aic_mte2_ratio 这种。
    ratio_cols = [
        f for f in fields
        if f != c_dur and (f.lower().endswith("ratio") or "_ratio" in f.lower())
    ]

    # 按 kernel 名聚合：次数、总耗时、各 ratio 的均值
    agg = {}
    for r in rows:
        key = r.get(c_name, "?")
        a = agg.setdefault(key, {"n": 0, "dur": 0.0, "block": r.get(c_block, "?"),
                                 "ratios": {c: [] for c in ratio_cols}})
        a["n"] += 1
        try:
            a["dur"] += float(r.get(c_dur) or 0)
        except ValueError:
            pass
        for c in ratio_cols:
            try:
                a["ratios"][c].append(float(r[c]))
            except (TypeError, ValueError):
                pass

    total = sum(a["dur"] for a in agg.values()) or 1.0
    print(f"\n  {'kernel':<42} {'次数':>5} {'总耗时(us)':>11} {'占比':>7} {'BlockDim':>9}")
    print("  " + "-" * 78)
    for name, a in sorted(agg.items(), key=lambda kv: -kv[1]["dur"]):
        short = name if len(name) <= 40 else name[:37] + "..."
        print(f"  {short:<42} {a['n']:>5} {a['dur']:>11.1f} "
              f"{a['dur'] / total * 100:>6.1f}% {str(a['block']):>9}")

    if not ratio_cols:
        print("\n  （没有 aic_*_ratio 列 —— 说明 profiling 没带上 PipeUtilization，"
              "只能看耗时，看不出时间花在访存还是计算）")
        return

    print(f"\n  AI Core 流水线占用率（均值）：")
    for name, a in sorted(agg.items(), key=lambda kv: -kv[1]["dur"]):
        vals = {c: (sum(v) / len(v)) for c, v in a["ratios"].items() if v}
        if not vals:
            continue
        short = name if len(name) <= 40 else name[:37] + "..."
        print(f"    {short}")
        for c, v in sorted(vals.items(), key=lambda kv: -kv[1]):
            print(f"      {c:<28} {v:>8.4f}")
    print("\n  判读：mte2(搬入)+mte3(搬出) 高 => 瓶颈是带宽；")
    print("        所有 ratio 都低 => AI Core 在空等，瓶颈是调度/launch 开销。")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("task", help="算子文件夹名，比如 hc_split_sinkhorn / mhc_post")
    ap.add_argument("--which", choices=["v0", "v1", "both"], default="both")
    ap.add_argument("--warmup", type=int, default=10,
                    help="Triton 首次调用要 JIT 编译，不排除掉会算进耗时")
    ap.add_argument("--active", type=int, default=5, help="实际采样的迭代数")
    ap.add_argument("--summary-only", metavar="DIR",
                    help="不跑 profiling，只解析已有产物目录")
    args = ap.parse_args()

    if args.summary_only:
        summarize_npu(Path(args.summary_only))
        return 0

    op_dir = REPO_ROOT / args.task
    if not ((op_dir / "v0").is_dir() and (op_dir / "v1").is_dir()):
        raise SystemExit(f"{op_dir} 不是算子文件夹（需要同时有 v0/ 和 v1/ 子目录）")

    dev_name, _ = pick_device()
    dev = torch.device(dev_name)
    print(f"task={args.task}  device={dev_name}  warmup={args.warmup} active={args.active}")

    # 输入统一取 v0 的，跟 auto_bench L530 一致，保证 v0/v1 是同一份数据
    _, v0_mod = build(args.task, "v0", dev)
    inputs = to_dev(v0_mod.get_inputs() or [], dev)

    subs = ["v0", "v1"] if args.which == "both" else [args.which]
    for sub in subs:
        print(f"\n{'=' * 70}\n=== {sub} ===")
        model, _ = build(args.task, sub, dev)
        outdir = op_dir / "scratch" / f"prof_{sub}"
        outdir.mkdir(parents=True, exist_ok=True)

        if dev_name == "npu":
            profile_npu(model, inputs, outdir, args.warmup, args.active)
            summarize_npu(outdir)
        else:
            profile_cuda(model, inputs, outdir, args.warmup, args.active)
        print(f"\n  产物: {outdir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
