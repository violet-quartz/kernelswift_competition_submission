#!/usr/bin/env python3
"""统计 v0 参考实现每次 forward 实际派发了多少个 aten 算子。

这是我们全部加速比预估的**事实依据**。auto_bench.py 的计时方式是
「每次 forward 单独 sync 计时」（L429-445），在这些题的数据规模下
耗时基本正比于 kernel launch 次数，所以：

    可达加速比 ≈ v0 的算子数 / v1 的算子数（有个由单次 launch+sync 固定开销决定的上限）

用 TorchDispatchMode 拦截派发，而不是用 profiler —— 好处是**不需要加速器**，
纯 CPU 上就能跑。你在拿到卡之前就能把这张基线表打出来。

用法:
    python bench/count_ops.py                 # 统计全部 v0
    python bench/count_ops.py --detail        # additionally 列出每个算子的调用次数
    python bench/count_ops.py --v1            # 统计 v1（用来验证融合后确实降下来了）
    python bench/count_ops.py --update        # 把结果写回 bench/tasks.json 的 aten_ops 字段
"""
import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "bench" / "tasks.json"


class OpCounter(TorchDispatchMode):
    """拦截 aten 派发并计数。

    注意这里数的是 **aten 算子调用次数**，不是最终的 kernel 数——
    一个 aten 算子在后端可能对应 1 个或多个 kernel，个别算子（如 view/reshape）
    甚至不产生 kernel。所以这个数字是「上界的良好代理」，用于排序和估算量级，
    不是精确的 launch 计数。真实 launch 数以上机后的 profiler 为准。
    """

    def __init__(self):
        super().__init__()
        self.counts = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.counts[str(func)] += 1
        return func(*args, **(kwargs or {}))


# 这些算子不产生实际计算 kernel，单独列出以便区分
NO_KERNEL = {
    "aten.view.default", "aten.reshape.default", "aten._unsafe_view.default",
    "aten.expand.default", "aten.transpose.int", "aten.t.default",
    "aten.detach.default", "aten.slice.Tensor", "aten.unsqueeze.default",
    "aten.squeeze.dim", "aten.permute.default", "aten.alias.default",
}


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"_ks_{path.stem}_{path.parent.name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def count_for(path: Path, cls_name: str, seed: int):
    mod = load_module(path)
    cls = getattr(mod, cls_name)

    torch.manual_seed(seed)
    model = cls(*(mod.get_init_inputs() or []))
    if hasattr(model, "eval"):
        model.eval()
    torch.manual_seed(seed)
    inputs = mod.get_inputs() or []

    # 先热一次，把 lazy init / 缓存分配这类一次性开销排除掉
    with torch.no_grad():
        model.forward(*inputs)

    counter = OpCounter()
    torch.manual_seed(seed)
    with torch.no_grad(), counter:
        model.forward(*inputs)
    return counter.counts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v1", action="store_true", help="统计 v1/ModelNew 而不是 v0/Model")
    p.add_argument("--detail", action="store_true", help="列出每个算子的调用次数")
    p.add_argument("--update", action="store_true", help="把 v0 结果写回 tasks.json 的 aten_ops")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--only", nargs="*", default=None)
    args = p.parse_args()

    sub, cls_name = ("v1", "ModelNew") if args.v1 else ("v0", "Model")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tasks = manifest["tasks"]
    if args.only:
        tasks = [t for t in tasks if t["name"] in args.only]

    print(f"统计 {sub}/ 的 aten 算子调用次数（设备: CPU，与加速器无关）\n")
    print(f"{'Task':<36} {'总数':>6} {'其中无kernel':>12} {'有效':>6}")
    print("-" * 64)

    totals = {}
    for t in tasks:
        name = t["name"]
        path = ROOT / sub / f"{name}.py"
        if not path.is_file():
            print(f"{name:<36} {'缺失':>6}")
            continue
        try:
            counts = count_for(path, cls_name, args.seed)
        except Exception as exc:
            print(f"{name:<36} 失败: {type(exc).__name__}: {exc}")
            continue
        total = sum(counts.values())
        free = sum(c for op, c in counts.items() if op in NO_KERNEL)
        totals[name] = total - free
        print(f"{name:<36} {total:>6} {free:>12} {total - free:>6}")
        if args.detail:
            for op, c in counts.most_common():
                tag = "  (无kernel)" if op in NO_KERNEL else ""
                print(f"    {c:>4} x {op}{tag}")
            print()

    if args.update and not args.v1:
        for t in manifest["tasks"]:
            if t["name"] in totals:
                t["aten_ops"] = totals[t["name"]]
        manifest["_baseline_note"] = (
            "aten_ops = v0 每次 forward 的有效 aten 算子调用数（已扣除 view/reshape 等无 kernel 的），"
            "由 bench/count_ops.py --update 实测填入。"
        )
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\n已把 aten_ops 写回 {MANIFEST}")


if __name__ == "__main__":
    main()
