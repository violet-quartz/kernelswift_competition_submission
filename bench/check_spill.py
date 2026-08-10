#!/usr/bin/env python3
"""检查 v1 kernel 是否寄存器溢出 —— 通用驱动，适配每个算子文件夹。

用 kernel.warmup() 只编译、不执行，读出编译器实际分配的寄存器数：

    n_regs   : 每个 program 用了多少寄存器
    n_spills : 溢出到 local memory（其实还是显存）的寄存器数 —— > 0 就是性能灾难
    smem     : shared memory 占用（字节）

跟 auto_bench.py 的计时口径无关，纯一次性诊断，不产出 results/ 下任何文件。

【为什么这里通用，warmup 怎么调用不通用】
"发现文件 -> warmup -> 读 n_regs/n_spills -> 打印判定" 这条链路对任何算子都一样，
是这个脚本负责的部分。但"给 kernel.warmup() 传哪些参数、grid 多大、哪些是
constexpr"，每个算子的 kernel 签名完全不同，没法在这里通用地生成。

约定跟 env/selftest.py 的 selftest_probe.py 一致：算子文件夹（判据同
bench/run_all.py 的 discover_operator_dirs() —— 同时有 v0/ 和 v1/ 子目录）下
放一个 spill_probe.py，提供模块级函数

    warmup(dev) -> triton kernel   # kernel.warmup(...) 的原始返回值，不要调 _init_handles()

本脚本负责发现它、调用它，剩下的事情通用处理。

用法:
    python3 bench/check_spill.py hc_split_sinkhorn
    python3 bench/check_spill.py mhc_post
"""
import argparse
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


def find_probe(task: str) -> Path:
    op_dir = REPO_ROOT / task
    if not ((op_dir / "v0").is_dir() and (op_dir / "v1").is_dir()):
        raise SystemExit(
            f"{op_dir} 不是一个算子文件夹（需要同时有 v0/ 和 v1/ 子目录），"
            f"task 名要传算子文件夹名，比如 hc_split_sinkhorn / mhc_post。"
        )
    probe_path = op_dir / "spill_probe.py"
    if not probe_path.is_file():
        raise SystemExit(
            f"{probe_path} 不存在。每个算子要检查 spill，需要在自己文件夹下放一个 "
            f"spill_probe.py，提供模块级函数 warmup(dev) -> kernel.warmup(...) 的返回值。"
        )
    return probe_path


def load_probe(task: str, probe_path: Path):
    spec = importlib.util.spec_from_file_location(f"_spill_probe_{task}", probe_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("task", help="算子文件夹名，比如 hc_split_sinkhorn / mhc_post")
    args = ap.parse_args()

    dev_name, _ = pick_device()
    dev = torch.device(dev_name)

    probe_path = find_probe(args.task)
    mod = load_probe(args.task, probe_path)
    if not hasattr(mod, "warmup"):
        raise SystemExit(f"{probe_path} 里找不到 warmup(dev) 函数")

    k = mod.warmup(dev)
    k._init_handles()

    print(f"task={args.task}  device={dev_name}")
    print(f"n_regs={k.n_regs}  n_spills={k.n_spills}  smem={k.metadata.shared}")
    if k.n_spills:
        print(
            "\n⚠ n_spills > 0 —— 寄存器装不下，溢出到 local memory，是性能灾难，"
            "需要回去改 kernel（减小同时活跃的临时变量、或拆分计算）。"
        )
    else:
        print("\n✅ 无寄存器溢出。")


if __name__ == "__main__":
    main()
