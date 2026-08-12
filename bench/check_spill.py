#!/usr/bin/env python3
"""检查 v1 kernel 是否寄存器溢出 —— 通用驱动，适配每个算子文件夹。

用 kernel.warmup() 只编译、不执行，读出编译器实际分配的寄存器数：

    n_regs   : 每个 program 用了多少寄存器
    n_spills : 溢出到 local memory（其实还是显存）的寄存器数 —— > 0 就是性能灾难
    smem     : shared memory 占用（字节）

跟 auto_bench.py 的计时口径无关，纯一次性诊断，不产出 results/ 下任何文件。

【这三个指标不是所有后端都有】
它们来自 CUDA 的 cuFuncGetAttribute + ptxas 输出，是 NVIDIA（以及对标 CUDA 的
沐曦 MACA）专属概念。昇腾 DaVinci 是 AI Core + Unified Buffer 结构，没有"每线程
寄存器数"和 CUDA 意义上的 shared memory，triton-ascend 只保留了字段名、值是占位符
—— 实测昇腾上读出 n_regs=0 n_spills=0 smem=1。

早先的版本直接判 `if n_spills:` ，昇腾上就会打印"✅ 无寄存器溢出" —— 这是**假阴性**
（被检测的条件是"有溢出"，报告"无"而实际未知），比不检查更糟，因为它会让人以为
这个维度已经排查过了。所以现在先判字段可不可信，不可信就明说本平台测不了。

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


def dump_metadata(k):
    """把后端实际报告的 metadata 原样打出来 —— 可能有平台特有的资源字段。"""
    md = getattr(k, "metadata", None)
    if md is None:
        print("    (没有 metadata 属性)")
        return
    try:
        print("   ", md._asdict())          # triton 的 metadata 通常是 namedtuple
    except AttributeError:
        try:
            print("   ", vars(md))
        except TypeError:
            print("   ", md)


def report(k, task: str, dev_name: str):
    """打印静态资源指标，返回 True/False/None（None = 本平台无法判定）。"""
    n_regs = getattr(k, "n_regs", None)
    n_spills = getattr(k, "n_spills", None)
    smem = getattr(getattr(k, "metadata", None), "shared", None)

    print(f"task={task}  device={dev_name}")
    print(f"n_regs={n_regs}  n_spills={n_spills}  smem={smem}")

    # 判据：CUDA 系后端的 n_regs 必然 > 0（任何 kernel 都要用寄存器）。
    # 读到 0 或读不到，说明后端没填这些字段，不是"用了 0 个寄存器"。
    if not (isinstance(n_regs, int) and n_regs > 0):
        print(f"\n⚠ 这些字段在 {dev_name} 后端上未被填充，**不能据此判断有无溢出**。")
        print("  n_regs / n_spills 来自 CUDA 的 cuFuncGetAttribute + ptxas，")
        print("  昇腾 DaVinci 没有等价概念（AI Core + Unified Buffer 结构）。")
        print("  要在这个平台上查片上资源，用昇腾侧的 profiler（msprof）。")
        print("\n  后端实际报告的 metadata（可能有平台特有字段）：")
        dump_metadata(k)
        return None

    if n_spills:
        print(
            "\n⚠ n_spills > 0 —— 寄存器装不下，溢出到 local memory，是性能灾难，"
            "需要回去改 kernel（减小同时活跃的临时变量、或拆分计算）。"
        )
        return False

    print("\n✅ 无寄存器溢出。")
    return True


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
    # _init_handles() 是私有 API，某些后端上可能直接抛异常。真抛了也要能走到
    # 下面的"无法判定"分支，把话说清楚，而不是连诊断结论都印不出来就崩掉。
    try:
        k._init_handles()
    except Exception as exc:
        print(f"task={args.task}  device={dev_name}")
        print(f"\n⚠ k._init_handles() 失败：{type(exc).__name__}: {exc}")
        print("  这个后端可能不支持读取静态资源指标，**无法判断有无溢出**。")
        return 2

    verdict = report(k, args.task, dev_name)
    # 退出码区分三种情况，方便以后挂 CI：0=确认没溢出，1=确认溢出，2=测不了
    if verdict is True:
        return 0
    if verdict is False:
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
