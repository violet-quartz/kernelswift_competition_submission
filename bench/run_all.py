#!/usr/bin/env python3
"""批量运行 v0/v1 对拍 + 计时，汇总成每个算子文件夹下 results/<chip>/ 的 json 和 markdown 表。

目录约定：bench/、env/ 是所有算子共用的基础设施，放在仓库根目录；每个算子一个
文件夹（如 hc_split_sinkhorn/），内部是 tasks/（该算子的原题文件，未修改，供
对照）、v0/、v1/、results/。task 清单是仓库根目录下共用的一份 bench/tasks.json，
每个 task 的 name 就是仓库根目录下对应算子文件夹的名字——跟 bench/check_spill.py
认 task 名的方式一致。

每个 task 都是拉起一个独立的 auto_bench.py 子进程来跑的，理由有两个：
  1. auto_bench.py 保持与官方仓库逐字一致（见 bench/auto_bench.py 顶部说明），
     不打补丁、不 import 它的内部函数，避免"我们改过评测脚本"的嫌疑；
  2. 进程隔离，某个 task 把 Triton 编译搞崩不会带走整批。

用法:
    python bench/run_all.py                      # 跑 bench/tasks.json 里的全部 task
    python bench/run_all.py --chip metax-c500    # 显式指定芯片名（只影响输出目录名）
    python bench/run_all.py --only hc_split_sinkhorn    # 只跑指定 task
    python bench/run_all.py --warmup 50 --repeat 100    # 开发期快速迭代
"""
import argparse
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AUTO_BENCH = REPO_ROOT / "bench" / "auto_bench.py"
MANIFEST = REPO_ROOT / "bench" / "tasks.json"


# auto_bench.py L562 的输出格式
PASS_RE = re.compile(
    r"PASS accuracy;\s*v0=([\d.eE+-]+)\s*ms,\s*v1=([\d.eE+-]+)\s*ms,\s*speedup=([\d.eE+-]+)x"
)


def load_task_names() -> list[str]:
    """读仓库根目录共享的 tasks.json，只取 name——内容故意写得很薄。"""
    if not MANIFEST.is_file():
        raise SystemExit(f"找不到 {MANIFEST}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return [t["name"] for t in manifest["tasks"]]


def resolve_op_dir(name: str) -> Path:
    """task name 就是仓库根目录下同名算子文件夹——跟 bench/check_spill.py 同一套约定。"""
    op_dir = REPO_ROOT / name
    if not ((op_dir / "v0").is_dir() and (op_dir / "v1").is_dir()):
        raise SystemExit(
            f'tasks.json 里的 task "{name}" 在 {op_dir} 下找不到算子文件夹'
            f"（需要同时有 v0/ 和 v1/ 子目录）"
        )
    return op_dir


def detect_chip() -> str:
    """探测当前加速器，返回形如 'cuda:MetaX C500' 的标识。

    探测顺序与 auto_bench.py L213 的 _iter_accelerators() 保持一致。
    """
    try:
        import torch
    except ImportError:
        return "cpu-no-torch"
    for name in ("gcu", "cuda", "npu", "mlu"):
        mod = getattr(torch, name, None)
        if mod is None:
            continue
        try:
            if not mod.is_available():
                continue
        except Exception:
            continue
        try:
            dev_name = mod.get_device_name(0)
        except Exception:
            dev_name = "unknown"
        return f"{name}:{dev_name}"
    return "cpu-no-accelerator"


def collect_env() -> dict:
    """快照当前环境，作为提交材料里"环境配置文件"的实证部分。"""
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "chip": detect_chip(),
    }
    try:
        import torch

        env["torch"] = torch.__version__
    except Exception as exc:
        env["torch"] = f"<unavailable: {exc}>"
    try:
        import triton

        env["triton"] = triton.__version__
    except Exception as exc:
        env["triton"] = f"<unavailable: {exc}>"
    return env


def run_one(op_dir: Path, name: str, args) -> dict:
    v0 = op_dir / "v0" / f"{name}.py"
    v1 = op_dir / "v1" / f"{name}.py"
    rec = {"name": name}
    if not v0.is_file() or not v1.is_file():
        rec.update(status="missing-file", message=f"v0 exists={v0.is_file()}, v1 exists={v1.is_file()}")
        return rec

    cmd = [
        sys.executable, str(AUTO_BENCH),
        "--v0_file", str(v0),
        "--v1_file", str(v1),
        "--warmup", str(args.warmup),
        "--repeat", str(args.repeat),
        "--seed", str(args.seed),
        "--atol", str(args.atol),
        "--rtol", str(args.rtol),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    rec["wall_s"] = round(time.perf_counter() - t0, 1)
    out = proc.stdout + proc.stderr
    rec["raw_output"] = out.strip()

    m = PASS_RE.search(out)
    if m:
        rec.update(
            status="pass",
            v0_ms=float(m.group(1)),
            v1_ms=float(m.group(2)),
            speedup=float(m.group(3)),
        )
    else:
        fail = re.search(r"^FAIL (.*)$", out, re.MULTILINE)
        rec.update(status="fail", message=(fail.group(1) if fail else out.strip()[-400:]))
    return rec


def verdict(rec: dict) -> str:
    """把一条结果翻译成"精度对拍过没过"。"""
    return "✅ 通过" if rec["status"] == "pass" else "❌ 未通过"


def render_markdown(chip: str, env: dict, records: list) -> str:
    lines = [
        f"# 性能测试结果 — {chip}",
        "",
        f"- 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Python: {env['python']} / torch: {env['torch']} / triton: {env['triton']}",
        f"- 平台: {env['platform']}",
        "",
        "| Task | v0 (ms) | v1 (ms) | Speedup | 结论 |",
        "|---|---:|---:|---:|---|",
    ]
    for r in records:
        if r["status"] == "pass":
            lines.append(
                f"| {r['name']} | {r['v0_ms']:.4f} | {r['v1_ms']:.4f} | "
                f"**{r['speedup']:.2f}x** | {verdict(r)} |"
            )
        else:
            msg = (r.get("message") or "")[:80].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {r['name']} | — | — | — | {verdict(r)}: {msg} |")

    ok = [r for r in records if r["status"] == "pass"]
    lines.append("")
    if ok:
        prod = 1.0
        for r in ok:
            prod *= r["speedup"]
        geo = prod ** (1.0 / len(ok))
        lines.append(f"**通过 {len(ok)}/{len(records)} 题，几何平均加速比 {geo:.2f}x**")
    else:
        lines.append(f"**通过 0/{len(records)} 题**")
    return "\n".join(lines) + "\n"


def check_python_version():
    """auto_bench.py 需要 Python >= 3.10，提前失败并说清原因。

    它在 L23-25 的 @dataclass 字段和 L101 的函数签名里用了 `float | None`
    这种 PEP 604 语法。这些注解是**运行时求值**的（文件里没有
    `from __future__ import annotations`），所以在 3.9 上会直接抛
    TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'，
    而且报错位置在 auto_bench 内部，很容易被误判成环境坏了。
    """
    if sys.version_info < (3, 10):
        v = ".".join(map(str, sys.version_info[:3]))
        raise SystemExit(
            f"当前 Python {v}，但 bench/auto_bench.py 需要 >= 3.10。\n"
            f"原因：它用了 `float | None`（PEP 604）且是运行时求值的注解。\n"
            f"解决：换一个 3.10+ 的解释器（conda create -n ks python=3.10），\n"
            f"      注意 torch/torch_npu/triton 要装到同一个环境里。"
        )


def main():
    check_python_version()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chip", default="auto", help="芯片标识，仅用于命名输出目录；默认自动探测")
    p.add_argument("--only", nargs="*", default=None, help="只跑指定的 task（跨算子文件夹按 task 名过滤）")
    p.add_argument("--warmup", type=int, default=200, help="与官方默认一致")
    p.add_argument("--repeat", type=int, default=500, help="与官方默认一致")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--atol", type=float, default=1e-2)
    p.add_argument("--rtol", type=float, default=1e-2)
    args = p.parse_args()

    all_names = load_task_names()
    if not all_names:
        raise SystemExit(f"{MANIFEST} 里没有任何 task")

    names = all_names
    if args.only:
        unknown = set(args.only) - set(all_names)
        if unknown:
            raise SystemExit(f"未知 task: {sorted(unknown)}\n可选: {all_names}")
        names = [n for n in all_names if n in args.only]

    env = collect_env()
    chip = args.chip if args.chip != "auto" else env["chip"].replace(":", "-").replace(" ", "_")

    print(f"芯片: {env['chip']}   torch: {env['torch']}   triton: {env['triton']}")
    if args.warmup != 200 or args.repeat != 500:
        print(f"注意: warmup={args.warmup} repeat={args.repeat} 与官方默认(200/500)不同，"
              f"这组数字只适合开发期迭代，**不要**用于最终提交的性能报告")
    print(f"共 {len(names)} 个 task\n")

    any_fail = False
    for i, name in enumerate(names, 1):
        op_dir = resolve_op_dir(name)
        print(f"[{i}/{len(names)}] {name} ... ", end="", flush=True)
        rec = run_one(op_dir, name, args)
        if rec["status"] == "pass":
            print(f"{rec['speedup']:.2f}x  ({rec['v0_ms']:.4f} → {rec['v1_ms']:.4f} ms)  {verdict(rec)}")
        else:
            print(f"{verdict(rec)}  {(rec.get('message') or '')[:120]}")
            any_fail = True

        outdir = op_dir / "results" / chip
        outdir.mkdir(parents=True, exist_ok=True)
        payload = {"chip": chip, "env": env, "bench_args": vars(args), "results": [rec]}
        (outdir / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        md = render_markdown(chip, env, [rec])
        (outdir / "RESULTS.md").write_text(md, encoding="utf-8")

        print(f"已写入 {outdir / 'results.json'}")
        print(f"已写入 {outdir / 'RESULTS.md'}\n")

    if any_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
