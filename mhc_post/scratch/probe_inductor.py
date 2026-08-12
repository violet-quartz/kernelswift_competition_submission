#!/usr/bin/env python3
"""观察 Inductor 对 v0 参考实现的融合程度（沐曦 / 昇腾 / 寒武纪通用）

仅用于探测，不进入提交材料。

用法:
    # 看生成的 kernel 源码
    TORCH_LOGS="output_code" python3 scratch/probe_inductor.py 2>&1 | tee /tmp/inductor.log

    # 看融合决策过程（比读生成代码更容易看出卡在哪）
    TORCH_LOGS="output_code,fusion" python3 scratch/probe_inductor.py 2>&1 | tee /tmp/inductor.log

    # 指定 v0 文件 / 后端
    KS_V0=v0/mhc_post.py KS_BACKEND=inductor python3 scratch/probe_inductor.py

跨平台要点:
  * 昇腾必须先 import torch_npu，torch.npu 才会注册；寒武纪同理要 torch_mlu。
    这一步必须在任何 torch.xxx.is_available() 之前完成。
  * 昇腾的默认 torch.compile 后端未必是 Inductor —— CANN 侧走的是 torchair。
    本脚本默认仍用 inductor（我们要看的就是它的融合），失败时会提示换后端。
  * 用 compiled(*inputs) 而不是 compiled.forward(*inputs)：后者在部分 torch
    版本里绕过 __call__ 上的 dynamo 钩子，导致压根没触发编译。
"""
import importlib.util
import os
import sys
from pathlib import Path

import torch

# --- 后端扩展：必须在探测设备之前导入 -------------------------------------
for _m in ("torch_npu", "torch_mlu"):
    try:
        importlib.import_module(_m)
        print(f"[env] {_m} 已导入")
    except ImportError:
        pass


def pick_device():
    """与 auto_bench.py 的 _iter_accelerators() 保持同样的探测顺序。"""
    for name in ("gcu", "cuda", "npu", "mlu"):
        mod = getattr(torch, name, None)
        if mod is None:
            continue
        try:
            if mod.is_available():
                return name, mod
        except Exception:
            continue
    return "cpu", None


def to_dev(x, dev):
    """递归搬运：inputs 可能是嵌套的 list/tuple/dict。"""
    if isinstance(x, torch.Tensor):
        return x.to(dev)
    if isinstance(x, (list, tuple)):
        return type(x)(to_dev(v, dev) for v in x)
    if isinstance(x, dict):
        return {k: to_dev(v, dev) for k, v in x.items()}
    return x


def main():
    root = Path(__file__).resolve().parent.parent
    v0_rel = os.environ.get("KS_V0", "v0/mhc_post.py")
    v0_path = (root / v0_rel).resolve()
    if not v0_path.is_file():
        print(f"!! 找不到 {v0_path}，用 KS_V0=<相对路径> 指定", file=sys.stderr)
        return 1

    spec = importlib.util.spec_from_file_location("v0_probe", v0_path)
    v0 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v0)
    print(f"[env] 加载 {v0_path}")

    dev_name, dev_mod = pick_device()
    print(f"[env] torch={torch.__version__}  device={dev_name}")
    if dev_mod is None:
        print("[env] !! 没探测到加速器，将在 CPU 上跑 —— "
              "Inductor 会生成 C++ 而不是 Triton，融合数量不可直接类比", file=sys.stderr)
    else:
        try:
            print(f"[env] {dev_mod.get_device_name(0)}  count={dev_mod.device_count()}")
        except Exception:
            pass
    device = torch.device(dev_name)

    model = v0.Model(*v0.get_init_inputs()).to(device).eval()
    inputs = to_dev(v0.get_inputs(), device)

    # --- 先跑一次 eager，确认模型本身在这块卡上是通的 ---------------------
    with torch.no_grad():
        ref = model(*inputs)
    if dev_mod is not None:
        dev_mod.synchronize()
    print(f"[eager] 通过，输出 "
          f"{tuple(ref.shape) if isinstance(ref, torch.Tensor) else type(ref).__name__}")

    # --- 断图检查：有 graph break 的话 Inductor 只看到碎片，融合数会被低估 ---
    try:
        import torch._dynamo as dynamo
        exp = dynamo.explain(model)(*inputs)
        nbreak = getattr(exp, "graph_break_count", None)
        print(f"[dynamo] graph_count={getattr(exp, 'graph_count', '?')} "
              f"graph_breaks={nbreak if nbreak is not None else '?'}")
        if nbreak:
            print("[dynamo] !! 有断图，下面的融合统计会偏低。原因:")
            for r in getattr(exp, "break_reasons", [])[:5]:
                print(f"          - {r}")
    except Exception as e:
        print(f"[dynamo] explain 不可用（不影响后续）: {type(e).__name__}: {e}")

    # --- 编译并执行 -------------------------------------------------------
    backend = os.environ.get("KS_BACKEND", "inductor")
    print(f"[compile] backend={backend}")
    try:
        compiled = torch.compile(model, backend=backend)
        with torch.no_grad():
            out = compiled(*inputs)          # 注意不是 compiled.forward(...)
        if dev_mod is not None:
            dev_mod.synchronize()
    except Exception as e:
        print(f"[compile] 失败: {type(e).__name__}: {e}", file=sys.stderr)
        if dev_name == "npu":
            print("\n昇腾上 Inductor 可能不受支持，试试 CANN 官方的图模式后端:", file=sys.stderr)
            print("    import torchair", file=sys.stderr)
            print("    KS_BACKEND=torchair python3 scratch/probe_inductor.py", file=sys.stderr)
            print("若走 torchair，就不会有 Triton kernel 产物 —— 说明这条链路上"
                  "「看 Inductor 融合程度」这个问题本身不适用。", file=sys.stderr)
        return 1

    # 对拍：编译前后结果应一致，否则后面的融合分析没有意义
    if isinstance(ref, torch.Tensor) and isinstance(out, torch.Tensor):
        try:
            torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
            print("[compile] 通过，且与 eager 结果一致")
        except AssertionError as e:
            print(f"[compile] !! 与 eager 结果不一致:\n{str(e)[:400]}")
    else:
        print("[compile] 通过（输出非单一张量，跳过对拍）")

    # --- 指出产物在哪 -----------------------------------------------------
    cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR") or "/tmp/torchinductor_" + (
        os.environ.get("USER") or "root")
    print(f"\n产物位置（TORCH_LOGS=output_code 时日志里也会打印确切路径）:\n  {cache}")
    print("\n数融合程度:")
    print(f"  find {cache} -name '*.py' -newermt '-5 minutes' | xargs grep -c '@triton.jit'")
    print("  # 数字 = Inductor 最终生成了几个 kernel，每个至少一次显存往返")
    print("  # 手写 kernel 的价值 = 把这个数字压到 1")
    return 0


if __name__ == "__main__":
    sys.exit(main())