#!/usr/bin/env python3
"""后端连通性自检：torch 能不能看到卡、Triton 能不能真的编译出可用 kernel。

由 env/capture.sh 调用，也可以单独跑：

    python3 env/selftest.py

【为什么这是一个独立文件而不是 capture.sh 里的 heredoc】
最初写成 `python3 - <<'PY' ... PY` 从 stdin 喂进去，结果在冒烟测试处炸了：

    File "<stdin>", line 60, in <module>
    inspect.getsourcelines(fn) -> OSError: could not get source code

原因是 @triton.jit 在构造 JITFunction 时要用 inspect 把 kernel 的**源码文本**
抠出来做 AST 解析（triton/runtime/jit.py 的 JITFunction.__init__ 里
self.starting_line_number = inspect.getsourcelines(fn)[1]）。
从 stdin 读入的代码在磁盘上没有对应文件，inspect 找不到源，直接 OSError。

结论：**任何含 @triton.jit 的代码都必须放在真实文件里**，不能走 stdin、
也不能用 exec(字符串)。这一条对后面调试 kernel 时同样适用。

退出码: 0 = 全通过，1 = 任一环节失败。
"""
import importlib
import sys

# 【tl 必须在模块级可见】
# @triton.jit 编译 kernel 时，kernel 体里的自由变量（tl.arange、tl.sum ...）是通过
# fn.__globals__ 解析的 —— 也就是**定义该函数的模块的全局命名空间**，而不是定义处
# 的函数局部作用域。所以把 `import triton.language as tl` 写在 main() 里，
# 装饰器本身能过（签名里的 tl.constexpr 在装饰时求值，那时 tl 是局部变量），
# 但 kernel 体真正编译时就找不到 tl 了。
#
# 这也是文件里一度出现 globals()['tl'] = tl 的由来 —— 那是对症的补丁，
# 这里从根上解决：直接在模块级 import。
#
# 仍然包在 try 里，是为了保住"优雅报错"的能力：triton 装错/没装时应该由
# main() 打印诊断信息并返回 1，而不是脚本一 import 就崩、连前面的 torch
# 和设备信息都来不及记进 env.lock.txt。
try:
    import triton
    import triton.language as tl

    _TRITON_IMPORT_ERROR = None
except Exception as _exc:  # noqa: BLE001 - 任何导入期异常都要能被 main() 报出来
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = _exc


def main() -> int:
    # --- ① 后端扩展 ---------------------------------------------------------
    # 与 v0/v1 里的 _ks_bootstrap() 保持一致：昇腾要 torch_npu、寒武纪要 torch_mlu，
    # 沐曦走 torch.cuda 不需要扩展，所以这里静默跳过是正常的。
    for m in ("torch_npu", "torch_mlu"):
        try:
            importlib.import_module(m)
            print(f"backend ext: {m} 已导入")
        except ImportError:
            pass

    # --- ② torch ------------------------------------------------------------
    try:
        import torch
    except Exception as e:
        print("torch 导入失败:", e)
        return 1
    print("torch:", torch.__version__)

    # --- ③ 探测加速器 -------------------------------------------------------
    # 顺序逐字复制 auto_bench.py L213 的 _iter_accelerators()。
    # 这里问的不是"机器上有没有卡"，而是"auto_bench 会不会找到跟我一样的东西"。
    found = None
    for name in ("gcu", "cuda", "npu", "mlu"):
        mod = getattr(torch, name, None)
        if mod is None:
            continue
        try:
            ok = mod.is_available()
        except Exception as e:
            print(f"  torch.{name}.is_available() 异常: {e}")
            continue
        print(f"  torch.{name}: available={ok}")
        if ok and found is None:
            found = name
            try:
                print(f"    device_count={mod.device_count()}  name={mod.get_device_name(0)}")
            except Exception as e:
                print(f"    设备信息不可用: {e}")

    if found is None:
        print("!! 探测不到任何加速器 —— auto_bench.py L494 会直接报 no accelerator available")
        return 1
    print("auto_bench 将使用的后端:", found)

    # --- ④ triton 及其来源 --------------------------------------------------
    if _TRITON_IMPORT_ERROR is not None:
        print("triton 导入失败:", _TRITON_IMPORT_ERROR)
        return 1
    print("triton:", triton.__version__)
    print("triton 来源:", triton.__file__)

    # torch 2.x 会拉一个上游 pytorch-triton 当依赖，厂商又装自己的分支，
    # 两者可能同时存在。拿错了的话 kernel 会去编译 NVIDIA 目标然后失败，
    # 报错信息通常指向 ptxas/cuda，很难联想到"装错 triton 了"。
    #
    # 唯一可靠的判据是**已注册的后端**。两个看起来更直观的判据都不管用，实测过：
    #   * triton.__file__  —— 厂商包同样装在通用的 site-packages/triton/ 下，区分不出来
    #   * triton.__version__ —— 沐曦上这里只有 "3.0.0"，厂商标记 "+metax3.3.0.2"
    #     只出现在 pip 包版本里，不进 __version__
    # 而 triton.backends 注册表在沐曦上实测返回 ['metax']，直接给出答案。
    backends = None
    try:                                   # Triton 3.x 的后端注册表
        from triton.backends import backends as _b
        backends = sorted(_b.keys())
        print("triton backends:", backends)
    except Exception:
        try:
            print("triton active driver:", type(triton.runtime.driver.active).__module__)
        except Exception:
            print("triton backends: (不可探测)")

    if backends is not None:
        vendor = [b for b in backends
                  if b.lower() not in ("nvidia", "amd", "cpu")]
        if vendor:
            print(f"triton 构建: 厂商版，后端 = {vendor}")
        else:
            print(f"triton 构建: 只注册了通用后端 {backends} —— 很可能是上游 pytorch-triton。")
            print("             若冒烟测试报 ptxas/cuda 相关错误，优先怀疑装错了 triton")

    # --- ⑤ 最小 kernel 冒烟测试 ---------------------------------------------
    # import 成功 != 工具链可用。真正要验的是：能编译出目标代码、能在卡上跑、
    # 而且结果得对（只跑不崩是不够的，编译器把 kernel 优化没了也不会崩）。
    # tl 已在模块级 import（见文件顶部说明），这里不能再写成局部 import
    @triton.jit
    def _smoke(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=m) * 2.0, mask=m)

    try:
        dev = torch.device(found)
        x = torch.randn(1024, device=dev)
        y = torch.empty_like(x)
        _smoke[(1,)](x, y, x.numel(), BLOCK=1024)
        getattr(torch, found).synchronize()
        if not torch.allclose(y, x * 2, atol=1e-5, rtol=1e-5):
            print("triton 冒烟测试: !! 编译跑通了但数值不对")
            return 1
        print("triton 冒烟测试: 通过（编译 + 执行 + 数值校验）")
    except Exception as e:
        print("triton 冒烟测试失败:", type(e).__name__, e)
        import traceback
        traceback.print_exc()
        return 1

    # --- ⑥ 本仓库真实 kernel 的编译冒烟 -------------------------------------
    # 上面那个是 tutorial 级的一维 kernel。这里再验一个更接近实战的：
    # 2D tile + 双向 reduction + 循环里的 loop-carried tile —— 正是
    # v1/hc_split_sinkhorn.py 用到的全部特性。厂商分支若在这些地方有缺失，
    # 在这里就会暴露，不用等到跑真 kernel。
    @triton.jit
    def _smoke2d(p_ptr, o_ptr, ITERS: tl.constexpr, HC: tl.constexpr):
        i = tl.arange(0, HC)[:, None]
        j = tl.arange(0, HC)[None, :]
        off = i * HC + j
        t = tl.load(p_ptr + off)
        t = tl.exp(t - tl.max(t, axis=1)[:, None])
        for _ in range(ITERS):
            t = t / (tl.sum(t, axis=1)[:, None] + 1e-6)
            t = t / (tl.sum(t, axis=0)[None, :] + 1e-6)
        tl.store(o_ptr + off, t)

    @triton.jit
    def sinkhorn_uv(c_ptr, o_ptr, ITERS: tl.constexpr, HC: tl.constexpr, EPS: tl.constexpr):
        i = tl.arange(0, HC)[:, None]
        j = tl.arange(0, HC)[None, :]
        off = i * HC + j
        K = tl.exp(-tl.load(c_ptr + off) / EPS)        # 循环外算一次，之后只读不改

        u = tl.zeros([HC], dtype=tl.float32) + 1.0
        v = tl.zeros([HC], dtype=tl.float32) + 1.0
        for _ in range(ITERS):                          # ← 可以用普通 range
            u = 1.0 / (tl.sum(K * v[None, :], axis=1) + 1e-6)
            v = 1.0 / (tl.sum(K * u[:, None], axis=0) + 1e-6)

        tl.store(o_ptr + off, u[:, None] * K * v[None, :])

    try:
        p = torch.randn(4, 4, device=dev)
        o = torch.empty_like(p)
        _smoke2d[(1,)](p, o, ITERS=20, HC=4)
        getattr(torch, found).synchronize()
        col = o.sum(dim=0)
        ok = torch.allclose(col, torch.ones_like(col), atol=1e-3, rtol=1e-3)
        print(f"2D tile / 双向 reduction / constexpr 循环: "
              f"{'通过' if ok else '!! 数值不对'}（Sinkhorn 后列和={col.tolist()}）")
        if not ok:
            return 1
    except Exception as e:
        print("2D tile 冒烟测试失败:", type(e).__name__, e)
        import traceback
        traceback.print_exc()
        print("\n-> v1/hc_split_sinkhorn.py 用的正是这些特性，需要按文件头的"
              "『跨芯片移植注意』改写（range -> tl.static_range，或改 3D tile 方案）")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
