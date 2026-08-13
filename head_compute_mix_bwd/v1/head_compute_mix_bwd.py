"""v1 Triton 优化实现 — head_compute_mix_bwd.py

这道题跟本仓库其它题不是一类：**它不是访存瓶颈，是固定开销瓶颈。**

规模小到没有访存问题
--------------------
输入 (2, 1024, 4) fp32 = 8192 个元素 = 32.8 KB/张量，整个 forward 读 2 写 3，
总共约 98 KB。按沐曦 C500 实测的 382 GB/s 算，搬完只要 0.26 us；而第一版实测
125.2 us —— 真正搬数据的部分占 0.2%。

所以**调 BLOCK_SIZE、调 1D/2D 访存模式在这道题上是在优化那 0.2%**。
（对比 mhc_post：那题搬 189.4 MB，访存模式改一条就有 1.47x。同样的手段换个
规模就完全失效，别把那边的经验直接搬过来。）

profiling 实测（沐曦 C500，bench/profile_chip.py，active=5）
------------------------------------------------------------
第一版（grid=8 + atomic_add + 两个 torch.zeros）的 Self CUDA 合计 61.696 us，
折合每次迭代 12.34 us：

    _head_compute_mix_bwd_kernel_2d   30.976/5 = 6.20 us   50.2%
    aten::fill_（两个 zeros 的 memset）30.720/5 = 6.14 us   49.8%

**两个 memset 跟真正干活的 kernel 一样贵，而它们清的是 1+4 个 float、共 20 字节。**
它们存在的唯一理由是 atomic_add 需要零初始化 —— 纯粹自找的开销。这是本版要
去掉的东西。

同时，设备侧 12.34 us 对 auto_bench 实测的 125.2 us（两把不同的尺子：前者是
纯设备执行时间，后者是每次 forward 都 sync 的墙钟），**说明设备侧九成时间是
闲着的，仗在 host 上打**。这个量级结论是稳的（差一个数量级），但两者相减得到
的"host 约 113 us"不该当精确值用 —— profiler 对 host 有约 1.55x 放大
（v0 profiled 354 us 对实测 174.8 us），且 active=5 样本很少。

本版的改动：grid 8 -> 1，去掉 atomic_add 和两个 zeros
----------------------------------------------------
关键认识是**设备侧有余量**。host 远慢于设备，意味着设备时间涨一些是免费的，
于是可以拿并行度换掉固定开销：

    单 program 全包 -> 规约在寄存器里累加，最后直接 store
                    -> 不需要 atomic_add
                    -> 不需要零初始化，三个输出全用 torch.empty
                    -> 省掉 6.14 us 设备时间，外加 host 侧两次分配 + 填充

代价是并行度从 8 个 program 降到 1 个，设备时间预计从 6.2 涨到 20-40 us，
但它藏在 host 时间底下。**这个取舍只在 host-bound 时成立** —— 如果哪天规模
变大到设备时间超过 host，要立刻改回多 program + 两段式规约。

天花板
------
auto_bench 每次 forward 都 sync，那约 23 us 两边都要付、改不掉，所以
speedup_max ≈ 174.8 / 23 ≈ 7.6x。**赛题目标写的 8-12x 数学上就够不着**，
这一点在动手前就该算清楚。本版的预期是 1.9-2.1x（125 -> 85-95 us）。

再往上就得处理 host 侧 Triton 的 python 启动路径（绑参数、算 specialization、
拼 cache key 再查表；profile 里 mcPointerGetAttribute 每次迭代被调 7 次，正好
是 kernel 的 7 个指针参数）。**本版刻意不做那个**：把 CompiledKernel 缓存起来
绕开 JITFunction.run 依赖各家 Triton 的内部 API（沐曦 3.0.0+metax 与昇腾 3.2.0
不一定一致），一旦在某颗芯片上传错参数就是正确性挂掉、直接不参与排名；而且它
省的是评测框架的开销、不是算子本身的工作量。先把这一步干净的收益拿到，再用
实测数据判断那个值不值得赌。

调参用的环境变量
----------------
    KS_BLOCK_ROWS=256   每轮处理多少行

**这个旋钮大概率是死的，别急着扫。** BLOCK_ROWS * NUM_ITERS = n_rows 恒等，
单 program 里怎么切都是同一份活，它只影响设备时间；而设备时间藏在 host 底下，
不进 auto_bench 的读数。判断要不要扫的信号是改完之后 profile 里的 Self CUDA：
还远低于 host 就别碰；涨到同量级才说明 grid=1 丢掉的 8 倍并行度开始收费，
那时要动的也不是这个参数，而是整个单 program 的设计得退回多 program + 两段式规约。

256 不是随手取的 —— 两头都会出事：往大到 2048 就是一发覆盖不循环，8192 个元素
同时活着必 spill；往小到 64 则 NUM_ITERS=32、static_range 展开 32 份，编译时间
和标量地址计算都涨而收益为零。
"""
import os

import torch
import torch.nn as nn
import triton
import triton.language as tl


def _env_int(name: str):
    """读一个整数环境变量，没设或非法就返回 None。

    [KS-PORT] 为什么读环境变量必须写在函数里、不能写成模块级的
    `_BLOCK = int(os.environ.get(...))`：auto_bench.py L74 的
    _filter_module_ast() 只保留 Import / ClassDef / FunctionDef / **字面量**赋值，
    模块级带函数调用的赋值会被整个丢弃 —— 跟 _ks_bootstrap() 必须包成函数
    是同一个原因。这里的调用点在 ModelNew.__init__ 里，函数体内不受那个过滤器影响。
    """
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        print(f"[head_compute_mix_bwd] 忽略非法的 {name}={raw!r}")
        return None


def _ks_bootstrap():
    """按需导入后端扩展，让 torch.npu / torch.mlu 命名空间真正出现。

    [KS-PORT] 为什么必须有这个函数、又为什么它长这样：
      * 昇腾要 `import torch_npu`、寒武纪要 `import torch_mlu`，否则 torch 上
        压根不存在 .npu / .mlu 属性。而 auto_bench.py L206 的 _iter_accelerators()
        正是用 getattr(torch, "npu", None) 来探测设备的 —— 没导入扩展，
        它就探测不到加速器，L494 直接抛 "no accelerator device available"。
      * 沐曦 C500 走 torch.cuda 命名空间，不需要任何扩展，所以这里必须能容忍
        ImportError 而不是硬 import。
      * 那为什么不写成模块级的 try/except？因为 auto_bench.py L74 的
        _filter_module_ast() 只保留 Import / ClassDef / FunctionDef / 字面量赋值
        四类节点，模块级的 try/except 是 ast.Try，**会被整个丢弃**。
        包进函数体里才能存活 —— 函数体内部不受那个过滤器影响。
      * 调用点放在 get_init_inputs() / get_inputs() 开头，因为 auto_bench.py
        L378-409 是先调这两个函数，之后才做设备探测（L516）。
    """
    import importlib

    for _mod in ("torch_npu", "torch_mlu"):
        try:
            importlib.import_module(_mod)
        except ImportError:
            pass


@triton.jit
def _head_compute_mix_bwd_kernel_single(
    input_mix_ptr,       # [N_ROWS, M] float32
    mhc_scale_ptr,       # [1]         float32
    mhc_base_ptr,        # [M]         float32
    grad_out_ptr,        # [N_ROWS, M] float32
    grad_input_mix_ptr,  # [N_ROWS, M] float32  out
    grad_mhc_scale_ptr,  # [1]         float32  out
    grad_mhc_base_ptr,   # [M]         float32  out
    N_ROWS: tl.constexpr,
    M: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,      # = next_power_of_2(M)，tl.arange 要求 2 的幂
    NUM_ITERS: tl.constexpr,    # = cdiv(N_ROWS, BLOCK_ROWS)，宿主侧算好传进来
):
    """**单 program** 处理全部数据，两个规约在寄存器里累加、最后直接 store。

    整个 kernel 只启动 1 个 program（grid=(1,)），所以两个规约不存在跨 program
    的竞争，不需要 atomic_add，输出也就不需要零初始化 —— 这正是要省掉的那
    6.14 us memset。见文件头：这道题 host-bound，设备侧的并行度损失是免费的。

    第一版是 grid=8 + atomic_add + 两个 torch.zeros，profiling 显示那两个只清
    20 字节的 memset 占了整整一半的设备时间，本 kernel 就是为了去掉它们。

    循环用 tl.static_range 而不是 range，两个原因：
      1. 完全展开成 SSA，acc_base / acc_scale 不是循环携带变量。沐曦的
         make_ttgir 段错误触发条件之一正是"2D tile 作为循环携带变量 + 循环体内
         规约"（hc_split_sinkhorn 撞过），这里从结构上避开。
      2. NUM_ITERS 只有 8，展开的编译代价可以忽略。hc_split_sinkhorn 那次
         static_range 编译超时是因为真实规模下迭代次数太多，跟这里不是一回事。
    """
    lanes = tl.arange(0, BLOCK_M)
    lane_m = lanes < M              # M 不是 2 的幂时盖住 padding 出来的列

    # scale / base 跟行无关，循环外读一次
    scale = tl.load(mhc_scale_ptr)
    base = tl.load(mhc_base_ptr + lanes, mask=lane_m, other=0.0)

    acc_base = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_scale = tl.zeros([], dtype=tl.float32)

    for it in tl.static_range(NUM_ITERS):
        rows = it * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        m = (rows[:, None] < N_ROWS) & lane_m[None, :]
        offset = rows[:, None] * M + lanes[None, :]

        x = tl.load(input_mix_ptr + offset, mask=m, other=0.0)
        g = tl.load(grad_out_ptr + offset, mask=m, other=0.0)

        s = tl.sigmoid(x * scale + base[None, :])
        grad_z = g * s * (1.0 - s)
        # 被 mask 掉的位置读回来是 x=g=0 -> s=sigmoid(0)=0.5 -> grad_z=0，
        # 规约里天然不贡献。这里仍显式 where 一次，是不想让正确性依赖
        # "sigmoid(0)*(1-sigmoid(0)) 乘上 g=0 恰好为 0" 这种巧合。
        grad_z = tl.where(m, grad_z, 0.0)

        tl.store(grad_input_mix_ptr + offset, grad_z * scale, mask=m)

        acc_base += tl.sum(grad_z, axis=0)      # 沿行规约 -> (BLOCK_M,)
        acc_scale += tl.sum(grad_z * x)         # 全规约 -> 标量

    tl.store(grad_mhc_base_ptr + lanes, acc_base, mask=lane_m)
    tl.store(grad_mhc_scale_ptr, acc_scale)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        # 环境变量在 __init__ 里读一次，不放 forward 里 —— 这道题 host-bound，
        # 被计时的路径上多一次 os.environ 查询都是实打实的成本。
        self._block_rows = _env_int("KS_BLOCK_ROWS") or 256

    def forward(
        self,
        input_mix: torch.Tensor,
        mhc_scale: torch.Tensor,
        mhc_base: torch.Tensor,
        grad_out: torch.Tensor,
    ) -> torch.Tensor:
        """
            Manual backward computation.

            Args:
                input_mix: (n0, n1, mhc_mult)
                mhc_scale: (1,)
                mhc_base: (mhc_mult,)
                grad_out: same shape as input_mix

            Returns:
                grad_input_mix, grad_mhc_scale, grad_mhc_base
        """
        n0, n1, mhc_mult = input_mix.shape
        dev = input_mix.device

        # 三个输出全用 empty：单 program 的规约在寄存器里做完才 store，不经过
        # atomic_add，所以不需要零初始化。profiling 显示第一版那两个 torch.zeros
        # 的 memset 占了整整一半的设备时间（6.14 / 12.34 us），而它们只清 20 字节。
        grad_input_mix = torch.empty((n0, n1, mhc_mult), dtype=torch.float32, device=dev)
        grad_mhc_scale = torch.empty((1,), dtype=torch.float32, device=dev)
        grad_mhc_base = torch.empty((mhc_mult,), dtype=torch.float32, device=dev)

        n_rows = n0 * n1
        block_rows = min(self._block_rows, triton.next_power_of_2(n_rows))
        _head_compute_mix_bwd_kernel_single[(1,)](
            input_mix,
            mhc_scale,
            mhc_base,
            grad_out,
            grad_input_mix,
            grad_mhc_scale,
            grad_mhc_base,
            N_ROWS=n_rows,
            M=mhc_mult,
            BLOCK_ROWS=block_rows,
            BLOCK_M=triton.next_power_of_2(mhc_mult),
            NUM_ITERS=triton.cdiv(n_rows, block_rows),
        )
        return grad_input_mix, grad_mhc_scale, grad_mhc_base


batch0 = 2
batch1 = 1024
mhc_mult = 4


def get_inputs():
    _ks_bootstrap()
    input_mix = torch.randn(batch0, batch1, mhc_mult, dtype=torch.float32)
    mhc_scale = torch.randn(1, dtype=torch.float32)
    mhc_base = torch.randn(mhc_mult, dtype=torch.float32)
    grad_out = torch.randn(batch0, batch1, mhc_mult, dtype=torch.float32)

    return [input_mix, mhc_scale, mhc_base, grad_out]


def get_init_inputs():
    _ks_bootstrap()
    return []
