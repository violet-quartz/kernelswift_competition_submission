"""v1 Triton 优化实现 — head_compute_mix_bwd.py

这道题跟本仓库其它题不是一类：**它不是访存瓶颈，是固定开销瓶颈。**

规模小到没有访存问题
--------------------
输入 (2, 1024, 4) fp32 = 8192 个元素 = 32.8 KB/张量，整个 forward 读 2 写 3，
总共约 98 KB。按沐曦 C500 的实测带宽算，搬完只要零点几微秒；而实测耗时是
一百多微秒 —— 真正搬数据的部分占千分之几。

所以**调访存模式在这道题上是在优化那千分之几**。（对比 mhc_post：那题搬
189.4 MB，访存模式改一条就有 1.47x。同样的手段换个规模就完全失效，别把那边的
经验直接搬过来。）

第一版的 profiling（沐曦 C500，bench/profile_chip.py，active=5）
--------------------------------------------------------------
grid=8 + atomic_add + 两个 torch.zeros，实测 125.2 us（1.40x）：

    Self CUDA 合计 61.696/5 = 12.34 us
      _head_compute_mix_bwd_kernel_2d   6.20 us   50.2%
      aten::fill_（两个 zeros 的 memset）6.14 us   49.8%

两个 memset 跟真正干活的 kernel 一样贵，而它们清的是 1+4 个 float、共 20 字节。
它们存在的唯一理由是 atomic_add 需要零初始化。

走过的弯路：grid 8 -> 1（失败，已回滚）
--------------------------------------
既然 memset 是 atomic_add 逼出来的，那让单 program 包办全部数据、规约在寄存器
里累加，就不需要原子操作、也不需要零初始化了。当时的论证是"设备时间藏在 host
底下，并行度损失免费"。**实测证明这个论证是错的**：

    kernel      6.20 -> 69.63 us   (11.2x)
    实测总耗时  125.2 -> 129.5 us  (1.40x -> 1.35x)

省下 6.14 us 的 memset，付出 63.4 us 的 kernel。

**错在哪：** host 那一百来微秒花在 kernel 启动**之前**（分配 + Triton 的 python
启动路径），而整个 forward 只有一个 kernel —— 它没法跟自己 launch 之前的 host
工作重叠；auto_bench 又每次 forward 都 sync。所以设备时间是**加在 host 后面**
的，不是藏在底下。对着两组数反推也印证了：

    第一版   host+sync = 125.2 - 12.34 = 112.9 us
    单 program host+sync = 129.5 - 69.63 =  59.9 us

（host 确实掉了 53 us，就是两个 zeros 的分配 + 填充路径，但被 kernel 涨的
63 us 全吃掉还倒亏。这个 wall ≈ host + device + sync 的模型只有两个点撑着，
方向可信、数值别当准数。）

本版：保住赢的那一半，退掉输的那一半
------------------------------------
回到 grid=8 + atomic_add（kernel 回到 6.20 us），但**两个 zeros 合并成一个
缓冲区**：一次 memset 代替两次，host 侧也少走一整条分配路径。切片出来的
grad_mhc_scale / grad_mhc_base 是 view，不产生 kernel。

天花板
------
auto_bench 每次 forward 都 sync，那约 23 us 两边都要付、改不掉，所以
speedup_max ≈ 174.8 / 23 ≈ 7.6x。**赛题目标写的 8-12x 数学上就够不着**，
这一点在动手前就该算清楚。

再往上就得处理 host 侧 Triton 的 python 启动路径（绑参数、算 specialization、
拼 cache key 再查表；profile 里 mcPointerGetAttribute 每次迭代被调 7 次，正好
是 kernel 的 7 个指针参数）。**刻意不做那个**：把 CompiledKernel 缓存起来绕开
JITFunction.run 依赖各家 Triton 的内部 API（沐曦 3.0.0+metax 与昇腾 3.2.0 不一定
一致），一旦在某颗芯片上传错参数就是正确性挂掉、直接不参与排名；而且它省的是
评测框架的开销、不是算子本身的工作量。

调参用的环境变量
----------------
    KS_BLOCK_SIZE=1024   每个 program 处理多少个元素（grid = 8192 / 它）

**这个旋钮现在值得扫了** —— 上面那次失败恰恰证明并行度对这道题很敏感
（grid 8 -> 1 让 kernel 慢了 11 倍）。1024 对应 grid=8，可以试试 512（grid=16）
和 2048（grid=4）：往大走原子竞争少但并行度低，往小走反之。
"""
import os

import torch
import torch.nn as nn
import triton
import triton.language as tl

# 两个规约输出挤在同一个缓冲区里，grad_mhc_base 从第 _ACC_PAD 个 float 开始。
# 取 4 是为了让它的地址落在 16 字节边界上 —— Triton 的 specialization 会检查
# 指针能否被 16 整除，错开这个边界会走到另一条（更慢的）代码路径上去。
# 必须是模块级**字面量**赋值：auto_bench.py L74 的 _filter_module_ast() 只保留
# Import / ClassDef / FunctionDef / 字面量赋值，带函数调用的会被整个丢弃。
_ACC_PAD = 4


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


def _default_block_size():
    """按后端选 KS_BLOCK_SIZE 的默认值。只在 __init__ 调一次。

    [KS-PORT] 这不是分支，是**按卡取默认常量** —— kernel 一个字不用改。
    两块卡的最优点差了一个数量级，写死任何一个换台机器就是随机数：
      * 沐曦：1024（原实测值，grid=8）
      * 昇腾：64。实测 1024→0.76x、256→0.92x、64→0.92x、4096→0.44x，
        往小走明显更好；64/128/256 三者互相分不开（噪声 8.6%），取 64。
    环境变量 KS_BLOCK_SIZE 仍然覆盖本函数，扫描时用它。

    ⚠ 昇腾上这道题**翻不过 v0**，天花板 ~0.87x，原因是结构性的：
    一次 Triton kernel 启动固定 18us（空 kernel 也是），而 v0 的十一个 torch
    算子每个只要 3.4us。这个 kernel 的实际计算不到 9us —— 把两个规约连同
    atomic_add **整个删掉**，时间纹丝不动（27.0 → 27.3us）。8192 个 float
    的规模不够摊薄一次启动，别再往 kernel 里调了。
    """
    try:
        import triton
        if "ascend" in triton.backends.backends:
            return 64
    except Exception:
        pass
    return 1024


def _use_atomic_reduce():
    """能不能用 tl.atomic_add 做规约。只在 __init__ 调一次。

    [KS-PORT] 燧原 S60（triton backends=['gcu']）**不支持 atomic**，
    tt.atomic_rmw 被后端明确标为非法，编译直接失败。其余卡都支持。

    判据写成**黑名单**（已知不支持的才走回退路径），因为 atomic 是 Triton 的
    通用能力、不支持才是例外 —— 这和 K_CONTIG 那种"只有实测赢过才切"的
    白名单方向相反，因为那是性能取舍、这是能力有无。

    KS_ATOMIC_REDUCE=0/1 可覆盖。给它环境变量是有教训的：K_CONTIG 当初被当成
    纯硬能力开关、没留覆盖口，结果想在别的卡上验"回退路径是不是反而更快"时
    只能改代码。能力开关也可能同时是性能轴。
    """
    raw = os.environ.get("KS_ATOMIC_REDUCE")
    if raw:
        return raw not in ("0", "false", "False")
    try:
        import triton
        return "gcu" not in triton.backends.backends
    except Exception:
        return True


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
def _head_compute_mix_bwd_kernel(
    input_mix_ptr,       # [N0, N1, M] float32
    mhc_scale_ptr,       # [1]         float32
    mhc_base_ptr,        # [M]         float32
    grad_out_ptr,        # [N0, N1, M] float32
    grad_input_mix_ptr,  # [N0, N1, M] float32  out
    grad_mhc_scale_ptr,  # [1]         float32  out（acc 缓冲区的第 0 个元素）
    grad_mhc_base_ptr,   # [M]         float32  out（acc 缓冲区第 _ACC_PAD 个起）
    N0: tl.constexpr,
    N1: tl.constexpr,
    M: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    ATOMIC_REDUCE: tl.constexpr,   # 见 _use_atomic_reduce()
):
    """每个 program 处理 BLOCK_ROWS 行，两个规约用 atomic_add 汇总。

    M 直接进 tl.arange，所以要求它是 2 的幂（本题 M=4）。这是刻意保留的限制：
    这个 kernel 的 6.20 us 是实测过的，为了泛化去改结构就得重新量一遍，而这道题
    host-bound、kernel 本身根本不是瓶颈，不值得动。

    atomic_add 要求两个规约输出预先清零 —— 调用方用**一个** torch.zeros 缓冲区
    把它们并排放好（见 forward），而不是两个。第一版用两个，profiling 显示那两次
    memset 占了整整一半的设备时间，而它们清的一共才 20 字节。

    被 mask 掉的行读回来 x=g=0 -> grad_z = 0 * s * (1-s) = 0，对两个规约都不贡献，
    所以原子累加不需要额外的 where。（本题 n_rows=2048、BLOCK_ROWS=256，grid=8
    整除，实际不会走到 mask 分支。）
    """
    pid = tl.program_id(0)

    rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    lanes = tl.arange(0, M)
    n_rows = N0 * N1
    mask = rows[:, None] < n_rows
    offset = rows[:, None] * M + lanes[None, :]

    x = tl.load(input_mix_ptr + offset, mask=mask, other=0.0)
    g = tl.load(grad_out_ptr + offset, mask=mask, other=0.0)

    scale = tl.load(mhc_scale_ptr)
    base = tl.load(mhc_base_ptr + lanes[None, :])

    s = tl.sigmoid(x * scale + base)
    grad_z = g * s * (1.0 - s)

    tl.store(grad_input_mix_ptr + offset, grad_z * scale, mask=mask)

    # [KS-PORT] 两条规约路径，硬能力差异：
    #   * 燧原 S60 **不支持 atomic**，编译期就被拒：
    #         failed to legalize operation 'tt.atomic_rmw' that was explicitly marked illegal
    #     只能各 program 写各自的槽位，最后由 host 侧收尾。
    #   * 其余卡用 atomic 一步到位，省掉那次 host 规约。
    # ATOMIC_REDUCE 是 tl.constexpr，没被选中的那支编译期就折掉。
    if ATOMIC_REDUCE:
        tl.atomic_add(grad_mhc_scale_ptr, tl.sum(grad_z * x))
        tl.atomic_add(grad_mhc_base_ptr + lanes, tl.sum(grad_z, axis=0))
    else:
        # 每个槽位恰好被写一次，所以调用方**不需要预清零**（省掉一次 memset）。
        tl.store(grad_mhc_scale_ptr + pid, tl.sum(grad_z * x))
        tl.store(grad_mhc_base_ptr + pid * M + lanes, tl.sum(grad_z, axis=0))


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        # 环境变量在 __init__ 里读一次，不放 forward 里 —— 这道题 host-bound，
        # 被计时的路径上多一次 os.environ 查询都是实打实的成本。
        self._block_size = _env_int("KS_BLOCK_SIZE") or _default_block_size()
        self._atomic_reduce = _use_atomic_reduce()

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

        grad_input_mix = torch.empty((n0, n1, mhc_mult), dtype=torch.float32, device=dev)

        block_rows = self._block_size // mhc_mult
        num_programs = triton.cdiv(n0 * n1 * mhc_mult, self._block_size)

        if self._atomic_reduce:
            # 两个规约输出合用一个零初始化缓冲区：一次 memset 代替两次，host 侧也少走
            # 一整条分配路径。第一版那两个 torch.zeros 在 profiling 里占了一半的设备
            # 时间（6.14 / 12.34 us），而它们一共只清 20 字节。
            # 下面两行切片都是 view，不产生 kernel、不额外分配。
            acc = torch.zeros(_ACC_PAD + mhc_mult, dtype=torch.float32, device=dev)
            grad_mhc_scale = acc[:1]
            grad_mhc_base = acc[_ACC_PAD:]
        else:
            # [KS-PORT] 无 atomic 的回退：每 program 一个槽位，kernel 全写一遍，
            # 所以用 torch.empty（不需要 memset），规约留给 host 侧的两次 sum。
            part_scale = torch.empty(num_programs, dtype=torch.float32, device=dev)
            part_base = torch.empty(num_programs * mhc_mult, dtype=torch.float32, device=dev)
            grad_mhc_scale = part_scale
            grad_mhc_base = part_base

        _head_compute_mix_bwd_kernel[(num_programs,)](
            input_mix,
            mhc_scale,
            mhc_base,
            grad_out,
            grad_input_mix,
            grad_mhc_scale,
            grad_mhc_base,
            N0=n0, N1=n1, M=mhc_mult, BLOCK_ROWS=block_rows,
            ATOMIC_REDUCE=self._atomic_reduce,
        )

        if not self._atomic_reduce:
            # kernel 写的是 (num_programs,) 和 (num_programs, M)，这里收成最终形状
            grad_mhc_scale = part_scale.sum(0, keepdim=True)              # (1,)
            grad_mhc_base = part_base.view(num_programs, mhc_mult).sum(0)  # (M,)

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
