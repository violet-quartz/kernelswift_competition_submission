"""v1 Triton 优化实现 — head_compute_mix_bwd.py
"""
import torch
import torch.nn as nn

import triton
import triton.language as tl
import os

def _env_int(name: str):
    """读一个整数环境变量，没设或非法就返回 None。

    [KS-PORT] 为什么读环境变量必须写在函数里、不能写成模块级的
    `_BLOCK_H = int(os.environ.get(...))`：auto_bench.py L74 的
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
        print(f"[mhc_post] 忽略非法的 {name}={raw!r}")
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
def _head_compute_mix_bwd_kernel_1d(
    input_mix_ptr,  # [N0, N1, M] float32
    mhc_scale_ptr,  # [1] float32
    mhc_base_ptr,   # [M] float32
    grad_out_ptr,   # [N0, N1, M] float32
    grad_input_mix_ptr,  # [N0, N1, M] float32
    grad_mhc_scale_ptr,  # [1] float32
    grad_mhc_base_ptr,   # [M] float32
    N0: tl.constexpr,
    N1: tl.constexpr,
    M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    n_elements = N0 * N1 * M

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    input_mix_offset = input_mix_ptr + offset
    mask = offset < n_elements
    lane = offset % M

    scale = tl.load(mhc_scale_ptr)  
    base = tl.load(mhc_base_ptr + lane)

    x = tl.load(input_mix_offset, mask=mask, other=0.0)
    g = tl.load(grad_out_ptr + offset, mask=mask, other=0.0)

    s = tl.sigmoid(x * scale + base)
    grad_z = g * s * (1.0 - s)
    grad_z = tl.where(mask, grad_z, 0.0)

    # ---- 逐元素输出 ----
    tl.store(grad_input_mix_ptr + offset, grad_z * scale, mask=mask)

    # ---- grad_mhc_scale：全局标量规约 ----
    tl.atomic_add(grad_mhc_scale_ptr, tl.sum(grad_z * x))

    # ---- grad_mhc_base：按 lane 分组规约 ----
    for k in tl.static_range(M):            # M 是 constexpr，这个循环会被完全展开
        tl.atomic_add(grad_mhc_base_ptr + k, tl.sum(tl.where(lane == k, grad_z, 0.0)))

@triton.jit
def _head_compute_mix_bwd_kernel_2d(
    input_mix_ptr,  # [N0, N1, M] float32
    mhc_scale_ptr,  # [1] float32
    mhc_base_ptr,   # [M] float32
    grad_out_ptr,   # [N0, N1, M] float32
    grad_input_mix_ptr,  # [N0, N1, M] float32
    grad_mhc_scale_ptr,  # [1] float32
    grad_mhc_base_ptr,   # [M] float32
    N0: tl.constexpr,
    N1: tl.constexpr,
    M: tl.constexpr,
    BLOCK_ROWS: tl.constexpr
):
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
    tl.atomic_add(grad_mhc_scale_ptr, tl.sum(grad_z * x))
    tl.atomic_add(grad_mhc_base_ptr + lanes, tl.sum(grad_z, axis=0))



class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self._block_size = _env_int("KS_BLOCK_SIZE") or 1024
        self._use_1d = _env_int("KS_USE_1D") or 0

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

        grad_input_mix = torch.empty((n0, n1, mhc_mult), dtype=torch.float32, device=input_mix.device)
        grad_mhc_scale = torch.zeros((1,), dtype=torch.float32, device=input_mix.device)
        grad_mhc_base = torch.zeros((mhc_mult,), dtype=torch.float32, device=input_mix.device)

        assert self._block_size % mhc_mult == 0
        assert (self._block_size // mhc_mult) & (self._block_size // mhc_mult - 1) == 0
        num_programs = triton.cdiv(n0 * n1 * mhc_mult, self._block_size)

        if self._use_1d:
            _head_compute_mix_bwd_kernel_1d[(num_programs, )](
                input_mix,
                mhc_scale,
                mhc_base,
                grad_out,
                grad_input_mix,
                grad_mhc_scale,
                grad_mhc_base,
                N0=n0, N1=n1, M=mhc_mult, BLOCK_SIZE=self._block_size
            )
        else:
            block_rows = self._block_size // mhc_mult
            _head_compute_mix_bwd_kernel_2d[(num_programs, )](
                input_mix,
                mhc_scale,
                mhc_base,
                grad_out,
                grad_input_mix,
                grad_mhc_scale,
                grad_mhc_base,
                N0=n0, N1=n1, M=mhc_mult, BLOCK_ROWS=block_rows
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
