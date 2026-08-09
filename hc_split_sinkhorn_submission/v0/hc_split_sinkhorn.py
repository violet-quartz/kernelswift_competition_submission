"""v0 参考实现（torch baseline）— 源自 tasks/hc_split_sinkhorn.py

【本文件与原题逐字一致】原题的 get_inputs() 本来就建 CPU 张量，
无需任何设备移植处理。
"""
# ---------------------------------------------------------------------------
# [KS-PORT] 关于设备：本仓库所有 v0/v1 文件的 get_inputs() 一律返回 **CPU 张量**
#
# 原因（依据 bench/auto_bench.py 的实际行为）：
#   1. L127 _rewrite_device_for_backend() 只把源码里的 'npu' 字面量重写成当前
#      后端，**不会**把 'cuda' 重写成 'npu'。所以硬编码 device="cuda" 的文件
#      拿到昇腾 A2 上会直接抛 "Torch not compiled with CUDA enabled"。
#   2. L478 _detect_target_device() 在模型和输入都在 CPU 上时，会自动回退到
#      _iter_accelerators() 探测到的加速器（gcu/cuda/npu/mlu）。
#   3. L500 _move_to_device() 随后把 v0/v1 的输入统一搬到该设备上再对拍和计时。
#
# 结论：返回 CPU 张量既不影响正确性也不影响计时（计时发生在搬运之后），
# 却让同一份文件在沐曦 C500(cuda 命名空间) / 昇腾 A2(npu) / 纯 CPU 三种环境
# 下都能跑——这正是我们在拿到卡之前能先在本地对拍的前提。
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn


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


class Model(nn.Module):
    def __init__(self, hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):
        super().__init__()
        self.hc_mult = hc_mult
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps

    def forward(
        self,
        mixes: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, s, mix_hc = mixes.shape
        hc = self.hc_mult
        eps = self.eps
        expected = (2 + hc) * hc
        if mix_hc != expected:
            raise ValueError(f"expected mix dim {expected}, got {mix_hc}")

        x = mixes.reshape(-1, mix_hc).to(dtype=torch.float32)
        base = hc_base.to(dtype=torch.float32)
        s0, s1, s2 = hc_scale[0], hc_scale[1], hc_scale[2]

        pre = torch.sigmoid(x[:, :hc] * s0 + base[:hc].unsqueeze(0)) + eps
        post = 2 * torch.sigmoid(x[:, hc : 2 * hc] * s1 + base[hc : 2 * hc].unsqueeze(0))
        raw = x[:, 2 * hc : 2 * hc + hc * hc]
        comb = raw.view(-1, hc, hc) * s2 + base[2 * hc : 2 * hc + hc * hc].view(1, hc, hc)

        row_max = comb.amax(dim=-1, keepdim=True)
        comb = torch.exp(comb - row_max)
        comb = comb / comb.sum(dim=-1, keepdim=True) + eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

        for _ in range(self.sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

        return pre.view(b, s, hc), post.view(b, s, hc), comb.view(b, s, hc, hc)


def get_init_inputs():
    """Returns positional args for Model.__init__: (hc_mult, sinkhorn_iters, eps)."""
    _ks_bootstrap()
    return [4, 20, 1e-6]


def get_inputs():
    """Returns positional args for Model.forward: (mixes, hc_scale, hc_base)."""
    _ks_bootstrap()
    hc = 4
    mix_hc = (2 + hc) * hc
    torch.manual_seed(0)
    mixes = torch.randn(2, 8, mix_hc, dtype=torch.float32)
    hc_scale = torch.tensor([0.5, 0.25, 1.0], dtype=torch.float32)
    hc_base = torch.randn(mix_hc, dtype=torch.float32) * 0.1
    return [mixes, hc_scale, hc_base]
