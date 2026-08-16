"""v0 参考实现（torch baseline）— 源自 tasks/centre_random_augmentation.py

【与原题的两处差异，都不改变计算本身】
  1. `get_inputs()` 去掉了 `device='cuda'`，改返回 CPU 张量（原因见下方 KS-PORT）
  2. 加了 `_ks_bootstrap()`，并在 get_inputs / get_init_inputs 开头调用

`Model`、`centre_random_augmentation`、`random_rotation_matrices`、`rot_vec_mul`
四者逐字未改。原题没有 `if __name__ == "__main__"` 块，无需删除。
原题在 tasks/centre_random_augmentation.py 里留了一份。
"""
# ---------------------------------------------------------------------------
# [KS-PORT] 关于设备：本仓库所有 v0/v1 文件的 get_inputs() 一律返回 **CPU 张量**
#
# 原因（依据 bench/auto_bench.py 的实际行为）：
#   1. L127 _rewrite_device_for_backend() 只把源码里的 'npu' 字面量重写成当前
#      后端，**不会**把 'cuda' 重写成 'npu'。所以硬编码 device='cuda' 的文件
#      拿到昇腾 A2 上会直接抛 "Torch not compiled with CUDA enabled"。
#   2. L478 _detect_target_device() 在模型和输入都在 CPU 上时，会自动回退到
#      _iter_accelerators() 探测到的加速器（gcu/cuda/npu/mlu）。
#   3. L500 _move_to_device() 随后把 v0/v1 的输入统一搬到该设备上再对拍和计时。
# ---------------------------------------------------------------------------
#
# [KS-PORT] 写 v1 时必须守住的契约 —— 本题只有一条，但它是**全仓库最硬的一条**：
#
#   **forward 里的 4 次 RNG 调用必须原样保留，次数、顺序、形状、dtype 一个都不能变。**
#
#   顺序是（`random_rotation_matrices` 里 3 次 + `centre_random_augmentation` 里 1 次）：
#       ① torch.rand(n_sample)        -> u1
#       ② torch.rand(n_sample)        -> u2
#       ③ torch.rand(n_sample)        -> u3
#       ④ torch.randn(n_sample, 3)    -> T
#
#   为什么这条是硬性的：auto_bench.py L440 在**每次计时 forward 之前**都调
#   set_seed(seed)，L378-420 在构造模型和精度对拍前也都重新播种。也就是说
#   v0 和 v1 的每一次 forward 都从同一个 RNG 状态出发 —— 只要 v1 消费随机数的
#   方式和 v0 完全一致，两边拿到的 u1/u2/u3/T 就逐位相同；差一次调用、差一个
#   形状，整个随机数流就错位，后面的数值对拍必挂，而且报错不会指向真正原因。
#
#   推论（给 v1 的实现建议，不是约束）：这 4 次 RNG 张量极小（n_sample=4，
#   总共 4+4+4+12=24 个 float），**留在 torch 里调用最省事也最安全**，把
#   u1/u2/u3/T 当普通输入传进 Triton kernel 即可。四元数→旋转矩阵的转换、
#   去中心化、rot_vec_mul、加平移、mask 乘法这些才是该融合的部分。
#
#   另外两条（比上面那条弱，但也别踩）：
#     * `Model` 没有任何 nn.Parameter / register_buffer，`state_dict()` 是空的，
#       L519 的 load_state_dict 天然不会出问题。
#     * `__init__` 的签名要能接住 `get_init_inputs()` 返回的 `[4, 1.0, False]`，
#       即 (n_sample, s_trans, centre_only)。centre_only=False，所以 RNG 那条
#       路径一定会走到；mask 非 None，所以中心用的是掩码加权平均、且末尾还有
#       一次 `x * mask` 的乘法。
# ---------------------------------------------------------------------------

import math
from typing import Optional
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


def random_rotation_matrices(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    生成 n 个随机旋转矩阵 [n,3,3]，基于随机四元数（均匀分布）。
    """
    u1 = torch.rand(n, device=device, dtype=dtype) # (n,)
    u2 = torch.rand(n, device=device, dtype=dtype)
    u3 = torch.rand(n, device=device, dtype=dtype)

    q1 = torch.sqrt(1 - u1) * torch.sin(2 * math.pi * u2) # (n,)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * math.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2 * math.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2 * math.pi * u3)
    # quaternion (x,y,z,w)
    x, y, z, w = q1, q2, q3, q4

    # convert to rotation matrix
    xx, yy, zz = x * x, y * y, z * z # (n,)
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = torch.stack(
        [
            1 - 2 * (yy + zz),
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            1 - 2 * (xx + zz),
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            1 - 2 * (xx + yy),
        ],
        dim=-1,
    ).reshape(n, 3, 3)
    return R


def rot_vec_mul(r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    r: [...,3,3], t: [...,3]
    """
    x, y, z = torch.unbind(t, dim=-1)
    return torch.stack(
        [
            r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z,
            r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z,
            r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z,
        ],
        dim=-1,
    )


def centre_random_augmentation(
    x_input_coords: torch.Tensor, # (n_atom, 3)
    n_sample: int = 1,
    s_trans: float = 1.0,
    centre_only: bool = False,
    mask: Optional[torch.Tensor] = None, # (n_atom，)
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Args:
        x_input_coords: [N_atom, 3]
        mask: [N_atom] 0/1 (可选)
    Returns:
        x_aug: [n_sample, N_atom, 3]
    """
    device = x_input_coords.device
    dtype = x_input_coords.dtype

    if mask is None:
        center = x_input_coords.mean(dim=-2, keepdim=True)
    else:
        m = mask.to(dtype=dtype).unsqueeze(-1) # (n_atom, 1)
        center = (x_input_coords * m).sum(dim=-2, keepdim=True) / (m.sum(dim=-2, keepdim=True) + eps) # (1, 3)
    x = x_input_coords - center # (n_atom, 3)
    x = x.unsqueeze(0).expand(n_sample, -1, -1).contiguous() # (n_sample, n_atom, 3)

    if centre_only:
        return x

    R = random_rotation_matrices(n_sample, device=device, dtype=dtype)  # [n_sample,3,3]
    T = s_trans * torch.randn(n_sample, 3, device=device, dtype=dtype) # (n_sample, 3)
    x = rot_vec_mul(R[:, None, :, :].expand(-1, x.shape[1], -1, -1), x) + T[:, None, :] # (n_sample, n_atom, 3)

    if mask is not None:
        x = x * mask.to(dtype=dtype)[None, :, None] # (n_sample, n_atom, 3)
    return x


class Model(nn.Module):
    def __init__(self, n_sample: int = 1, s_trans: float = 1.0, centre_only: bool = False):
        super().__init__()
        self.n_sample = n_sample
        self.s_trans = s_trans
        self.centre_only = centre_only

    def forward(self, x_input_coords: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return centre_random_augmentation(
            x_input_coords=x_input_coords,
            n_sample=self.n_sample,
            s_trans=self.s_trans,
            centre_only=self.centre_only,
            mask=mask,
        )


# ==========================================
# Hyperparameters & Data Generation
# ==========================================

N_ATOM = 256
N_SAMPLE = 4
S_TRANS = 1.0
CENTRE_ONLY = False


def get_inputs():
    _ks_bootstrap()
    torch.manual_seed(42)

    # x_input_coords: [N_ATOM, 3], float32
    # mask: [N_ATOM], float32，全 1
    x_input_coords = torch.randn(N_ATOM, 3)
    mask = torch.ones(N_ATOM, dtype=torch.float32)

    return [x_input_coords, mask]


def get_init_inputs():
    _ks_bootstrap()
    # n_sample=4, s_trans=1.0, centre_only=False
    return [N_SAMPLE, S_TRANS, CENTRE_ONLY]
