"""v0 参考实现（torch baseline）— 源自 tasks/music_flamingo_rotary_embedding.py

【与原题的三处差异，都不改变计算本身】
  1. `get_inputs()` 去掉了 `device="cuda"`，改返回 CPU 张量（原因见下方 KS-PORT）
  2. 加了 `_ks_bootstrap()`，并在 get_inputs / get_init_inputs 开头调用
  3. 删掉了原题末尾的 `if __name__ == "__main__"` 演示块 —— 它硬编码 `.cuda()`，
     且 auto_bench.py L74 的 _filter_module_ast() 本来就会把 ast.If 整个丢弃

`Model` 类本身逐字未改。原题在 tasks/music_flamingo_rotary_embedding.py 里留了一份。
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
#
# 本题的额外一条：`get_inputs()` 返回的第二项 `SEQ` 是**普通 python int**，
# 不是张量。它会原样传给 forward 的 seq_len 参数，不参与设备搬运。
# ---------------------------------------------------------------------------
#
# [KS-PORT] 写 v1 时必须守住的两条契约（auto_bench.py 的行为决定的）：
#   * **buffer 名必须保持 `inv_freq` 和 `position_angles`。** auto_bench.py L519
#     用 load_state_dict 把 v0 的参数/buffer 灌进 ModelNew，**失败是被静默吞掉的**
#     —— 改了名字不会报错，只会让 ModelNew 拿着自己随机初始化的 buffer 去算，
#     然后在数值对拍那一步莫名其妙地挂掉。
#   * **`__init__` 的签名要能接住 `get_init_inputs()` 返回的 `[64, 256, 10000.0]`**，
#     即 (dim, max_seq_len, base)。
# ---------------------------------------------------------------------------

import math

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

    # [KS-PORT] torch_musa 是后加的：摩尔线程实测确认，**不显式导入 torch_musa
    # 的话 torch.musa 压根不存在**（getattr(torch, "musa") is None），
    # auto_bench 的设备探测自然也就找不到加速器。
    # ⚠ 但只加这一行还不够 —— auto_bench.py L213 的 _iter_accelerators() 只遍历
    #   (gcu, cuda, npu, mlu)，musa 不在其中。MTT S4000 上实测：即使 torch.musa
    #   可用，_iter_accelerators() 仍返回 []，_detect_target_device() 直接抛
    #   "no accelerator device available"；而 sync_devices() 也会变成空操作。
    #   这是**评测脚本侧的缺口**，需要赛方把 musa 加进那个列表；这里先把我们
    #   这半边做对，等对面支持时立刻可用，且对其它卡零副作用。
    for _mod in ("torch_npu", "torch_mlu", "torch_musa"):
        try:
            importlib.import_module(_mod)
        except ImportError:
            pass


class Model(nn.Module):
    """MusicFlamingoRotaryEmbedding: batch (song) + time positional embedding.
    Returns (cos, sin) where cos/sin combines batch and time frequencies."""

    def __init__(
        self,
        dim: int = 64,
        max_seq_len: int = 256,
        base: float = 10000.0,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (
            torch.arange(0, dim, 2, dtype=torch.float) / dim
        ))
        self.register_buffer("inv_freq", inv_freq)

        positions = torch.arange(max_seq_len, dtype=torch.float)
        positions_norm = positions / max_seq_len * (2 * math.pi)
        position_angles = positions_norm.unsqueeze(-1) * inv_freq
        position_angles = position_angles.repeat_interleave(2, dim=-1)
        self.register_buffer("position_angles", position_angles)

    def forward(self, timestamps: torch.Tensor, seq_len: int):
        batch_positions = torch.arange(
            timestamps.shape[0], device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        batch_positions = batch_positions / self.max_seq_len
        batch_freqs = batch_positions.unsqueeze(-1) * self.inv_freq
        batch_freqs = batch_freqs.repeat_interleave(2, dim=-1)

        batch_freqs = batch_freqs[:, None, :]
        time_freqs = self.position_angles[:seq_len][None, :, :]
        batch_freqs, time_freqs = torch.broadcast_tensors(batch_freqs, time_freqs)
        freqs = torch.cat((batch_freqs, time_freqs), dim=-1)
        angle = (-timestamps * 2 * math.pi).to(freqs)
        freqs = freqs * angle.unsqueeze(-1)
        return freqs.cos(), freqs.sin()


def get_inputs():
    _ks_bootstrap()
    # timestamps: [batch_size, seq_len] — normalized song timestamps per time step
    B, SEQ = 4, 32
    timestamps = torch.rand(B, SEQ)
    return [timestamps, SEQ]


def get_init_inputs():
    _ks_bootstrap()
    return [64, 256, 10000.0]
