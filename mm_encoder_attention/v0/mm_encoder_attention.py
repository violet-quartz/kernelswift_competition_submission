"""v0 参考实现（torch baseline）— 源自 tasks/mm_encoder_attention.py

【与原题的三处差异，都不改变计算本身】
  1. `get_inputs()` 去掉了 `device="cuda"`，改返回 CPU 张量（原因见下方 KS-PORT）
  2. 加了 `_ks_bootstrap()`，并在 get_inputs / get_init_inputs 开头调用
  3. 删掉了原题末尾的 `if __name__ == "__main__"` 演示块 —— 它硬编码 `.cuda()`，
     且 auto_bench.py L74 的 _filter_module_ast() 本来就会把 ast.If 整个丢弃

`Model` 类逐字未改。原题在 tasks/mm_encoder_attention.py 里留了一份。
"""
# ---------------------------------------------------------------------------
# [KS-PORT] 关于设备：本仓库所有 v0/v1 文件的 get_inputs() 一律返回 **CPU 张量**
#   1. auto_bench.py L127 _rewrite_device_for_backend() 只把 'npu' 重写成当前
#      后端，**不会**把 'cuda' 重写成 'npu'。
#   2. L478 _detect_target_device() 在模型和输入都在 CPU 上时自动回退到探测到
#      的加速器；L500 _move_to_device() 再统一搬运。计时发生在搬运之后。
# ---------------------------------------------------------------------------
#
# [KS-PORT] 写 v1 时的契约：
#   * `Model` 没有任何 nn.Parameter / register_buffer，`state_dict()` 是空的，
#     L519 的 load_state_dict 天然不会出问题。
#   * `__init__` 的签名要能接住 `get_init_inputs()` 返回的 `[8, 64, 8]`，
#     即 (num_heads, head_size, num_kv_heads)。scale 不是入参，由
#     `1.0 / (head_size ** 0.5)` 在 __init__ 里算出，实际 = 0.125。
#
# [KS-PORT] 本题与 flex_attention 是**同一个 kernel**（分析结论）：
#
#              flex_attention          mm_encoder_attention（本题）
#     bsz      1                       2
#     因果     is_causal=True          **非因果**（全连接注意力）
#     输入布局 [T, H, D] 三维           [B, S, H*D] 三维，forward 里 view 成 4 维
#     其余     heads=8, head_size=64, seq=83, fp16 —— 完全相同
#
#   建议**先做 flex_attention**，那道题写好带 `IS_CAUSAL: tl.constexpr` 的
#   flash-attention kernel 之后，本题基本只需要改 grid（多一个 batch 维）和
#   把 IS_CAUSAL 传 False，是五道剩余题里边际成本最低的一道。
#
#   收益来源（不是算力）：总算力仅 2×8×83×83×64×2×2 ≈ 28 MFLOP。v0 的成本
#   在 `view + transpose(1,2)` 造出的非连续张量，以及末尾
#   `out.transpose(1, 2).reshape(bsz, q_len, -1)` 强制的连续化拷贝 ——
#   融合 kernel 直接按 [B, S, H*D] 的输出布局写回，这笔省掉。
#
#   同样有死代码：`num_kv_heads == num_heads == 8`，v1 不需要实现 GQA。
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, num_heads: int = 8, head_size: int = 64, num_kv_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.scale = 1.0 / (head_size ** 0.5)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """forward_native → _forward_sdpa with cu_seqlens=None.
        Inputs: [bsz, seq_len, num_heads * head_size]
        """
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        # Reshape to [bsz, seq, num_heads, head_size]
        q = query.view(bsz, q_len, self.num_heads, self.head_size).transpose(1, 2)
        k = key.view(bsz, kv_len, self.num_kv_heads, self.head_size).transpose(1, 2)
        v = value.view(bsz, kv_len, self.num_kv_heads, self.head_size).transpose(1, 2)
        # scaled_dot_product_attention: [bsz, num_heads, seq, head_size]
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        # Reshape back to [bsz, seq, num_heads * head_size]
        return out.transpose(1, 2).reshape(bsz, q_len, -1)


def get_inputs():
    _ks_bootstrap()
    # query: [bsz, q_len, num_heads * head_size], float16
    # key:   [bsz, kv_len, num_kv_heads * head_size], float16
    # value: [bsz, kv_len, num_kv_heads * head_size], float16
    bsz, seq_len, num_heads, head_size, dtype = 2, 83, 8, 64, torch.float16
    hidden = num_heads * head_size
    query = torch.randn(bsz, seq_len, hidden, dtype=dtype)
    key = torch.randn(bsz, seq_len, hidden, dtype=dtype)
    value = torch.randn(bsz, seq_len, hidden, dtype=dtype)
    return [query, key, value]


def get_init_inputs():
    _ks_bootstrap()
    # num_heads, head_size, num_kv_heads
    return [8, 64, 8]
