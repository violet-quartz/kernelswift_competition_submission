"""v0 参考实现（torch baseline）— 源自 tasks/flex_attention.py

【与原题的三处差异，都不改变计算本身】
  1. `get_inputs()` 去掉了 `device="cuda"`，改返回 CPU 张量（原因见下方 KS-PORT）
  2. 加了 `_ks_bootstrap()`，并在 get_inputs / get_init_inputs 开头调用
  3. 删掉了原题末尾的 `if __name__ == "__main__"` 演示块 —— 它硬编码 `.cuda()`，
     且 auto_bench.py L74 的 _filter_module_ast() 本来就会把 ast.If 整个丢弃

`Model` 类逐字未改。原题在 tasks/flex_attention.py 里留了一份。
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
#   * `__init__` 的签名要能接住 `get_init_inputs()` 返回的 `[8, 64, None, 8]`，
#     即 (num_heads, head_size, scale, num_kv_heads)。**注意第三项是 None**，
#     原题靠 `scale or 1.0 / (head_size ** 0.5)` 兜底，实际 scale = 0.125。
#
# [KS-PORT] 本题与 mm_encoder_attention 是**同一个 kernel**（分析结论）：
#
#   两道题的数学几乎一样，差别只有两处，都能用 tl.constexpr 开关吃掉：
#              flex_attention          mm_encoder_attention
#     bsz      1（unsqueeze(0)）        2
#     因果     is_causal=True          非因果
#     其余     heads=8, head_size=64, seq=83, fp16 —— 完全相同
#   所以建议**先写本题**，再把同一个带 `IS_CAUSAL: tl.constexpr` 的
#   flash-attention kernel 复用到 mm_encoder_attention 上。
#
#   收益来源（不是算力）：本题总算力仅 8×83×83×64×2×2 ≈ 14 MFLOP，
#   seq_len=83 压根不在厂商 SDPA 的优化区间里。v0 还额外付了两笔：
#     * `unsqueeze(0).transpose(1, 2)` 三次，制造非连续张量；
#     * 末尾 `.squeeze(0).transpose(0, 1).reshape(...)` 强制一次连续化拷贝。
#   融合 kernel 直接按输出布局写回，这两笔都省掉。
#
#   还有一处死代码：`num_kv_heads == num_heads == 8`，所以
#   `if self.num_kv_heads != self.num_heads` 里的 repeat_interleave **永远不执行**，
#   v1 不需要实现 GQA 分支。
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
    def __init__(self, num_heads: int = 8, head_size: int = 64,
                 scale: float = None, num_kv_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale or 1.0 / (head_size ** 0.5)
        self.num_kv_heads = num_kv_heads

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        # query: [num_tokens, num_heads, head_size]
        # key/value: [num_tokens, num_kv_heads, head_size]
        num_tokens = query.shape[0]
        q = query.unsqueeze(0).transpose(1, 2)
        k = key.unsqueeze(0).transpose(1, 2)
        v = value.unsqueeze(0).transpose(1, 2)
        if self.num_kv_heads != self.num_heads:
            r = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(r, dim=1)
            v = v.repeat_interleave(r, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale,
                                             is_causal=True)
        return out.squeeze(0).transpose(0, 1).reshape(
            num_tokens, self.num_heads * self.head_size)


def get_inputs():
    _ks_bootstrap()
    # query: [num_tokens, num_heads, head_size], float16
    # key:   [num_tokens, num_kv_heads, head_size], float16
    # value: [num_tokens, num_kv_heads, head_size], float16
    num_tokens, num_heads, head_size = 83, 8, 64
    dtype = torch.float16
    query = torch.randn(num_tokens, num_heads, head_size, dtype=dtype)
    key   = torch.randn(num_tokens, num_heads, head_size, dtype=dtype)
    value = torch.randn(num_tokens, num_heads, head_size, dtype=dtype)
    return [query, key, value]


def get_init_inputs():
    _ks_bootstrap()
    # num_heads=8, head_size=64, scale=None（→ 0.125）, num_kv_heads=8
    return [8, 64, None, 8]
