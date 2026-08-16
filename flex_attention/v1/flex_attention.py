import torch
import torch.nn as nn
import triton
import triton.language as tl


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
def _flash_attention_kernel(
):
    # 建议把 IS_CAUSAL 做成 tl.constexpr —— 同一个 kernel 还要复用到
    # mm_encoder_attention（非因果、bsz=2）那道题上。
    pass


class ModelNew(nn.Module):
    def __init__(self, num_heads: int = 8, head_size: int = 64,
                 scale: float = None, num_kv_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        # ⚠️ get_init_inputs() 传进来的 scale 是 None，靠这个 or 兜底 → 0.125
        self.scale = scale or 1.0 / (head_size ** 0.5)
        self.num_kv_heads = num_kv_heads

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        # query/key/value: [num_tokens, num_heads, head_size]  float16
        # 返回: [num_tokens, num_heads * head_size]  float16
        #
        # 因果注意力（is_causal=True），bsz=1，seq=83。
        # num_kv_heads == num_heads == 8，v0 里的 repeat_interleave 是死代码，
        # 不需要实现 GQA。
        # 详见 v0/flex_attention.py 顶部的 KS-PORT 说明。
        raise NotImplementedError


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
