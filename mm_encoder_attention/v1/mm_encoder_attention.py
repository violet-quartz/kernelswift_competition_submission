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
    # 本题建议直接复用 flex_attention 那道题写好的 kernel，
    # 传 IS_CAUSAL=False，grid 上多一个 batch 维即可。
    pass


class ModelNew(nn.Module):
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
        # query/key/value: [bsz, seq_len, num_heads * head_size]  float16
        # 返回: [bsz, seq_len, num_heads * head_size]  float16
        #
        # **非因果**全连接注意力，bsz=2, seq=83, heads=8, head_size=64。
        # num_kv_heads == num_heads == 8，不需要实现 GQA。
        # 收益主要来自省掉 v0 的 transpose 非连续张量和末尾的连续化拷贝 ——
        # 融合 kernel 直接按 [B, S, H*D] 布局写回。
        # 详见 v0/mm_encoder_attention.py 顶部的 KS-PORT 说明。
        raise NotImplementedError


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
