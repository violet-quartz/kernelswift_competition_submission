import torch
import torch.nn as nn
import torch.nn.functional as F
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
def _splade_pool_kernel(
):
    pass


class ModelNew(nn.Module):
    """SPLADESparsePooler: MLM head logits → ReLU log(1+x) pooled over sequence (max or sum)."""

    def __init__(
        self,
        hidden_size: int = 768,
        vocab_size: int = 30522,
        pooling: str = "max",
    ):
        super().__init__()
        # ⚠️ 子模块名必须逐字保持 dense / layer_norm / decoder，形状也要一致。
        #    auto_bench.py L519 的 load_state_dict 失败是**静默**的 ——
        #    改名不会报错，只会让随机初始化的权重参与计算，然后数值对拍莫名挂掉。
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=True)
        self.pooling = pooling

    def forward(self, hidden_states: torch.Tensor, seq_lens: torch.Tensor) -> list:
        # hidden_states: [83, 768] float32   seq_lens: [4] int32，和为 83
        # ⚠️ 返回类型必须是 **list**（4 个 [30522] 张量），不能是 tuple ——
        #    auto_bench.py L328 的 compare_values 会先比类型。
        #
        # 本题是唯一一道撞访存 roofline 的：decoder 权重 93.8 MB 必读，
        # 按沐曦实测 382 GB/s 地板就是 ~245µs，天花板约 2x。
        # 廉价保底方案：decoder 保持 torch nn.Linear，只用 Triton 融合
        # log1p(relu(x)) + 分段 pooling，顺带干掉 seq_lens.tolist() 的 host 同步。
        # 详见 v0/SPLADE_sparse_pooler.py 顶部的 KS-PORT 说明。
        raise NotImplementedError


def get_inputs():
    _ks_bootstrap()
    # hidden_states: [83, 768], float32
    # seq_lens: [4], int32，和为 83
    seq_lens = torch.tensor([20, 25, 18, 20], dtype=torch.int32)
    hidden_states = torch.randn(83, 768)
    return [hidden_states, seq_lens]


def get_init_inputs():
    _ks_bootstrap()
    # hidden_size=768, vocab_size=30522, pooling="max"
    return [768, 30522, "max"]
