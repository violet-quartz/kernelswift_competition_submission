"""v0 参考实现（torch baseline）— 源自 tasks/SPLADE_sparse_pooler.py

【与原题的三处差异，都不改变计算本身】
  1. `get_inputs()` 去掉了 `device="cuda"`，改返回 CPU 张量（原因见下方 KS-PORT）
  2. 加了 `_ks_bootstrap()`，并在 get_inputs / get_init_inputs 开头调用
  3. 删掉了原题末尾的 `if __name__ == "__main__"` 演示块 —— 它硬编码 `.cuda()`，
     且 auto_bench.py L74 的 _filter_module_ast() 本来就会把 ast.If 整个丢弃

`Model` 类逐字未改。原题在 tasks/SPLADE_sparse_pooler.py 里留了一份。
"""
# ---------------------------------------------------------------------------
# [KS-PORT] 关于设备：本仓库所有 v0/v1 文件的 get_inputs() 一律返回 **CPU 张量**
#   1. auto_bench.py L127 _rewrite_device_for_backend() 只把 'npu' 重写成当前
#      后端，**不会**把 'cuda' 重写成 'npu'。
#   2. L478 _detect_target_device() 在模型和输入都在 CPU 上时自动回退到探测到
#      的加速器；L500 _move_to_device() 再统一搬运。计时发生在搬运之后。
# ---------------------------------------------------------------------------
#
# [KS-PORT] 写 v1 时必须守住的契约：
#
#   * **子模块名必须逐字保持 `dense` / `layer_norm` / `decoder`**，且形状一致：
#       dense:      nn.Linear(768, 768)          weight [768, 768]   + bias
#       layer_norm: nn.LayerNorm(768, eps=1e-12) weight [768]        + bias
#       decoder:    nn.Linear(768, 30522)        weight [30522, 768] + bias
#     auto_bench.py L519 用 load_state_dict 把 v0 的权重灌进 ModelNew，
#     **失败是被静默吞掉的** —— 改了名字不会报错，只会让 ModelNew 拿着自己
#     随机初始化的权重去算，然后在数值对拍那步莫名其妙地挂掉。
#     （`act = nn.GELU()` 没有参数，名字无所谓，但留着更省事。）
#   * `__init__` 的签名要能接住 `get_init_inputs()` 返回的 `[768, 30522, "max"]`，
#     即 (hidden_size, vocab_size, pooling)。
#   * **forward 返回的是 `list`（4 个 [30522] 张量），不是张量、也不是 tuple。**
#     auto_bench.py L328 的 compare_values 支持 list，但会先比长度再逐项比 ——
#     v1 必须同样返回 list，返回 tuple 会当成类型不匹配直接判错。
#
# [KS-PORT] v1 的优化空间在哪、以及它的硬上限（分析结论，不是约束）：
#
#   **本题是五道剩余题里唯一一道真正撞访存 roofline 的**，别按前几道的思路
#   去期待 10x+。账是这么算的：
#     * `decoder` 权重 30522 × 768 × 4B = **93.8 MB**，每次 forward 必读一遍，
#       谁也绕不开。按 mhc_post 实测的沐曦有效带宽 382 GB/s，**地板就是 ~245µs**。
#     * 能省的只有：`[83, 30522]` 中间张量（10.1 MB）在 log1p/relu 前后的两次
#       往返、`seq_lens.tolist()` 那**一次 host 同步**、以及 pooling 循环的
#       4 次 kernel 启动。
#   合起来撑死 ~2x，而要吃满得写带 fused epilogue 的分块 GEMM —— 五道题里
#   **代码量最大、回报最低**，所以排在最后。
#
#   廉价保底方案（时间紧时走这条）：`decoder` 保持 torch nn.Linear 不动
#   （厂商 GEMM 本来就最优），只用 Triton 融合后面的 `log1p(relu(x))` +
#   分段 pooling，顺带干掉 `.tolist()` 的同步。工作量小一个数量级，
#   大概 1.3-1.6x，先保证"完成"。
#
#   注意 pooling 的分段边界：seq_lens = [20, 25, 18, 20]，和为 83，正好等于
#   hidden_states 的行数；`pooling="max"` 走 `chunk.max(dim=0).values` 这一支。
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

    for _mod in ("torch_npu", "torch_mlu"):
        try:
            importlib.import_module(_mod)
        except ImportError:
            pass


class Model(nn.Module):
    """SPLADESparsePooler: MLM head logits → ReLU log(1+x) pooled over sequence (max or sum)."""

    def __init__(
        self,
        hidden_size: int = 768,
        vocab_size: int = 30522,
        pooling: str = "max",
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=True)
        self.pooling = pooling

    def forward(self, hidden_states: torch.Tensor, seq_lens: torch.Tensor) -> list:
        # run MLM head
        x = self.decoder(self.layer_norm(self.act(self.dense(hidden_states))))
        # SPLADE activation: log(1 + relu(logits))
        x = torch.log1p(F.relu(x))
        # pool per sequence
        result = []
        offset = 0
        for L in seq_lens.tolist():
            chunk = x[offset:offset + L]   # [L, vocab]
            if self.pooling == "max":
                result.append(chunk.max(dim=0).values)
            else:
                result.append(chunk.sum(dim=0))
            offset += L
        return result


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
