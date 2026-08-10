import importlib.util
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent  # 按你实际路径调整
spec = importlib.util.spec_from_file_location(
    "v0_probe", ROOT / "v0" / "mhc_post.py"
)
v0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v0)

device = "cuda"  # 沐曦上 torch.cuda 可用；按 auto_bench.py 的探测顺序，这里通常就是它
model = v0.Model(*v0.get_init_inputs()).to(device).eval()
inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in v0.get_inputs()]

compiled = torch.compile(model)
with torch.no_grad():
    compiled.forward(*inputs)