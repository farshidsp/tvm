import torch
from torch import nn
import tvm
from tvm import relay

model = torch.hub.load("PeterL1n/RobustVideoMatting", "resnet50")

rec = [None] * 4

downsample_ratio = 0.375

frame = torch.randn(1, 3, 1280, 720)


class TraceWrapper(nn.Module):
    def __init__(self, model, downsample_ratio):
        super().__init__()
        self.model = model
        self.downsample_ratio = downsample_ratio

    def forward(self, *inp):
        return self.model(*inp, downsample_ratio)

# for r in rec:
#     print(r.shape)

with torch.no_grad():
    fgr, pha, *rec = model(frame, *rec, downsample_ratio)
    model_trace = torch.jit.trace(TraceWrapper(model, downsample_ratio), [frame, *rec])

    shape_list = [("inp0", frame.shape)]

    for i, r in enumerate(rec):
        shape_list.append(("rec%d" % i, r.shape))

    print(shape_list)

    mod, params = relay.frontend.from_pytorch(model_trace, shape_list)
    mod = relay.transform.ToMixedPrecision("float16")(mod)

    print(mod)

    with open("rvm_fp16.json", "w") as fo:
        fo.write(tvm.ir.save_json(mod))

    with open("rvm_fp16.params", "wb") as fo:
        fo.write(relay.save_param_dict(params))

    # with tvm.transform.PassContext(opt_level=3):
    #     relay.build(mod, target="llvm", params=params)

        # print(opt_mod)
