import numpy as np

import torch
from torch import nn

from torch.quantization import (
    QuantStub,
    DeQuantStub,
    fuse_modules,
    QuantWrapper,
)

import tvm
from tvm import relay


def get_qconfig(per_channel):
    from torch.quantization.observer import MovingAverageMinMaxObserver
    from torch.quantization.observer import default_weight_observer

    if per_channel:
        return torch.quantization.get_default_qconfig("fbgemm")
    else:
        act = MovingAverageMinMaxObserver.with_args(reduce_range=False)
        return torch.quantization.QConfig(activation=act, weight=default_weight_observer)


def quantize_model(model, inp, per_channel=False):
    model.fuse_model()
    model.qconfig = get_qconfig(per_channel)
    torch.quantization.prepare(model, inplace=True)
    model(inp)
    torch.quantization.convert(model, inplace=True)


class Linear(nn.Module):
    def __init__(self, N, K, with_relu=False):
        super().__init__()
        layers = [nn.Linear(K, N)]
        if with_relu:
            layers.append(nn.ReLU())
        self.fc = nn.Sequential(*layers)
        self.quant_wrap = QuantWrapper(self.fc)
        self.with_relu = with_relu

    def forward(self, x):
        return self.quant_wrap(x)

    def fuse_model(self):
        if self.with_relu:
            fuse_modules(self.fc, ["0", "1"], inplace=True)


class ConvBn(nn.Module):
    def __init__(self, I, O, kernel, with_relu=False):
        super().__init__()
        layers = [nn.Conv2d(I, O, kernel, bias=True), nn.BatchNorm2d(O)]
        if with_relu:
            layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)
        self.quant_wrap = QuantWrapper(self.conv)
        self.with_relu = with_relu

    def forward(self, x):
        return self.quant_wrap(x)

    def fuse_model(self):
        indices = ["0", "1"]
        if self.with_relu:
            indices.append("2")
        fuse_modules(self.conv, indices, inplace=True)


def generate_quantized_conv2d():
    I = 64
    O = 64
    H = 56
    W = 56
    kH = 3
    kW = 3

    input_name = "input"

    per_channel = True
    ishape = (1, I, H, W)
    raw_module = ConvBn(I, O, kH)

    raw_module.eval()
    inp = torch.rand(ishape)

    quantize_model(raw_module, inp, per_channel=per_channel)

    script_module = torch.jit.trace(raw_module, inp).eval()

    with torch.no_grad():
        pt_result = script_module(inp.clone()).numpy()

    input_shapes = [(input_name, ishape)]

    mod, params = relay.frontend.from_pytorch(script_module, input_shapes)

    with open("qnn_conv2d.json", "w") as fo:
        fo.write(tvm.ir.save_json(mod))

    with open("qnn_conv2d.params", "wb") as fo:
        fo.write(relay.save_param_dict(params))

    np.save("qconv2d_input.npy", inp.numpy())
    np.save("qconv2d_output.npy", pt_result)


def generate_quantized_dense():
    M = 128
    N = 768
    K = 768

    input_name = "input"

    per_channel = True
    ishape = (M, K)
    raw_module = Linear(N, K)

    raw_module.eval()
    inp = torch.rand(ishape)

    quantize_model(raw_module, inp, per_channel=per_channel)

    script_module = torch.jit.trace(raw_module, inp).eval()

    with torch.no_grad():
        pt_result = script_module(inp.clone()).numpy()

    input_shapes = [(input_name, ishape)]

    mod, params = relay.frontend.from_pytorch(script_module, input_shapes)

    with open("qnn_dense_m%dn%dk%d.json" % (M, N, K), "w") as fo:
        fo.write(tvm.ir.save_json(mod))

    with open("qnn_dense_m%dn%dk%d.params" % (M, N, K), "wb") as fo:
        fo.write(relay.save_param_dict(params))

    np.save("input_m%dn%dk%d.npy" % (M, N, K), inp.numpy())
    np.save("output_m%dn%dk%d.npy" % (M, N, K), pt_result)


def generate_qresnet50():
    from torchvision.models.quantization import resnet as qresnet

    model = qresnet.resnet50(pretrained=True).eval()

    pt_inp = torch.randn(1, 3, 224, 224)
    quantize_model(model, pt_inp, per_channel=True)

    script_module = torch.jit.trace(model, pt_inp).eval()

    input_name = "image"
    input_shapes = [(input_name, pt_inp.shape)]
    mod, params = relay.frontend.from_pytorch(
        script_module, input_shapes, keep_quantized_weight=True
    )

    with open("qresnet50.json", "w") as fo:
        fo.write(tvm.ir.save_json(mod))

    with open("qresnet50.params", "wb") as fo:
        fo.write(relay.save_param_dict(params))


generate_qresnet50()
# generate_quantized_conv2d()
# generate_quantized_dense()
