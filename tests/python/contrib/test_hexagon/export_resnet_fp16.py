import torch
import tvm

from tvm import relay


def generate_resnet50():
    from torchvision.models import resnet

    model = resnet.resnet50(pretrained=True).eval()

    pt_inp = torch.randn(1, 3, 224, 224)

    script_module = torch.jit.trace(model, pt_inp).eval()

    input_name = "image"
    input_shapes = [(input_name, pt_inp.shape)]
    mod, params = relay.frontend.from_pytorch(
        script_module, input_shapes
    )
    mod = relay.transform.ToMixedPrecision("float16")(mod)

    # with tvm.transform.PassContext(
    #     opt_level=3,
    # ):
        # opt_mod, _ = relay.optimize(mod, target="llvm", params=params)
        # print(opt_mod)
        # return

    with open("resnet50_fp16.json", "w") as fo:
        fo.write(tvm.ir.save_json(mod))

    with open("resnet50_fp16.params", "wb") as fo:
        fo.write(relay.save_param_dict(params))


def generate_resnet18():
    from torchvision.models import resnet

    model = resnet.resnet18(pretrained=True).eval()

    pt_inp = torch.randn(1, 3, 224, 224)

    script_module = torch.jit.trace(model, pt_inp).eval()

    input_name = "image"
    input_shapes = [(input_name, pt_inp.shape)]
    mod, params = relay.frontend.from_pytorch(
        script_module, input_shapes
    )
    mod = relay.transform.ToMixedPrecision("float16")(mod)

    # with tvm.transform.PassContext(
    #     opt_level=3,
    # ):
        # opt_mod, _ = relay.optimize(mod, target="llvm", params=params)
        # print(opt_mod)
        # return

    with open("resnet18_fp16.json", "w") as fo:
        fo.write(tvm.ir.save_json(mod))

    with open("resnet18_fp16.params", "wb") as fo:
        fo.write(relay.save_param_dict(params))


generate_resnet18()
