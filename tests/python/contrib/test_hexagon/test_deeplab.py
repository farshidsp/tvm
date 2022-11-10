"""
Run end to end model using meta schedual on fp16
"""
import tempfile
import pathlib
from re import T
import numpy as np
import onnx
import tvm
from tvm import relay
from tvm import meta_schedule as ms
from tvm.contrib.hexagon.meta_schedule import get_hexagon_local_builder, get_hexagon_rpc_runner
from tvm.relay.backend import Executor

from tvm.meta_schedule.cost_model.xgb_model import XGBModel

TUNE = False
SAVE_TUNING_DATABSE = True
EXECUTOR = relay.backend.Executor("graph", {"link-params": True})

# model = tvm.testing.parameter("resnet", "inception", "srgan")
model = tvm.testing.parameter("deeplab")
target_hexagon = tvm.target.hexagon("v69")
target_llvm = tvm.target.Target("llvm")

def convert_conv2d_layout(mod, desired_layouts):
    with tvm.transform.PassContext(opt_level=3):
        seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
        return seq(mod)


def tune_ms(mod, params, hexagon_launcher, work_dir):
    # This line is necessary for link-params to take effect during
    # task extraction and relay.build(...).
    mod = mod.with_attr("executor", EXECUTOR)

    target = tvm.target.Target(target_hexagon, host=target_hexagon)

    if TUNE:
        if SAVE_TUNING_DATABSE:
            database = ms.relay_integration.tune_relay(
                mod=mod,
                target=target,
                params=params,
                work_dir=work_dir,
                # for faster tuning
                max_trials_global=20000,
                max_trials_per_task=8,
                num_trials_per_iter=8,
                strategy="replay-trace",
                # max_trials_global=20000,
                # num_trials_per_iter=32,
                # max_trials_per_task=128,
                # strategy="evolutionary",
                builder=get_hexagon_local_builder(),
                runner=get_hexagon_rpc_runner(hexagon_launcher, number=20),
                # Without this, the same workloads with different constant weights
                # are treated as distinct tuning tasks.
                module_equality="ignore-ndarray",
            )
            return ms.relay_integration.compile_relay(
                database=database,
                mod=mod,
                target=target,
                params=params,
            )
        else:
            with tempfile.TemporaryDirectory() as work_dir:
                database = ms.relay_integration.tune_relay(
                    mod=mod,
                    target=target,
                    params=params,
                    work_dir=work_dir,
                    # for faster tuning
                    max_trials_global=20000,
                    max_trials_per_task=8,
                    num_trials_per_iter=8,
                    strategy="replay-trace",
                    # max_trials_global=20000,
                    # num_trials_per_iter=32,
                    # max_trials_per_task=128,
                    # strategy="evolutionary",
                    builder=get_hexagon_local_builder(),
                    runner=get_hexagon_rpc_runner(hexagon_launcher, number=20),
                    # Without this, the same workloads with different constant weights
                    # are treated as distinct tuning tasks.
                    module_equality="ignore-ndarray",
                )
                return ms.relay_integration.compile_relay(
                    database=database,
                    mod=mod,
                    target=target,
                    params=params,
                )
    else:
        database = ms.database.JSONDatabase(
            "%s/database_workload.json" % work_dir, 
            "%s/database_tuning_record.json" % work_dir, 
            module_equality="ignore-ndarray")
        return ms.relay_integration.compile_relay(
            database=database,
            mod=mod,
            target=target,
            params=params,
        )



def get_onnx_input_shapes(onnx_model):
    initializer_names = [n.name for n in onnx_model.graph.initializer]
    input_shapes = {}
    input_dtypes = {}
    # The inputs contains both the inputs and parameters. We are just interested in the
    # inputs so skip all parameters listed in graph.initializer
    for input_info in onnx_model.graph.input:
        if input_info.name not in initializer_names:
            name, shape, dtype, _ = relay.frontend.onnx.get_info(input_info)
            for i in range(len(shape)):
                # update what are ostensibly batch dims to 1
                if not isinstance(shape[i], int):
                    shape[i] = 1
            input_shapes.update({input_info.name: shape})
            input_dtypes.update({input_info.name: dtype})

    return input_shapes, input_dtypes



def tvm_convert_to_fp16(mod, params, run_other_opts=True, fast_math=True):
    if run_other_opts:
        mod = tvm.relay.transform.FastMath()(mod) if fast_math else mod
        mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
        BindPass = tvm.relay.transform.function_pass(
            lambda fn, new_mod, ctx: tvm.relay.build_module.bind_params_by_name(fn, params),
            opt_level=1,
        )
        mod = BindPass(mod)
        mod = tvm.relay.transform.FoldConstant()(mod)
        mod = tvm.relay.transform.CombineParallelBatchMatmul()(mod)
        mod = tvm.relay.transform.FoldConstant()(mod)

    # Run fp16 pass
    mod = tvm.relay.transform.InferType()(mod)
    mod = tvm.relay.transform.ToMixedPrecision()(mod)

    if run_other_opts:
        # run one more pass to clean up new subgraph
        mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
        mod = tvm.relay.transform.FoldConstant()(mod)
        mod = tvm.relay.transform.CombineParallelBatchMatmul()(mod)
        mod = tvm.relay.transform.FoldConstant()(mod)
        mod = tvm.relay.transform.FastMath()(mod) if fast_math else mod
    return mod


@tvm.testing.requires_hexagon
def test_runtime(hexagon_launcher, model):

    print(model)
    work_dir = "work_" + model + "_fp16"
    
    model_json = "deeplab_fp16.json"
    model_params = "deeplab_fp16.params"
    
    # DATA_DIR = pathlib.Path(__file__).parent
    # onnx_file_path = DATA_DIR / f"{model}.onnx"
    
    # onnx_model = onnx.load(onnx_file_path)
    # input_shapes, input_dtypes = get_onnx_input_shapes(onnx_model)
    # mod, params = relay.frontend.from_onnx(
    #     onnx_model, shape=input_shapes, freeze_params=True, opset=13
    # )

    # mod = tvm_convert_to_fp16(mod, params)

    # mod = convert_conv2d_layout(mod, {"nn.conv2d": ["NHWC", "HWIO"]})

    inp = np.random.randn(1, 512, 512, 3).astype("float32")
    input_name = "ImageTensor:0"
    
    with open(model_json, "r") as file:
        mod = tvm.ir.load_json(file.read())

    with open(model_params, "rb") as file:
        params = relay.load_param_dict(file.read())
        
    
    hexagon_lowered = tune_ms(mod, params, hexagon_launcher, work_dir)
    print("tuning finished")

    with tvm.transform.PassContext(opt_level=3):
        llvm_lowered = tvm.relay.build(
            mod,
            tvm.target.Target(target_llvm, host=target_llvm),
            params=params,
        )

    with hexagon_launcher.create_session() as session:
        print("session created")

        graph_mod = session.get_executor_from_factory(hexagon_lowered)
        graph_mod.set_input(input_name, inp.copy())

        llvm_graph_mod = tvm.contrib.graph_executor.GraphModule(llvm_lowered["default"](tvm.cpu(0)))
        llvm_graph_mod.set_input(input_name, inp.copy())

        graph_mod.run()
        hexagon_output = graph_mod.get_output(0).numpy()

        llvm_graph_mod.run()
        ref_result = llvm_graph_mod.get_output(0).numpy()
        print(
            np.max(np.abs(ref_result - hexagon_output)),
            np.mean(np.abs(ref_result - hexagon_output)),
        )

        time_ms = graph_mod.benchmark(session.device, number=1, repeat=2).mean * 1e3

        print("time elapsed: ", time_ms)

        debug_ex = session.get_graph_debug_executor(
            hexagon_lowered.get_graph_json(), hexagon_lowered.lib
        )
        print(debug_ex.profile(input_name=inp.copy()))
        
        tvm.testing.assert_allclose(ref_result, hexagon_output, atol=2e-1)


if __name__ == "__main__":
    tvm.testing.main()
