"""
Run end to end model using meta schedual on fp16
"""
import tempfile

from re import T
import numpy as np
import onnx
import tvm
from tvm import relay
from tvm import meta_schedule as ms
from tvm.contrib.hexagon.meta_schedule import get_hexagon_local_builder, get_hexagon_rpc_runner
from tvm.relay.backend import Executor
from tvm.meta_schedule.cost_model.xgb_model import XGBModel

# from tvm.script import relax as R
from tvm.script import tir as T

TUNE = True
SAVE_TUNING_DATABSE = True
EXECUTOR = relay.backend.Executor("graph", {"link-params": True})

target_hexagon = tvm.target.hexagon("v69")
target_llvm = tvm.target.Target("llvm -num-cores 64")

def convert_conv2d_layout(mod, desired_layouts):
    with tvm.transform.PassContext(opt_level=3):
        seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
        return seq(mod)


def tune_ms(mod, params, work_dir):
    # This line is necessary for link-params to take effect during
    # task extraction and relay.build(...).
    mod = mod.with_attr("executor", EXECUTOR)

    target = tvm.target.Target(target_llvm, host=target_llvm)

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
                # builder=get_hexagon_local_builder(),
                # runner=get_hexagon_rpc_runner(hexagon_launcher, number=20),
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
                database = ms.tir_integration.tune_relay(
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



# @tvm.script.ir_module
# class TuningBug:
#     @T.prim_func
#     def main(p0: T.Buffer[(T.int64(1), T.int64(128), T.int64(128), T.int64(144)), "float16"], T_relu: T.Buffer[(T.int64(1), T.int64(128), T.int64(128), T.int64(48)), "float16"]):
#         # function attr dict
#         T.func_attr({"global_symbol": "main", "tir.noalias": True})
#         # body
#         # with T.block("root")
#         pad_temp = T.alloc_buffer([T.int64(1), T.int64(128), T.int64(128), T.int64(144)], dtype="float16")
#         conv2d_nhwc = T.alloc_buffer([T.int64(1), T.int64(128), T.int64(128), T.int64(48)], dtype="float16")
#         T_add = T.alloc_buffer([T.int64(1), T.int64(128), T.int64(128), T.int64(48)], dtype="float16")
#         fused_nn_conv2d_constant_59 = T.allocate_const([8213, 7814, 9066, 8865, 10041, 7281, 10568, 10615, 10275, 8641, 9530, 7235, 8146, 11332, 10645, 6152, 7654, 9328, 8489, 9940, -26592...], "float16", [1, 1, 1, 48])
#         fused_nn_conv2d_constant_59_1 = T.buffer_decl([1, 1, 1, 48], dtype="float16", data=fused_nn_conv2d_constant_59)
#         fused_constant_59 = T.allocate_const([-22943, 11693, -20536, -21322, 11299, 4133, 12534, 10989, -23940, -20371, 11279, -32768, -21315, -21109, 11120, 6541, 9895, -21323, -22356, 9442, -26598...], "float16", [1, 1, 144, 48])
#         fused_constant_59_1 = T.buffer_decl([1, 1, 144, 48], dtype="float16", data=fused_constant_59)
#         for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(128), T.int64(128), T.int64(144)):
#             with T.block("pad_temp"):
#                 v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
#                 T.reads(p0[v_i0, v_i1, v_i2, v_i3])
#                 T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
#                 pad_temp[v_i0, v_i1, v_i2, v_i3] = p0[v_i0, v_i1, v_i2, v_i3]
#         for nn, yy, xx, ff, ry, rx, rc in T.grid(T.int64(1), T.int64(128), T.int64(128), T.int64(48), T.int64(1), T.int64(1), T.int64(144)):
#             with T.block("conv2d_nhwc"):
#                 v_nn, v_yy, v_xx, v_ff, v_ry, v_rx, v_rc = T.axis.remap("SSSSRRR", [nn, yy, xx, ff, ry, rx, rc])
#                 T.reads(pad_temp[v_nn, v_yy + v_ry, v_xx + v_rx, v_rc], fused_constant_59_1[v_ry, v_rx, v_rc, v_ff])
#                 T.writes(conv2d_nhwc[v_nn, v_yy, v_xx, v_ff])
#                 T.block_attr({"layout_free_placeholders":[fused_constant_59_1]})
#                 with T.init():
#                     conv2d_nhwc[v_nn, v_yy, v_xx, v_ff] = T.float16(0)
#                 conv2d_nhwc[v_nn, v_yy, v_xx, v_ff] = conv2d_nhwc[v_nn, v_yy, v_xx, v_ff] + pad_temp[v_nn, v_yy + v_ry, v_xx + v_rx, v_rc] * fused_constant_59_1[v_ry, v_rx, v_rc, v_ff]
#         for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(128), T.int64(128), T.int64(48)):
#             with T.block("T_add"):
#                 v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
#                 T.reads(conv2d_nhwc[v_ax0, v_ax1, v_ax2, v_ax3], fused_nn_conv2d_constant_59_1[v_ax0, T.int64(0), T.int64(0), v_ax3])
#                 T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
#                 T_add[v_ax0, v_ax1, v_ax2, v_ax3] = conv2d_nhwc[v_ax0, v_ax1, v_ax2, v_ax3] + fused_nn_conv2d_constant_59_1[v_ax0, T.int64(0), T.int64(0), v_ax3]
#         for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(128), T.int64(128), T.int64(48)):
#             with T.block("T_relu"):
#                 v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
#                 T.reads(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
#                 T.writes(T_relu[v_ax0, v_ax1, v_ax2, v_ax3])
#                 T_relu[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(T_add[v_ax0, v_ax1, v_ax2, v_ax3], T.float16(0))


def test_deeplab():
    work_dir = "work_deeplab_fp16_x86"

    model_json = "deeplab_fp16.json"
    model_params = "deeplab_fp16.params" 

    inp = np.random.randn(1, 512, 512, 3).astype("float32")
    input_name = "ImageTensor:0"

    with open(model_json, "r") as file:
        mod = tvm.ir.load_json(file.read())

    with open(model_params, "rb") as file:
        params = relay.load_param_dict(file.read())

    lowered = tune_ms(mod, params, work_dir)


# @tvm.testing.requires_hexagon
# def test_bug():

#     work_dir = "work_bug_fp16"
#     mod = TuningBug
#     params = {
#         "p0": np.random.rand(1, 256, 256, 32).astype(np.float16),
#     }

#     hexagon_lowered = tune_ms(mod, params, work_dir)
#     print("tuning finished")

#     with tvm.transform.PassContext(opt_level=3):
#         llvm_lowered = tvm.relay.build(
#             mod,
#             tvm.target.Target(target_llvm, host=target_llvm),
#             params=params,
#         )

#     with hexagon_launcher.create_session() as session:
#         print("session created")

#         graph_mod = session.get_executor_from_factory(hexagon_lowered)
#         graph_mod.set_input(input_name, inp.copy())

#         llvm_graph_mod = tvm.contrib.graph_executor.GraphModule(llvm_lowered["default"](tvm.cpu(0)))
#         llvm_graph_mod.set_input(input_name, inp.copy())

#         graph_mod.run()
#         hexagon_output = graph_mod.get_output(0).numpy()

#         llvm_graph_mod.run()
#         ref_result = llvm_graph_mod.get_output(0).numpy()
#         print(
#             np.max(np.abs(ref_result - hexagon_output)),
#             np.mean(np.abs(ref_result - hexagon_output)),
#         )

#         time_ms = graph_mod.benchmark(session.device, number=1, repeat=2).mean * 1e3

#         print("time elapsed: ", time_ms)

#         debug_ex = session.get_graph_debug_executor(
#             hexagon_lowered.get_graph_json(), hexagon_lowered.lib
#         )
#         print(debug_ex.profile(input_name=inp.copy()))

#         tvm.testing.assert_allclose(ref_result, hexagon_output, atol=2e-1)


if __name__ == "__main__":
    tvm.testing.main()