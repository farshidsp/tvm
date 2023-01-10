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

from models import export_onnx
from models.run_model import get_onnx_input_shapes, tvm_convert_to_fp16, onnx_to_relay
from tvm.meta_schedule.cost_model.xgb_model import XGBModel

# from tvm.script import relax as R
from tvm.script import tir as T

TUNE = True
SAVE_TUNING_DATABSE = True
EXECUTOR = relay.backend.Executor("graph", {"link-params": True})

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



@tvm.script.ir_module
class TuningBug:
    @T.prim_func
    def main(p0: T.Buffer[(T.int64(1), T.int64(256), T.int64(256), T.int64(32)), "float16"], T_add: T.Buffer[(T.int64(1), T.int64(256), T.int64(256), T.int64(16)), "float16"]):
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        pad_temp = T.alloc_buffer([T.int64(1), T.int64(256), T.int64(256), T.int64(32)], dtype="float16")
        conv2d_nhwc = T.alloc_buffer([T.int64(1), T.int64(256), T.int64(256), T.int64(16)], dtype="float16")
        fused_nn_conv2d_constant_55 = T.allocate_const([19377, 19121, 13468, -13142, 14843, 20768, 18466, 18487, -14401, 19536, -14757, -18081, -13490, -15044, -13144, -14595], "float16", [1, 1, 1, 16])
        fused_nn_conv2d_constant_55_1 = T.buffer_decl([1, 1, 1, 16], dtype="float16", data=fused_nn_conv2d_constant_55)
        fused_constant_55 = T.allocate_const([13532, 13655, 14718, 13088, -19251, -18856, -17132, 15435, 14992, 14609, 16079, 14677, -16357, -20516, 17388, 15589, -32768, 0, 0, 0, -32768, -32768, -32768, 0, 0, 0, -32768, 0, -32768, -32768, 0, 0, -32768, 0, -32768, 0, -32768, 0, -32768, -32768, 0, 0, -32768, 0, -32768, -32768, 0, 0, 14372, -16251, 14622, 14364, 12315, -15858, 16748, 15399, -16643, -17678, 13425, -20105, -20610, -20066, 12278, 15356, 12937, 15058, -17406, 12462, 11095, -16483, -16380, 15333, -17319, -23271, -16496, -19105, 15983, -18912, 15491, 12918, 5983, 976, 4413, -29255, 5097, 6384, -25680, 5054, 5960, 371, 3303, 4414, 5701, 7321, -25742, 5347, 13348, 16501, -17265, -22489, -17087, -16933, -18313, 15503, -20626, -17883, -17975, -16913, 11898, 12818, -19515, 12171, 14967, -17988, -16890, -22858, 16469, -20284, -17033, -17372, 14524, 10037, 14071, -18653, -18666, 16507, -17937, 16996, 14453, -18544, -19925, 8590, -15260, -18918, -18551, -17002, -19846, -21132, -21993, 13664, -18075, -17361, -20563, -16895, -18780, -15863, -20109, 13796, 14055, -17847, -16855, -18137, 12201, -20001, -17626, 15427, -17255, 15429, -23568, -16359, -20057, 14864, 11401, -19524, -17956, -18090, 11346, 14343, 15111, -19243, 14410, 15332, -18803, 14415, 16226, 15107, -17931, 15599, -19354, -19041, -19395, -17191, 14208, -17273, 15289, -17257, -17141, 15964, 14906, 15670, 15697, 13608, -20128, 10137, -19335, 16410, 14039, 15121, -18486, 14319, 13464, -16118, 13746, -20747, -18925, -17479, -21478, -21483, 15064, 14308, -17494, 15149, -16951, -19272, 12252, 14594, -17997, -17590, 16476, -18139, 12239, 16978, 13594, -16756, -32768, -32768, -32768, -32768, 0, -32768, -32768, 0, 0, 0, 0, -32768, 0, -32768, 0, 0, 5975, 750, 4390, -29279, 5101, 6386, -25682, 5039, 5947, 313, 3273, 4392, 5694, 7318, -25739, 5351, -17726, 14517, 16539, 13455, 15959, -17269, -17873, 9883, -18444, -21159, -21816, 13032, -18435, 15390, -22038, -16294, 0, 0, -32768, -32768, -32768, -32768, -32768, -32768, 0, -32768, 0, -32768, 0, -32768, -32768, -32768, 10591, 11486, -20198, 10714, 12372, -19331, -19259, -21462, -19737, -21028, -18346, -21340, 10312, 10399, -19814, -21042, -20135, -18970, 16467, -21080, -18199, -17193, -18354, 13403, 16714, -19676, 13338, -18134, 12630, -20772, -17396, 15780, -15741, -17587, -17549, -16852, -17624, 14167, 11420, 15595, -21468, -18401, 15365, -17769, 14563, -18340, 13397, 13469, -32768, 0, 0, -32768, -32768, -32768, -32768, -32768, 0, 0, 0, -32768, 0, -32768, -32768, 0, 5944, 495, 4291, -29286, 5124, 6386, -25691, 5070, 5925, 200, 3313, 4342, 5678, 7305, -25744, 5360, -17880, 14982, -17337, -19999, 17602, -19329, -19687, -20596, -18112, 11618, 14964, -19370, -18780, 15678, -23440, 15379, -18151, -21239, -20074, -24124, -19247, 13631, 15809, 10861, 14906, -18702, -15710, -18092, -18015, 16203, 14810, -19789, 14576, -16782, 13649, -19519, 12616, -19069, -16534, 14254, -18072, 13414, -18238, -18902, 15878, -18705, 15110, -18100, 14161, 14895, 15810, -21990, 12427, -17845, 13509, -16073, -17964, -17527, 15591, -16672, 16006, 13677, 16088, -23467, -21988, -21778, -18664, -19128, 13825, -18392, -17262, -17938, 14491, -18104, 14774, 16808, 17297, -18980, -19107, -21888, 11544, -17732, -16607, 13529, 15563, -15956, 14218, -20212, 17725, 12300, 15277, -16716, 14602, -16409, 13727, -15886, 11534, -18208, 13971, 10851, 13072, 14960, 12351, -17703, -20826, -21456, -18846, -18097, 14726, -18916, 15833, 12625, -16129, 14371, -18503, 17540, -16266, -16184, 15701, -17320, -17678, 16697, -19272, -22827, 15152, -18003, 13987, 17156, -18920, -22413, 13432, -18283, 13680, -17162, 13841, -17940, -17338, -19324, -17932, 15158, -17130, -16205, -18320, 14694], "float16", [1, 1, 32, 16])
        fused_constant_55_1 = T.buffer_decl([1, 1, 32, 16], dtype="float16", data=fused_constant_55)
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(256), T.int64(256), T.int64(32)):
            with T.block("pad_temp"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(p0[v_i0, v_i1, v_i2, v_i3])
                T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                pad_temp[v_i0, v_i1, v_i2, v_i3] = p0[v_i0, v_i1, v_i2, v_i3]
        for nn, yy, xx, ff, ry, rx, rc in T.grid(T.int64(1), T.int64(256), T.int64(256), T.int64(16), T.int64(1), T.int64(1), T.int64(32)):
            with T.block("conv2d_nhwc"):
                v_nn, v_yy, v_xx, v_ff, v_ry, v_rx, v_rc = T.axis.remap("SSSSRRR", [nn, yy, xx, ff, ry, rx, rc])
                T.reads(pad_temp[v_nn, v_yy + v_ry, v_xx + v_rx, v_rc], fused_constant_55_1[v_ry, v_rx, v_rc, v_ff])
                T.writes(conv2d_nhwc[v_nn, v_yy, v_xx, v_ff])
                T.block_attr({"layout_free_placeholders":[fused_constant_55_1]})
                with T.init():
                    conv2d_nhwc[v_nn, v_yy, v_xx, v_ff] = T.float16(0)
                conv2d_nhwc[v_nn, v_yy, v_xx, v_ff] = conv2d_nhwc[v_nn, v_yy, v_xx, v_ff] + pad_temp[v_nn, v_yy + v_ry, v_xx + v_rx, v_rc] * fused_constant_55_1[v_ry, v_rx, v_rc, v_ff]
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(256), T.int64(256), T.int64(16)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(conv2d_nhwc[v_ax0, v_ax1, v_ax2, v_ax3], fused_nn_conv2d_constant_55_1[v_ax0, T.int64(0), T.int64(0), v_ax3])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = conv2d_nhwc[v_ax0, v_ax1, v_ax2, v_ax3] + fused_nn_conv2d_constant_55_1[v_ax0, T.int64(0), T.int64(0), v_ax3]
    


@tvm.testing.requires_hexagon
def test_bug(hexagon_launcher):

    work_dir = "work_bug_fp16"
    mod = TuningBug
    params = {
        "p0": np.random.rand(1, 256, 256, 32).astype(np.float16),
    }

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
