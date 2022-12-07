"""
Run end to end model using meta schedual on fp16
"""
import tempfile
import os
import time

from re import T
import numpy as np
import onnx
import tvm
from tvm import relay
from tvm.script import tir as T

from tvm import meta_schedule as ms
from tvm.contrib.hexagon.meta_schedule import get_hexagon_local_builder, get_hexagon_rpc_runner
from tvm.relay.backend import Executor

from tvm.meta_schedule.cost_model.xgb_model import XGBModel

TUNE = True
SAVE_TUNING_DATABSE = True
EXECUTOR = relay.backend.Executor("graph", {"link-params": True})


target_hexagon = tvm.target.hexagon("v69")
target_llvm = tvm.target.Target("llvm")



def tune_ms(mod, params, hexagon_launcher, work_dir, use_relax):
    # This line is necessary for link-params to take effect during
    # task extraction and relay.build(...).
    mod = mod.with_attr("executor", EXECUTOR)

    target = tvm.target.Target(target_hexagon, host=target_hexagon)
    ms_tune = ms.relay_integration.tune_relay
    ms_compile = ms.relay_integration.compile_relay

    if use_relax:
        ms_tune = ms.relax_integration.tune_relax
        ms_compile = ms.relax_integration.compile_relax        

    if TUNE:
        if SAVE_TUNING_DATABSE:
            database = ms_tune(
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
                # Enable after rebasing relax repo
                # module_equality="ignore-ndarray",
            )
            return ms_compile(
                database=database,
                mod=mod,
                target=target,
                params=params,
            )
        else:
            with tempfile.TemporaryDirectory() as work_dir:
                database = ms_tune(
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
                    # Enable after rebasing relax repo
                    # module_equality="ignore-ndarray",
                )
                return ms_compile(
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
        return ms_compile(
            database=database,
            mod=mod,
            target=target,
            params=params,
        )

@tvm.testing.requires_hexagon
def test_runtime(hexagon_launcher):

    model_json = "workload_1.json"
    with open(model_json, "r") as file:
        mod = tvm.ir.load_json(file.read())

    inp = np.random.randn(1, 299, 299, 3).astype("float32")
    input_name = "Mul:0"
    params= {}
    use_relax = False
    work_dir = "work_bug_inception_int8"
    import pdb;
    pdb.set_trace()
    hexagon_lowered = tune_ms(mod, params, hexagon_launcher, work_dir, use_relax)
    print("tuning finished")

    if use_relax:
        mod = relay_translator.from_relay(mod_relay["main"], "llvm", params)
        ex = relax.vm.build(mod, "llvm")
        dev = tvm.cpu()
        vm_rt = relax.VirtualMachine(ex, dev)
        data = tvm.nd.array(inp, dev)
        ref_result = vm_rt["main"](data).numpy()   
    else: 
        with tvm.transform.PassContext(opt_level=3):
            llvm_lowered = tvm.relay.build(
                mod,
                tvm.target.Target(target_llvm, host=target_llvm),
                params=params,
            )   
        llvm_graph_mod = tvm.contrib.graph_executor.GraphModule(llvm_lowered["default"](tvm.cpu(0)))
        llvm_graph_mod.set_input("main", inp.copy())
        llvm_graph_mod.run()
        ref_result = llvm_graph_mod.get_output(0).numpy()


    with hexagon_launcher.create_session() as session:
        print("session created")
        
        if use_relax:
            dev = session.device
            vm_mod = session.get_executor_from_factory(hexagon_lowered)
            vm_rt = relax.VirtualMachine(vm_mod, dev)
            data = tvm.nd.array(inp, dev)
            vm_rt.set_input("main", data)
            vm_rt.invoke_stateful("main")
            hexagon_output = vm_rt.get_outputs("main").numpy()
            
            t_mean = 0
            NUM_OF_RUN = 3
            for i in range(NUM_OF_RUN):
                t0 = time.time()
                vm_rt.invoke_stateful("main")
                t_mean += time.time() - t0
            
            print("run finished in (s) ", t_mean/NUM_OF_RUN)
            
        else:
            graph_mod = session.get_executor_from_factory(hexagon_lowered)
            graph_mod.set_input(input_name, inp.copy())
            graph_mod.run()
            hexagon_output = graph_mod.get_output(0).numpy()
            time_ms = graph_mod.benchmark(session.device, number=1, repeat=2).mean * 1e3
            print("time elapsed: ", time_ms)
            debug_ex = session.get_graph_debug_executor(
            hexagon_lowered.get_graph_json(), hexagon_lowered.lib
            )
            print(debug_ex.profile(input_name=inp.copy()))    
            
            
        print(
            np.max(np.abs(ref_result - hexagon_output)),
            np.mean(np.abs(ref_result - hexagon_output)),
        )

        tvm.testing.assert_allclose(hexagon_output, ref_result, atol=2e-1)



if __name__ == "__main__":
    tvm.testing.main()