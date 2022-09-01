# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import numpy as np

import tvm.testing
from tvm import relay
from tvm import meta_schedule as ms
from tvm.contrib.hexagon.meta_schedule import get_hexagon_local_builder, get_hexagon_rpc_runner
from tvm.relay.backend import Executor
from tvm.meta_schedule.tune import tune_extracted_tasks
from tvm.meta_schedule.relay_integration import extract_task_from_relay


def get_conv2d_nchw(
    d_shape,
    w_shape,
    padding,
    strides=(1, 1),
):
    out_dtype = "float16"

    data = relay.var("data", shape=d_shape, dtype="float16")
    weight = relay.var("weight", shape=w_shape, dtype="float16")
    out_channel = w_shape[0]
    return relay.nn.conv2d(
        data=data,
        weight=weight,
        kernel_size=w_shape[2:],
        channels=out_channel,
        padding=padding,
        strides=strides,
        out_dtype=out_dtype,
    )

def convert_conv2d_layout(mod, desired_layouts):
    with tvm.transform.PassContext(opt_level=3):
        seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
        return seq(mod)


@tvm.testing.requires_hexagon
def test_conv2d_f16f16f16(hexagon_launcher):
    target_hexagon = tvm.target.hexagon("v69", link_params=True)
    target = tvm.target.Target(target_hexagon, host=target_hexagon)
    I = 128
    O = 128
    H = 28
    W = 28
    kH = 1
    kW = 1
    padding = (0, 0)
    strides = (2, 2)

    data_shape = (1, I, H, W)
    weight_shape = (O, I, kH, kW)
    bias_shape = (weight_shape[0],)

    bias = relay.var("bias", shape=bias_shape, dtype="float16")

    conv2d = get_conv2d_nchw(data_shape, weight_shape, padding, strides=strides)
    bias_add = relay.nn.bias_add(conv2d, bias)

    use_bias = True

    if use_bias:
        out = bias_add
    else:
        out = conv2d

    mod = tvm.IRModule.from_expr(out)

    data_np = np.random.randn(*data_shape).astype("float16")
    weight_np = np.random.randn(*weight_shape).astype("float16")
    bias_np = np.random.randn(weight_shape[0]).astype("float16")
    params = {"weight": weight_np, "bias": bias_np}

    out_ty = relay.transform.InferType()(mod)

    _, _, P, Q = out_ty["main"].body.checked_type.shape
    mod = convert_conv2d_layout(mod, {"nn.conv2d": ["NHWC", "HWIO"]})

    target_llvm = tvm.target.Target("llvm")

    with tvm.transform.PassContext(
        opt_level=3,
    ):
        lib_ref = relay.build(mod, target=target_llvm, params=params)

    rt_mod_ref = tvm.contrib.graph_executor.GraphModule(lib_ref["default"](tvm.cpu(0)))

    rt_mod_ref.set_input("data", data_np)

    rt_mod_ref.run()

    ref = rt_mod_ref.get_output(0).numpy()

    work_dir = "work2"
    config = ms.TuneConfig(
        # strategy="replay_trace",
        strategy="evolutionary",
        num_trials_per_iter=8,
        max_trials_per_task=8,
        max_trials_global=8,
    )

    executor = Executor("graph", {"link-params": True})

    lib = ms.tune_relay(
        mod=mod,
        params=params,
        target=target,
        config=config,
        work_dir=work_dir,
        builder=get_hexagon_local_builder(),
        runner=get_hexagon_rpc_runner(hexagon_launcher, number=20),
        executor=executor,
    )

    print("tuning finished")

    with hexagon_launcher.start_session() as session:
        print("session created")
        rt_mod = session.get_executor_from_factory(lib)

        rt_mod.set_input("data", data_np)

        rt_mod.run()

        out = rt_mod.get_output(0).numpy()

        print(np.max(np.abs(ref - out)), np.mean(np.abs(ref - out)))


@tvm.testing.requires_hexagon
def test_resnet50(hexagon_launcher):
    target_hexagon = tvm.target.hexagon("v69")
    target = tvm.target.Target(target_hexagon, host=target_hexagon)
    target_llvm = tvm.target.Target("llvm")

    with open("resnet50_fp16.json", "r") as fi:
        mod = tvm.ir.load_json(fi.read())

    with open("resnet50_fp16.params", "rb") as fi:
        params = relay.load_param_dict(fi.read())

    # with tvm.transform.PassContext(opt_level=3):
    #     opt_mod, _ = relay.optimize(mod, target=target, params=params)
    #     print(opt_mod)
    # return

    mod = convert_conv2d_layout(mod, {"nn.conv2d": ["NHWC", "HWIO"]})

    inp = np.random.randn(1, 3, 224, 224).astype("float32")
    input_name = "image"

    work_dir = "work"
    config = ms.TuneConfig(
        # strategy="replay_trace",
        strategy="evolutionary",
        num_trials_per_iter=32,
        max_trials_per_task=64,
        max_trials_global=50000,
    )

    executor = Executor("graph", {"link-params": True})

    pass_config = {"relay.FuseOps.link_params": True,
                   "relay.backend.use_meta_schedule": True,
                   "relay.backend.tir_converter": "default"
                   }

    extracted_tasks = extract_task_from_relay(mod, target, params, pass_config=pass_config)

    tune_tasks = []

    for task in extracted_tasks:
        if not "dense" in task.task_name:
            tune_tasks.append(task)

    work_dir = "work"

    database = tune_extracted_tasks(
        tune_tasks,
        config,
        work_dir,
        builder=get_hexagon_local_builder(),
        runner=get_hexagon_rpc_runner(hexagon_launcher, number=20),
    )

    with target, database:
        with tvm.transform.PassContext(
            opt_level=3,
            config={
                "relay.backend.use_meta_schedule": True,
                "relay.backend.use_meta_schedule_dispatch": target.kind.name != "cuda",
                "relay.backend.tir_converter": "default",
            },
        ):
            hexagon_lowered = relay.build(mod, target=target, params=params, executor=executor)

    # hexagon_lowered = ms.tune_relay(
    #     mod=mod,
    #     params=params,
    #     target=target,
    #     config=config,
    #     work_dir=work_dir,
    #     builder=get_hexagon_local_builder(),
    #     runner=get_hexagon_rpc_runner(hexagon_launcher, number=20),
    #     executor=executor,
    # )

    print("tuning finished")

    with tvm.transform.PassContext(opt_level=3):
        llvm_lowered = tvm.relay.build(
            mod,
            tvm.target.Target(target_llvm, host=target_llvm),
            params=params,
        )

    with hexagon_launcher.start_session() as session:
        print("session created")

        graph_mod = session.get_executor_from_factory(hexagon_lowered)
        graph_mod.set_input(input_name, inp.copy())

        llvm_graph_mod = tvm.contrib.graph_executor.GraphModule(llvm_lowered["default"](tvm.cpu(0)))
        llvm_graph_mod.set_input(input_name, inp.copy())

        import time

        graph_mod.run()
        hexagon_output = graph_mod.get_output(0).numpy()

        llvm_graph_mod.run()
        ref_result = llvm_graph_mod.get_output(0).numpy()
        print(np.max(np.abs(ref_result - hexagon_output)), np.mean(np.abs(ref_result - hexagon_output)))

        time_ms = graph_mod.benchmark(session.device, number=1, repeat=20).mean * 1e3

        print("time elapsed: ", time_ms)

        debug_ex = session.get_graph_debug_executor(hexagon_lowered.get_graph_json(), hexagon_lowered.lib)
        print(debug_ex.profile(input_name=inp.copy()))



@tvm.testing.requires_hexagon
def test_dense(hexagon_launcher):
    mod = tvm.parser.fromtext(
        """
#[version = "0.0.5"]
  def @main(%p0: Tensor[(1, 2048), float16] /* ty=Tensor[(1, 2048), float16] */, %p1: Tensor[(1000, 2048), float16] /* ty=Tensor[(1000, 2048), float16] */, %p2: Tensor[(1, 1000), float16] /* ty=Tensor[(1, 1000), float16] */, hash="0ff840526f3d7027") -> Tensor[(1, 1000), float16] {
    %0 = nn.dense(%p0, %p1, units=None, out_dtype="float16") /* ty=Tensor[(1, 1000), float16] */;
    add(%0, %p2) /* ty=Tensor[(1, 1000), float16] */
  }
""")

    params = {"p1": np.random.randn(1000, 2048).astype("float16"),
              "p2": np.random.randn(1, 1000).astype("float16")}
    target_hexagon = tvm.target.hexagon("v69")
    target = tvm.target.Target(target_hexagon, host=target_hexagon)

    executor = Executor("graph", {"link-params": True})
    config = ms.TuneConfig(
        # strategy="replay_trace",
        strategy="evolutionary",
        num_trials_per_iter=8,
        max_trials_per_task=8,
        max_trials_global=8,
    )

    pass_config = {"relay.FuseOps.link_params": True,
                   "relay.backend.use_meta_schedule": True,
                   "relay.backend.tir_converter": "default"
                   }

    extracted_tasks = extract_task_from_relay(mod, target, params, pass_config=pass_config)

    tune_tasks = []

    for task in extracted_tasks:
        if not "dense" in task.task_name:
            tune_tasks.append(task)

    work_dir = "work"

    database = tune_extracted_tasks(
        tune_tasks,
        config,
        work_dir,
        builder=get_hexagon_local_builder(),
        runner=get_hexagon_rpc_runner(hexagon_launcher, number=20),
    )

    with target, database:
        with tvm.transform.PassContext(
            opt_level=3,
            config={
                "relay.backend.use_meta_schedule": True,
                "relay.backend.use_meta_schedule_dispatch": target.kind.name != "cuda",
                "relay.backend.tir_converter": "default",
            },
        ):
            hexagon_lowered = relay.build(mod, target=target, params=params, executor=executor)

    # print(hexagon_lowered.lib.get_source("asm"))
    with hexagon_launcher.start_session() as session:
        graph_mod = session.get_executor_from_factory(hexagon_lowered)
        graph_mod.run()
        time_ms = graph_mod.benchmark(session.device, number=1, repeat=20).mean * 1e3

        print("time elapsed: ", time_ms)

        return


@tvm.testing.requires_hexagon
def test_pool(hexagon_session):
    mod = tvm.parser.fromtext(
        """
#[version = "0.0.5"]
  def @main(%p051: Tensor[(1, 1, 112, 112, 64), float16] /* ty=Tensor[(1, 1, 112, 112, 64), float16] */, hash="6f35ef50ce92dd21", layout="NCHW64c", out_layout="") -> Tensor[(1, 1, 56, 56, 64), float16] {
    nn.max_pool2d(%p051, pool_size=[3, 3], strides=[2, 2], padding=[1, 1, 1, 1], layout="NCHW64c") /* ty=Tensor[(1, 1, 56, 56, 64), float16] */
  }
""")

    params = {}
    target_hexagon = tvm.target.hexagon("v69", link_params=True)

    with tvm.transform.PassContext(opt_level=3):
        hexagon_lowered = relay.build(
            mod,
            tvm.target.Target(target_hexagon, host=target_hexagon),
            params=params,
        )

    # print(hexagon_lowered.lib.get_source("asm"))
    graph_mod = hexagon_session.get_executor_from_factory(hexagon_lowered)
    graph_mod.run()
    time_ms = graph_mod.benchmark(hexagon_session.device, number=1, repeat=20).mean * 1e3

    print("time elapsed: ", time_ms)
