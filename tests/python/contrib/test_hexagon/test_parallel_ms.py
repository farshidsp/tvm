import tempfile

import numpy as np

import tvm.testing
from tvm import relay

from tvm import meta_schedule as ms
from tvm.contrib.hexagon.meta_schedule import get_hexagon_local_builder, get_hexagon_rpc_runner
from tvm.relay.backend import Executor


@tvm.testing.requires_hexagon
def test_conv2d_conv2d(hexagon_launcher):
    I, O, H, W = 64, 64, 56, 56
    kH = kW = 3

    strides = (1, 1)
    padding = (1, 1)

    data_shape = (1, H, W, I)
    w1_shape = (kH, kW, I, O)
    w2_shape = (kH, kW, I, O * 2)

    data = relay.var("data", shape=data_shape, dtype="float16")
    weight1 = relay.var("weight1", shape=w1_shape, dtype="float16")
    weight2 = relay.var("weight2", shape=w2_shape, dtype="float16")

    conv1 = relay.nn.conv2d(
        data=data,
        weight=weight1,
        kernel_size=(kH, kW),
        channels=O,
        padding=padding,
        strides=strides,
        out_dtype="float16",
        data_layout="NHWC",
        kernel_layout="HWIO",
    )

    conv2 = relay.nn.conv2d(
        data=conv1,
        weight=weight2,
        kernel_size=(kH, kW),
        channels=w2_shape[-1],
        padding=padding,
        strides=strides,
        out_dtype="float16",
        data_layout="NHWC",
        kernel_layout="HWIO",
    )

    # Single conv2d
    # mod = tvm.IRModule.from_expr(conv1)

    # Two conv2d
    mod = tvm.IRModule.from_expr(conv2)

    data_np = np.random.uniform(0, 255, size=data_shape).astype("float16")
    weight1_np = np.random.uniform(0, 255, size=w1_shape).astype("float16")
    weight2_np = np.random.uniform(0, 255, size=w2_shape).astype("float16")

    params = {"weight1": weight1_np, "weight2": weight2_np}

    target_hexagon = tvm.target.hexagon("v69")
    target = tvm.target.Target(target_hexagon, host=target_hexagon)

    config = ms.TuneConfig(
        strategy="replay_trace",
        num_trials_per_iter=8,
        max_trials_per_task=8,
        max_trials_global=16,
    )

    executor = Executor("graph", {"link-params": True})

    with tempfile.TemporaryDirectory() as work_dir:
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
