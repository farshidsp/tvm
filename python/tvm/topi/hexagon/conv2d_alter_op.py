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
# pylint: disable=invalid-name,unused-variable,unused-argument,no-member
"""Dense alter op functions for ARM"""

import tvm
from tvm import te
from tvm import relay
from tvm import autotvm
from ..utils import get_const_tuple
from .. import nn
from ..nn.utils import get_pad_tuple
from ..nn import conv2d_legalize, conv2d_alter_layout


@conv2d_alter_layout.register("hexagon")
def _alter_conv2d_layout(attrs, inputs, tinfos, out_type):
    target = tvm.target.Target.current(allow_none=False)
    dispatch_ctx = autotvm.task.DispatchContext.current
    new_attrs = {k: attrs[k] for k in attrs.keys()}

    # Parse the attributes.
    padding = attrs.get_int_tuple("padding")
    strides = attrs.get_int_tuple("strides")
    dilation = attrs.get_int_tuple("dilation")
    data_layout = attrs["data_layout"]
    kernel_layout = attrs["kernel_layout"]
    data_tensor, kernel_tensor = tinfos
    data_dtype = data_tensor.dtype
    kernel_dtype = kernel_tensor.dtype
    out_dtype = out_type.dtype

    impl, outs = relay.backend.te_compiler.select_implementation(
        relay.op.get("nn.conv2d"), attrs, tinfos, out_type, target
    )
    if impl.name.find("winograd") != -1:
        if dilation != (1, 1):
            logger.warning("Does not support weight pre-transform for dilated convolution.")
            return None

        assert data_layout == "NHWC" and kernel_layout == "HWIO"
        N, H, W, CI = get_const_tuple(data_tensor.shape)
        KH, KW, _, CO = get_const_tuple(kernel_tensor.shape)

        # Pre-compute weight transformation in winograd
        tile_size = 4
        # HWIO -> OIHW
        kernel_transform = relay.transpose(inputs[1], axes=[3, 2, 0, 1])
        # alpha, alpha, CO, CI
        weight = relay.nn.contrib_conv2d_winograd_weight_transform(
            kernel_transform, tile_size=tile_size
        )
        new_attrs["tile_size"] = tile_size
        new_attrs["channels"] = CO
        return relay.nn.contrib_conv2d_winograd_without_weight_transform(
            inputs[0], weight, **new_attrs
        )
    return None


@nn.conv2d_legalize.register("hexagon")
def _conv2d_legalize(attrs, inputs, arg_types):
    data_layout = attrs["data_layout"]
    kernel_layout = attrs["kernel_layout"]

    output_tensor = arg_types[2]

    # Collect the input exprs.
    data, kernel = inputs

    if data_layout != "NHWC" or kernel_layout != "HWIO":
        return None

    # Collect the input tensors.
    data_tensor, kernel_tensor = arg_types[0], arg_types[1]
    out_channel = kernel_tensor.shape[0]

    # Dilation not supported yet. Return None if dilation is not (1, 1)
    dilation = attrs.get_int_tuple("dilation")
    if not (dilation[0] == 1 and dilation[1] == 1):
        return None

    # No legalization for depthwise convolutions yet.
    groups = attrs.get_int("groups")
    if groups != 1:
        return None

    # Get the conv attrs
    new_attrs = {k: attrs[k] for k in attrs.keys()}

    padding = attrs.get_int_tuple("padding")
    kh, kw = attrs.get_int_tuple("kernel_size")
    pt, pl, pb, pr = get_pad_tuple(padding, (kh, kw))

    # TODO: pad on input channel?
    in_channel_vector_length = 1
    in_channel = data_tensor.shape[3].value

    out_channel_vector_length = 64 if output_tensor.dtype == "float16" else 128
    out_channel = kernel_tensor.shape[3].value

    if out_channel % out_channel_vector_length != 0:
        new_out_channel = (
            (out_channel + out_channel_vector_length) // out_channel_vector_length
        ) * out_channel_vector_length
        diff = new_out_channel - out_channel
        kernel = relay.nn.pad(kernel, pad_width=((0, 0), (0, 0), (0, 0), (0, diff)))

        new_attrs["channels"] = new_out_channel
        out = relay.nn.conv2d(data, kernel, **new_attrs)
        original_out_shape = [x.value for x in output_tensor.shape]
        return relay.strided_slice(out, begin=[0, 0, 0, 0], end=original_out_shape)
    else:
        return relay.nn.conv2d(data, kernel, **new_attrs)
