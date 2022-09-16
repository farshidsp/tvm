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
    return None
    print("SOMETHING")
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

    # Flags to remember if the expr is modified
    ic_modified = False
    oc_modified = False
    
    # TODO: pad on input channel?
    # Find the value of input and output channel.
    in_channel = -1
    out_channel = -1
    if attrs["data_layout"] == "NHWC" and attrs["kernel_layout"] == "HWIO":
        in_channel = data_tensor.shape[3].value
        out_channel = kernel_tensor.shape[3].value
    elif attrs["data_layout"] == "NCHW" and attrs["kernel_layout"] == "OIHW":
        in_channel = data_tensor.shape[1].value
        out_channel = kernel_tensor.shape[0].value
    else:
        return None        
    in_channel_vector_length = 4

    if in_channel % in_channel_vector_length != 0:
        new_in_channel = (
            (in_channel + in_channel_vector_length) // in_channel_vector_length
        ) * in_channel_vector_length
        diff = new_in_channel - in_channel
        if attrs["data_layout"] == "NHWC" and attrs["kernel_layout"] == "HWIO":
            data = relay.nn.pad(data, pad_width=((0, 0), (0, 0), (0, 0), (0, diff)))
            kernel = relay.nn.pad(kernel, pad_width=((0, 0), (0, 0), (0, diff), (0, 0)))
            ic_modified = True
        elif attrs["data_layout"] == "NCHW" and attrs["kernel_layout"] == "OIHW":
            pad_width = ((0, 0), (0, diff), (0, 0), (0, 0))
            data = relay.nn.pad(data, pad_width=pad_width)
            kernel = relay.nn.pad(kernel, pad_width=pad_width)
            ic_modified = True
        else:
            return None
    

    out_channel_vector_length = 64 if output_tensor.dtype == "float16" else 128
    if out_channel % out_channel_vector_length != 0:
        new_out_channel = (
            (out_channel + out_channel_vector_length) // out_channel_vector_length
        ) * out_channel_vector_length
        diff = new_out_channel - out_channel
        if attrs["data_layout"] == "NHWC" and attrs["kernel_layout"] == "HWIO":
            kernel = relay.nn.pad(kernel, pad_width=((0, 0), (0, 0), (0, 0), (0, diff)))
            oc_modified = True
        elif attrs["data_layout"] == "NCHW" and attrs["kernel_layout"] == "OIHW":
            kernel = relay.nn.pad(kernel, pad_width=((0, diff), (0, 0), (0, 0), (0, 0)))
            oc_modified = True
        else:
            return None
    if oc_modified:
        print("SOMETHING HAS CHANGED!")
        new_attrs["channels"] = new_out_channel
        out = relay.nn.conv2d(data, kernel, **new_attrs)
        original_out_shape = [x.value for x in output_tensor.shape]
        out = relay.strided_slice(out, begin=[0, 0, 0, 0], end=original_out_shape)
    else:
        out = relay.nn.conv2d(data, kernel, **new_attrs)
    return out

