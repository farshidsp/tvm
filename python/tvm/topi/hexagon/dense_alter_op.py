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


@nn.dense_legalize.register("hexagon")
def _dense_legalize(attrs, inputs, arg_types):
    new_attrs = {k: attrs[k] for k in attrs.keys()}
    # Collect the input tensors.
    x_tensor, y_tensor = arg_types[0], arg_types[1]
    dtype = x_tensor.dtype

    # Collect the output tensor.
    output_tensor = arg_types[2]

    # Collect the input exprs.
    x, y = inputs

    M, K = x_tensor.shape
    N, K = y_tensor.shape
    try:
        M = M.value
        K = K.value
        N = N.value
    except AttributeError:
        # todo: deal with unfixed shape when compiling wdl model
        return None

    # vec_len = 1024 //
    if dtype == "float16":
        vec_len = 64
    elif "int8" in dtype:
        vec_len = 128

    if N % vec_len != 0:
        N_padded = ((N + vec_len) // vec_len) * 64
        dn = N_padded - N

        y_ = relay.nn.pad(y, pad_width=((0, dn), (0, 0)))

        # If units is explicitly specified, it is used to compute the output shape.
        # We need to update units after padding to prevent a type error.
        if attrs["units"] is not None:
            new_attrs["units"] = N + dn

        out_ = relay.nn.dense(x, y_, **new_attrs)
        out =  relay.strided_slice(out_, begin=[0, 0], end=[x.value for x in output_tensor.shape])
        return out

    return None
