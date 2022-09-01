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

"""Schedule for dense operator"""

import tvm
from tvm.topi.utils import get_const_tuple, traverse_inline
from tvm import te
from .. import tag


def schedule_dense(outs):
    """Schedule for dense op.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of dense in the format
        of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.te.tensor.Tensor) else outs
    s = tvm.te.create_schedule([x.op for x in outs])
    tvm.te.schedule.AutoInlineInjective(s)
    return s


def dense_pack(data, weight, bias=None, out_dtype=None):
    """Compute dense with transformed weight."""
    M, K = get_const_tuple(data.shape)  # batch, in_dim
    N, _, packw_bn = get_const_tuple(weight.shape)  # out_dim
    N = N * packw_bn

    packw = weight

    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda y, x: te.sum(
            data[y, k].astype(out_dtype)
            * packw[idxdiv(x, packw_bn), k, idxmod(x, packw_bn)].astype(out_dtype),
            axis=k,
        ),
        tag="dense_pack",
    )
    if bias is not None:
        C = te.compute((M, N), lambda i, j: C[i, j] + bias[j].astype(out_dtype), tag=tag.BROADCAST)
    return C


def _schedule_dense_pack_template(s, C, O):
    A, packedB = s[C].op.input_tensors

    y, x = s[C].op.axis

    xo, xi = s[C].split(x, factor=packedB.shape[-1])
    s[C].reorder(y, xo, xi)
    # if C == O:
    #     s[C].parallel(xyt)
    s[C].vectorize(xi)

    if C != O:
        y, x = s[O].op.axis
        xo, xi = s[O].split(x, factor=packedB.shape[-1])
        s[O].reorder(y, xo, xi)

        s[C].compute_at(s[O], xo)
        s[O].vectorize(xi)
        # s[O].parallel(xyt)
    return s


def schedule_dense_pack(outs):
    """Create the schedule for dense_nopack"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "dense_pack" in op.tag:
            _schedule_dense_pack_template(s, op.output(0), outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s
