import tvm
from tvm import te
import tvm.testing
import numpy as np
import tvm.topi.testing
from tvm.topi.utils import get_const_tuple, traverse_inline
from tvm.topi.nn.utils import get_pad_tuple
from tvm.topi.nn.pad import pad
from tvm.topi.arm_cpu.tensor_intrin import dot_int8_int8_int32_neon_82


pytest_plugins = [
    "tvm.contrib.hexagon.pytest_plugin",
]


def conv2d_NCHWc_int8(
    data, kernel, stride, padding, dilation, out_dtype="int32", n_elems=32
):
    # layout and out_layout are not used here,
    # we keep them for debug convenience when dumping autotvm workload
    HSTR, WSTR = stride if isinstance(stride, (tuple, list)) else (stride, stride)
    dilation_h, dilation_w = (
        dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation)
    )

    n, ic_chunk, ih, iw, ic_bn = get_const_tuple(data.shape)
    in_channel = ic_chunk * ic_bn
    oc_chunk, ic_chunk_group, kernel_height, kernel_width, _, oc_bn, _ = get_const_tuple(
        kernel.shape
    )
    groups = ic_chunk // ic_chunk_group

    dilated_kernel_h = (kernel_height - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_width - 1) * dilation_w + 1

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    HPAD = pad_top + pad_down
    WPAD = pad_left + pad_right

    # output shape
    out_height = (ih + HPAD - dilated_kernel_h) // HSTR + 1
    out_width = (iw + WPAD - dilated_kernel_w) // WSTR + 1
    oshape = (n, oc_chunk, out_height, out_width, oc_bn)
    pad_before = (0, 0, pad_top, pad_left, 0)
    pad_after = (0, 0, pad_down, pad_right, 0)

    # DOPAD
    DOPAD = HPAD != 0 or WPAD != 0
    if DOPAD:
        data_pad = pad(data, pad_before, pad_after, name="data_pad")
    else:
        data_pad = data

    kh = te.reduce_axis((0, kernel_height), name="kh")
    kw = te.reduce_axis((0, kernel_width), name="kw")

    ic_outer = te.reduce_axis((0, in_channel // ic_bn), name="ic_outer")
    ic_f_inner = te.reduce_axis((0, ic_bn // n_elems), name="ic_f_inner")
    ic_s_inner = te.reduce_axis((0, n_elems), name="ic_s_inner")
    return te.compute(
        oshape,
        lambda n, oc_chunk, oh, ow, oc_block: te.sum(
            data_pad[
                n,
                ic_outer,
                oh * HSTR + kh * dilation_h,
                ow * WSTR + kw * dilation_w,
                ic_f_inner * n_elems + ic_s_inner,
            ].astype(out_dtype)
            * kernel[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block, ic_s_inner].astype(
                out_dtype
            ),
            axis=[kh, kw, ic_outer, ic_f_inner, ic_s_inner],
        ),
        name="conv2d_NCHWc_int8",
        tag="conv2d_NCHWc_int8",
        attrs={"schedule_rule": "meta_schedule.conv2d_NCHWc_int8"},
    )


def schedule_conv2d_NCHWc_int8(outs):
    """Create schedule for tensors"""
    s = te.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def _callback(op):
        if "conv2d_NCHWc_int8" in op.tag:
            conv_out = op.output(0)
            kernel_vec = conv_out.op.input_tensors[1]
            data_vec = conv_out.op.input_tensors[0]
            data = (
                data_vec.op.input_tensors[0]
                if isinstance(data_vec.op, te.tensor.ComputeOp) and "pad" not in data_vec.op.tag
                else data_vec
            )
            if isinstance(data.op, te.tensor.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]

            args = [s, data_vec, kernel_vec, conv_out, outs[0]]
            # int8 conv kernel is 7-dim
            _, _, kh, kw, _, _, n_elems = get_const_tuple(kernel_vec.shape)
            assert n_elems == 4
            dtype = "uint" if data.dtype == "uint8" else "int"
            intrin = dot_int8_int8_int32_neon_82(int32_lanes=4, dtype=dtype)

            inline_fused = True

            if kh == 1 and kw == 1:
                schedule_conv_NCHWc_cpu_1x1_int8(
                    *args, int32_lanes=4, int8_elems=4, intrin=intrin, inline_fused=inline_fused
                )
            else:
                schedule_conv_NCHWc_cpu_common_int8(
                    *args, int32_lanes=4, int8_elems=4, intrin=intrin, inline_fused=inline_fused
                )

    traverse_inline(s, outs[0].op, _callback)
    return s


def schedule_conv_NCHWc_cpu_common_int8(
    s,
    data_vec,
    kernel_vec,
    conv_out,
    last,
    int32_lanes=16,
    int8_elems=4,
    intrin=None,
    inline_fused=True,
):
    reg_n = 4 # TODO
    unroll_kw = False
    _, _, _, _, ic_bn = get_const_tuple(data_vec.shape)
    _, _, _, _, oc_bn = get_const_tuple(conv_out.shape)

    # schedule pad
    if isinstance(s[data_vec].op, te.tensor.ComputeOp) and "pad" in data_vec.op.tag:
        batch, ic_chunk, ih, iw, ic_block = s[data_vec].op.axis
        parallel_axis = s[data_vec].fuse(batch, ic_chunk, ih)
        s[data_vec].parallel(parallel_axis)
        data_vec = data_vec.op.input_tensors[0]

    # schedule 5-D NCHW[x]c conv
    C, O = conv_out, last
    CC = s.cache_write(C, "global")

    batch, oc_chunk, oh, ow, oc_block = s[C].op.axis
    ow_chunk, ow_block = s[C].split(ow, factor=reg_n)
    s[C].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
    parallel_axis = s[C].fuse(batch, oc_chunk, oh)
    s[C].vectorize(oc_block)
    if C == O:
        s[C].parallel(parallel_axis)

    s[CC].compute_at(s[C], parallel_axis)
    _, oc_chunk, oh, ow, oc_block = s[CC].op.axis
    kh, kw, ic_outer, ic_f_inner, ic_s_inner = s[CC].op.reduce_axis

    ow_chunk, ow_block = s[CC].split(ow, factor=reg_n)

    assert oc_bn % int32_lanes == 0, f"oc_bn={oc_bn} % int32_lanes={int32_lanes} != 0"
    assert (
        ic_bn % int8_elems == 0
    ), f"ic_bn={ic_bn} % int8_elems={int8_elems} != 0"  # (u)int8 elements in (u)int32

    oc_f_inner, oc_s_inner = s[CC].split(oc_block, factor=int32_lanes)

    if unroll_kw:
        s[CC].reorder(
            oc_chunk,
            oh,
            ow_chunk,
            ic_outer,
            kh,
            ic_f_inner,
            kw,
            ow_block,
            oc_f_inner,
            oc_s_inner,
            ic_s_inner,
        )
        s[CC].unroll(kw)
    else:
        s[CC].reorder(
            oc_chunk,
            oh,
            ow_chunk,
            ic_outer,
            kh,
            kw,
            ic_f_inner,
            ow_block,
            oc_f_inner,
            oc_s_inner,
            ic_s_inner,
        )

    if intrin is not None:
        s[CC].tensorize(oc_s_inner, intrin)
    s[CC].unroll(ow_block)
    s[CC].unroll(oc_f_inner)

    if C != O:
        out_ndim = len(s[O].op.axis)
        if out_ndim == 5:
            batch, oc_chunk, oh, ow, oc_block = s[O].op.axis
            ow_chunk, ow_block = s[O].split(ow, factor=reg_n)
            s[O].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
        elif out_ndim == 4:
            batch, oc, oh, ow = s[O].op.axis
            ow_chunk, ow_block = s[O].split(ow, factor=reg_n)
            oc_chunk, oc_block = s[O].split(oc, factor=oc_bn)
            s[O].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
        else:
            raise ValueError("Unsupported output ndim: %s" % out_ndim)
        parallel_axis = s[O].fuse(batch, oc_chunk, oh)
        if inline_fused:
            s[C].compute_at(s[O], ow_block)
        else:
            s[C].compute_at(s[O], parallel_axis)
        s[O].vectorize(oc_block)
        s[O].parallel(parallel_axis)

    return s


ic_bn = 4
oc_bn = 4
n_elems = 4

data = te.placeholder((1, 64 // ic_bn, 56, 56, ic_bn), name="data", dtype="uint8")
kernel = te.placeholder((64 // oc_bn, 64 // ic_bn, 3, 3, ic_bn // n_elems, oc_bn, n_elems), name="data", dtype="uint8")

strides = (1, 1)
padding = (1, 1)
dilation = (1, 1)

out = conv2d_NCHWc_int8(
    data, kernel, strides, padding, dilation, out_dtype="int32", n_elems=32
)

schedule_conv2d_NCHWc_int8([out])
