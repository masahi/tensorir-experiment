import tvm
from tvm import te
import tvm.testing
import numpy as np
import tvm.topi.testing
from tvm.topi.utils import get_const_tuple, traverse_inline
from tvm.topi.nn.utils import get_pad_tuple
from tvm.topi.nn.pad import pad
from tvm.topi.arm_cpu.tensor_intrin import dot_int8_int8_int32_neon_82
from tvm.topi.hexagon.dense import dot_u8u8i32_vrmpy
from tvm.topi.transform import layout_transform

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

            args = [s, data_vec, conv_out, outs[0]]
            # int8 conv kernel is 7-dim
            _, _, kh, kw, _, _, n_elems = get_const_tuple(kernel_vec.shape)
            # assert n_elems == 4
            dtype = "uint" if data.dtype == "uint8" else "int"
            intrin = dot_u8u8i32_vrmpy()

            inline_fused = True

            out_width = conv_out.shape[3]
            reg_n = 1
            for n in range(31, 0, -1):
                if out_width % n == 0:
                    reg_n = n
                    break

            if kh == 1 and kw == 1:
                schedule_conv_NCHWc_cpu_1x1_int8(
                    *args, int32_lanes=32, int8_elems=4, reg_n=reg_n, intrin=intrin, inline_fused=inline_fused
                )
            else:
                schedule_conv_NCHWc_cpu_common_int8(
                    *args, int32_lanes=32, int8_elems=4, reg_n=reg_n, intrin=intrin, inline_fused=inline_fused
                )

    traverse_inline(s, outs[0].op, _callback)
    return s


def schedule_conv_NCHWc_cpu_common_int8(
    s,
    data_vec,
    conv_out,
    last,
    reg_n,
    int32_lanes=32,
    int8_elems=4,
    intrin=None,
    inline_fused=True,
):
    unroll_kw = False
    _, _, _, _, ic_bn = get_const_tuple(data_vec.shape)
    _, _, _, _, oc_bn = get_const_tuple(conv_out.shape)

    # schedule pad
    if isinstance(s[data_vec].op, te.tensor.ComputeOp) and "pad" in data_vec.op.tag:
        batch, ic_chunk, ih, iw, ic_block = s[data_vec].op.axis
        parallel_axis = s[data_vec].fuse(batch, ic_chunk, ih)
        # s[data_vec].parallel(parallel_axis)
        data_vec = data_vec.op.input_tensors[0]

    # schedule 5-D NCHW[x]c conv
    C, O = conv_out, last
    CC = s.cache_write(C, "global")

    batch, oc_chunk, oh, ow, oc_block = s[C].op.axis
    ow_chunk, ow_block = s[C].split(ow, factor=reg_n)
    s[C].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
    parallel_axis = s[C].fuse(batch, oc_chunk, oh)
    s[C].vectorize(oc_block)

    # if C == O:
    #     s[C].parallel(parallel_axis)

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

    s[CC].tensorize(oc_s_inner, intrin)

    s[CC].unroll(ow_block)
    s[CC].unroll(oc_f_inner)

    if C != O:
        batch, oc_chunk, oh, ow, oc_block = s[O].op.axis
        ow_chunk, ow_block = s[O].split(ow, factor=reg_n)
        s[O].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
        parallel_axis = s[O].fuse(batch, oc_chunk, oh)

        if inline_fused:
            s[C].compute_at(s[O], ow_block)
        else:
            s[C].compute_at(s[O], parallel_axis)
        s[O].vectorize(oc_block)
        # s[O].parallel(parallel_axis)

    return s


@tvm.testing.requires_hexagon
def test_conv2d_u8u8i32_vrmpy(hexagon_session):
    dtype = "uint8"
    ic_bn = 32
    oc_bn = 32
    n_elems = 4
    I = 64
    O = 64
    H = 56
    W = 56
    kH = 3
    kW = 3

    data_packed = te.placeholder((1, I // ic_bn, H, W, ic_bn), name="data", dtype=dtype)
    kernel_packed = te.placeholder((O // oc_bn, I // ic_bn, kH, kW, ic_bn // n_elems, oc_bn, n_elems), dtype=dtype)

    strides = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)

    out = conv2d_NCHWc_int8(
        data_packed, kernel_packed, strides, padding, dilation, out_dtype="int32", n_elems=n_elems
    )

    s = schedule_conv2d_NCHWc_int8([out])

    target_hexagon = tvm.target.hexagon("v68", link_params=True)
    target = tvm.target.Target(target_hexagon, host=target_hexagon)

    f = tvm.build(s, [data_packed, kernel_packed, out], target)

    module = hexagon_session.load_module(f)

    a_np = np.random.randint(low=0, high=100, size=(1, I, H, W)).astype("int32")
    w_np = np.random.randint(low=0, high=100, size=(O, I, kH, kW)).astype("int32")
    c_np = tvm.topi.testing.conv2d_nchw_python(a_np, w_np, strides, padding)

    packed_data_np = np.zeros(get_const_tuple(data_packed.shape)).astype(dtype)
    packed_w_np = np.zeros(get_const_tuple(kernel_packed.shape)).astype(dtype)

    for i in range(I):
        for h in range(H):
            for w in range(W):
                packed_data_np[0, i // ic_bn, h, w, i % ic_bn] = a_np[0, i, h, w]

    for o in range(O):
        for i in range(I):
            for h in range(kH):
                for w in range(kW):
                    packed_w_np[o // oc_bn, i // ic_bn, h, w, (i % ic_bn) // n_elems, o % oc_bn, i % n_elems] = w_np[o, i, h, w]

    dev = hexagon_session.device

    a = tvm.nd.array(packed_data_np.astype(dtype), dev)
    w = tvm.nd.array(packed_w_np.astype(dtype), dev)

    c = tvm.nd.array(np.zeros(get_const_tuple(out.shape), dtype=out.dtype), dev)

    module(a, w, c)

    P, Q = c_np.shape[2:4]

    evaluator = module.time_evaluator(module.entry_name, dev, number=20)
    time_ms = evaluator(a, w, c).mean * 1e3
    gflops = (O * P * Q * I * kH * kW) * 2 / 1e9
    print("time elapsed: ", time_ms)
    print("GOPS:", gflops / (time_ms / 1e3))

    out_packed = c.numpy()

    out = np.zeros(c_np.shape).astype("int32")

    for o in range(O):
        for h in range(P):
            for w in range(Q):
                out[0, o, h, w] = out_packed[0, o // oc_bn, h, w, o % oc_bn]

    np.testing.assert_equal(out, c_np)


test_conv2d_u8u8i32_vrmpy(None)
